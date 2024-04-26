import numpy as np
import tensorflow as tf
import time

from sac_eo.common.buffers import TrajectoryBuffer
from sac_eo.common.buffer_utils import aggregate_data
from sac_eo.common.samplers import batch_simtrajectory_sampler

from sac_eo.algs.base_onpolicy_alg import BaseOnPolicyAlg

class MBRLOnPolicyAlg(BaseOnPolicyAlg):
    """Base class for MBRL algorithms with on-policy updates."""

    def __init__(self,idx,env,env_eval,actor,critics,models,alg_kwargs,
        mf_update_kwargs):
        """Initializes MBRLOnPolicyAlg class. See BaseOnPolicyAlg for detais.
        
        Args:
            models (list): ensemble of learned dynamics models
        """
        super(MBRLOnPolicyAlg,self).__init__(idx,env,env_eval,actor,critics,
            alg_kwargs,mf_update_kwargs)
        
        self.models = models
        self.B = len(self.models)
        self.sim_batch_size_per_model = int(self.sim_batch_size / self.B)

        self.models_trainable = []
        for model in self.models:
            self.models_trainable = self.models_trainable + model.trainable

        self.sim_data = [TrajectoryBuffer(self.s_dim,self.a_dim,self.gamma,
            self.lam,self.sim_buffer_size) for _ in range(self.B)]

    def _setup(self,alg_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(MBRLOnPolicyAlg,self)._setup(alg_kwargs)

        self.sim_buffer_size = alg_kwargs['sim_buffer_size']
        if self.sim_buffer_size:
            self.sim_buffer_size = int(self.sim_buffer_size)

        self.sim_horizon = alg_kwargs['sim_horizon']
        self.sim_batch_type = alg_kwargs['sim_batch_type']
        self.sim_batch_size = alg_kwargs['sim_batch_size']

        self.model_lr = alg_kwargs['model_lr']
        self.model_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_lr)
        self.reset_model_optimizer=alg_kwargs['reset_model_optimizer']

        self.model_num_epochs = alg_kwargs['model_num_epochs']
        self.model_batch_size = alg_kwargs['model_batch_size']
        self.model_batch_shuffle = alg_kwargs['model_batch_shuffle']
        
        self.model_max_updates = alg_kwargs['model_max_updates']
        self.model_max_grad_norm = alg_kwargs['model_max_grad_norm']
        self.model_holdout_ratio = alg_kwargs['model_holdout_ratio']
        self.model_holdout_epochs = alg_kwargs['model_holdout_epochs']

    def _set_rms(self):
        """Shares normalizers with all relevant classes."""
        super(MBRLOnPolicyAlg,self)._set_rms()
        for model in self.models:
            model.set_rms(self.normalizer)

    def _update(self):
        """Updates actor, critic, and models."""
        self._update_models()
        self._update_actor_critic()

    def _collect_sim_data(self,model,buffer):
        """"Collects simulated data from given learned dynamics models."""
        steps_start = buffer.steps_total
        traj_start = buffer.traj_total
        batch_size_cur = 0
        while batch_size_cur < self.sim_batch_size_per_model:
            remaining = self.sim_batch_size_per_model - batch_size_cur
            if self.sim_batch_type == 'steps':
                num_traj = int(remaining / self.sim_horizon)
                horizon = self.sim_horizon
                if num_traj == 0:
                    num_traj = 1
                    horizon = remaining
            else:
                num_traj = remaining
                horizon = self.sim_horizon
                            
            s_init = np.squeeze(
                self.env_data.get_states(batch_size=num_traj))
            rollout_batch = batch_simtrajectory_sampler(
                model,self.actor,horizon,s_init)
            buffer.add_batch(*rollout_batch)

            if self.sim_batch_type == 'steps':
                steps_cur = buffer.steps_total
                batch_size_cur = steps_cur - steps_start
            else:
                traj_cur = buffer.traj_total
                batch_size_cur = traj_cur - traj_start

    def _collect_sim_data_all(self,models):
        """"Collects simulated data from all learned dynamics models."""
        for idx in range(self.B):
            model = models[idx]
            buffer = self.sim_data[idx]
            buffer.reset()
            self._collect_sim_data(model,buffer)

    def _get_rollout_data(self):
        """"Returns rollout data for policy learning."""
        rollout_data_list = []
        for idx in range(self.B):
            buffer = self.sim_data[idx]
            if self.num_critics == self.B:
                critic = self.critics[idx]
            else:
                critic = self.critics[0]
            rollout_data_active = buffer.get_update_info(critic)
            rollout_data_list.append(rollout_data_active)
        
        return rollout_data_list

    def _update_actor_critic(self):
        """Updates actor and critic."""
        time_ac_start = time.time()

        env_states = self.env_data.get_states()
        ent = tf.reduce_mean(self.actor.entropy(env_states))
        kl_info = self.actor.get_kl_info(env_states)
        
        for _ in range(self.num_mf_updates):
            time_sim_data_start = time.time()
            self._collect_sim_data_all(self.models)
            rollout_data_list = self._get_rollout_data()
            rollout_data_all = tuple(zip(*rollout_data_list))
            rollout_data_agg = tuple(map(aggregate_data,zip(*rollout_data_list)))
            steps_update = len(rollout_data_agg[0])
            time_sim_data_end = time.time()
            time_sim_data = time_sim_data_end - time_sim_data_start
            
            time_critic_start = time.time()
            log_critics = self._update_critics(rollout_data_all)
            time_critic_end = time.time()
            time_critic = time_critic_end - time_critic_start
            
            time_actor_start = time.time()
            log_actor = self._update_actor(rollout_data_agg)
            time_actor_end = time.time()
            time_actor = time_actor_end - time_actor_start

            log_update = {
                'steps_update':     steps_update,
                'time_actor':       time_actor,
                'time_critic':      time_critic,
                'time_sim_data':    time_sim_data
            }

            self.logger.log_train(log_update)
            self.logger.log_train(log_actor)
            self.logger.log_train_ensemble(log_critics)

        time_ac_end = time.time()
        time_ac_agg = time_ac_end - time_ac_start

        kl = tf.reduce_mean(self.actor.kl(env_states,kl_info))

        log_actor_agg = {
            'ent_agg':      ent.numpy(),
            'kl_agg':       kl.numpy(),
            'alpha_agg':    self.mf_algo.alpha.numpy(),
            'time_ac_agg':  time_ac_agg
        }
        self.logger.log_train(log_actor_agg)

    def _update_models(self):
        """Updates learned dynamics models."""
        time_start = time.time()
        
        s_all, a_all, sp_all, r_all = self.env_data.get_model_info()

        ent_all = []
        for model in self.models:
            ent = tf.reduce_mean(model.entropy(s_all,a_all))
            ent_all.append(ent.numpy())
        ent_all = np.array(ent_all)
        
        if self.model_holdout_ratio > 0.0:
            num_data_points_total = len(r_all)
            num_data_points_train = int(
                num_data_points_total*(1-self.model_holdout_ratio))

            idx = np.arange(num_data_points_total)
            np.random.shuffle(idx)
            train_idx = idx[:num_data_points_train]
            holdout_idx = idx[num_data_points_train:]

            s_train = s_all[train_idx]
            a_train = a_all[train_idx]
            sp_train = sp_all[train_idx]
            r_train = r_all[train_idx]

            s_holdout = s_all[holdout_idx]
            a_holdout = a_all[holdout_idx]
            sp_holdout = sp_all[holdout_idx]
            r_holdout = r_all[holdout_idx]
        else:
            num_data_points_train = len(r_all)
            s_train, a_train, sp_train, r_train = s_all, a_all, sp_all, r_all

        holdout_best = np.inf
        epochs_since_best = 0        
        num_updates = 0
        # Minibatch update loop for models
        for ep in range(self.model_num_epochs):
            idx = np.arange(num_data_points_train)
            if self.model_batch_shuffle:
                idx = np.tile(idx,(self.B,1))
                for model_idx in idx:
                    np.random.shuffle(model_idx)
            else:
                np.random.shuffle(idx)
                idx = np.tile(idx,(self.B,1))

            sections = np.arange(0,num_data_points_train,
                self.model_batch_size)[1:]

            batches = np.array_split(idx,sections,axis=1)
            if (num_data_points_train % self.model_batch_size != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                s_batch = s_train[batch_idx]
                a_batch = a_train[batch_idx]
                sp_batch = sp_train[batch_idx]
                r_batch = r_train[batch_idx]
                
                self._apply_model_grads(s_batch,sp_batch,a_batch,r_batch)

                num_updates += 1
                if num_updates >= self.model_max_updates:
                    break

            if num_updates >= self.model_max_updates:
                break

            if self.model_holdout_ratio > 0.0:
                holdout_loss_all = 0.0
                for model in self.models:
                    holdout_loss = model.get_loss(
                        s_holdout,sp_holdout,a_holdout,r_holdout)
                    holdout_loss_all += holdout_loss
                
                if holdout_loss_all < holdout_best:
                    holdout_best = holdout_loss_all
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                
                if epochs_since_best >= self.model_holdout_epochs:
                    break
                        
        if self.reset_model_optimizer == True:
            self.model_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_lr)
        
        log_model_fit_all = []
        for model in self.models:
            loss_train = model.get_loss(s_train,sp_train,a_train,r_train)
            if self.model_holdout_ratio > 0.0:
                loss_holdout = model.get_loss(
                    s_holdout,sp_holdout,a_holdout,r_holdout)
            else:
                loss_holdout = tf.constant(np.inf)
            loss_all = model.get_loss(s_all,sp_all,a_all,r_all)
            loss_mse, loss_reward = model.get_losses_eval(
                s_all,sp_all,a_all,r_all)

            log_model_fit = {
                'model_loss':           loss_all.numpy(),
                'model_loss_train':     loss_train.numpy(),
                'model_loss_holdout':   loss_holdout.numpy(),
                'model_loss_mse':       loss_mse.numpy(),
                'model_loss_reward':    loss_reward.numpy(),
            }
            log_model_fit_all.append(log_model_fit)
        
        time_end = time.time()
        time_model_fit = time_end - time_start

        log_model_fit_agg = {
            'time_model_fit':       time_model_fit,
            'model_ent':            ent_all,
            'model_loss_epochs':    ep + 1
        }
            
        self.logger.log_train(log_model_fit_agg)
        self.logger.log_train_ensemble(log_model_fit_all)

    # @tf.function
    def _apply_model_grads(self,s_batch,sp_batch,a_batch,r_batch):
        """Applies model gradients."""
        with tf.GradientTape() as tape:
            loss_all = 0.0
            for idx, model in enumerate(self.models):
                s_active = s_batch[idx]
                sp_active = sp_batch[idx]
                a_active = a_batch[idx]
                r_active = r_batch[idx]
            
                loss = model.get_loss(s_active,sp_active,a_active,r_active)
                loss_all += loss
        
        grads = tape.gradient(loss_all,self.models_trainable)
        if self.model_max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(
                grads,self.model_max_grad_norm*self.B)
        
        self.model_optimizer.apply_gradients(zip(grads, self.models_trainable))

    def _dump_stats(self):
        final = super(MBRLOnPolicyAlg,self)._dump_stats()
        
        # Model weights
        final['model_weights'] = [model.get_weights() for model in self.models]
        final['reward_weights'] = [model.get_reward_weights() 
            for model in self.models]

        return final
