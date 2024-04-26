import numpy as np
import tensorflow as tf
import gym
import time

from sac_eo.common.normalizer import RunningNormalizers
from sac_eo.common.buffers import TrajectoryBuffer
from sac_eo.common.buffer_utils import aggregate_data
from sac_eo.common.samplers import trajectory_sampler
from sac_eo.common.corruptor import TrajectoryCorruptor
from sac_eo.common.logger import Logger
from sac_eo.algs.model_free import TRPO, PPO

class BaseOnPolicyAlg:
    """Base class for algorithms with on-policy updates."""

    def __init__(self,idx,env,env_eval,actor,critics,alg_kwargs,
        mf_update_kwargs):
        """Initializes BaseOnPolicyAlg class.

        Args:
            idx (int): index for checkpoint file
            env (object): normalized environment
            env_eval (object): normalized environment for evaluation
            actor (object): policy
            critics (list): list of value functions
            alg_kwargs (dict): algorithm parameters
            mf_update_kwargs (dict): model free update parameters
        """
        self._setup(alg_kwargs)
        
        self.env = env
        self.env_eval = env_eval
        self.actor = actor
        self.critics = critics

        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        self.normalizer = RunningNormalizers(self.s_dim,self.a_dim,self.gamma,
            self.init_rms_stats)
        
        self.B = 1
        self.num_critics = len(self.critics)
        self.critic_trainable = []
        for critic in self.critics:
            self.critic_trainable = self.critic_trainable + critic.trainable
        
        self.env_data = TrajectoryBuffer(self.s_dim,self.a_dim,self.gamma,
            self.lam,self.env_buffer_size)
        
        self.corruptor = TrajectoryCorruptor(self.s_noise_std,self.s_noise_type)
        
        mf_update_kwargs['ent_targ'] = self.env_data.a_dim * -1
        if self.mf_algo_name == 'trpo':
            self.mf_algo = TRPO(self.actor,mf_update_kwargs)
        elif self.mf_algo_name == 'ppo':
            self.mf_algo = PPO(self.actor,mf_update_kwargs)
        else:
            raise ValueError('invalid mf_algo')
        
        self.logger = Logger()
        self.checkpoint_name = '%s_%d'%(self.checkpoint_file,idx)
        self.current_reward=0
    
    def _setup(self,alg_kwargs):
        """Sets up hyperparameters as class attributes.
        
        Args:
            alg_kwargs (dict): dictionary of hyperparameters
        """
        
        self.gamma = alg_kwargs['gamma']
        self.lam = alg_kwargs['lam']
        self.exp_state_ratio=alg_kwargs['max_exp_state_ratio']

        self.env_buffer_size = alg_kwargs['env_buffer_size']
        if self.env_buffer_size:
            self.env_buffer_size = int(self.env_buffer_size)
        
        self.init_rms_stats = alg_kwargs['init_rms_stats']

        self.save_path = alg_kwargs['save_path']
        self.checkpoint_file = alg_kwargs['checkpoint_file']
        self.save_freq = alg_kwargs['save_freq']

        self.last_eval = 0
        self.eval_freq = alg_kwargs['eval_freq']
        self.eval_num_traj = alg_kwargs['eval_num_traj']
        
        self.mf_algo_name = alg_kwargs['mf_algo']

        self.env_horizon = alg_kwargs['env_horizon']
        self.env_batch_type = alg_kwargs['env_batch_type']
        self.env_batch_size_init = alg_kwargs['env_batch_size_init']
        self.env_batch_size = alg_kwargs['env_batch_size']

        self.s_noise_std = alg_kwargs['s_noise_std']
        self.s_noise_type = alg_kwargs['s_noise_type']

        self.critic_lr = alg_kwargs['critic_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)
        self.critic_update_it = alg_kwargs['critic_update_it']
        self.critic_nminibatch = alg_kwargs['critic_nminibatch']

        self.num_mf_updates = alg_kwargs['num_mf_updates']
        self.alg_seed=alg_kwargs['alg_seed']
        self.rng = np.random.default_rng(self.alg_seed)

    
    def _collect_expert_data(self):
        return None

    def _collect_env_data(self,num_timesteps,update_normalizers=True,only_model_normalizer=False):
        time_start = time.time()
        if num_timesteps == 0:
            batch_size = self.env_batch_size_init
        else:
            batch_size = self.env_batch_size

        steps_start = self.env_data.steps_total
        traj_start = self.env_data.traj_total

        batch_size_cur = 0
        J_tot_all = []
        while batch_size_cur < batch_size:
            if self.env_batch_type == 'steps':
                horizon = np.minimum(batch_size-batch_size_cur,self.env_horizon)
            else:
                horizon = self.env_horizon

            traj_and_J = trajectory_sampler(
                self.env,self.actor,horizon,eval=True,corruptor=self.corruptor)
            s_traj, a_traj, r_traj, sp_traj, d_traj, J_tot = traj_and_J
            if update_normalizers == True:
                if only_model_normalizer == True:
                    self.model_normalizer.update_rms(s_traj,a_traj,r_traj,sp_traj)
                else:
                    self.normalizer.update_rms(s_traj,a_traj,r_traj,sp_traj)
            self.env_data.add(s_traj,a_traj,r_traj,sp_traj,d_traj)

            if horizon == self.env_horizon:
                J_tot_all.append(J_tot)
            
            if self.env_batch_type == 'steps':
                steps_cur = self.env_data.steps_total
                batch_size_cur = steps_cur - steps_start
            else:
                traj_cur = self.env_data.traj_total
                batch_size_cur = traj_cur - traj_start

        steps_end = self.env_data.steps_total
        traj_end = self.env_data.traj_total
        
        J_tot_ave = np.mean(J_tot_all)
        steps_new = steps_end - steps_start
        traj_new = traj_end - traj_start

        time_end = time.time()
        time_env_data = time_end - time_start
        
        log_env_data = {
            'J_tot':            J_tot_ave,
            'steps':            steps_new,
            'traj':             traj_new,
            'time_env_data':    time_env_data
        }
        self.current_reward=J_tot_ave
        self.logger.log_train(log_env_data)

        return steps_new
    
    def _evaluate(self,num_timesteps):
        """Evaluates current policy."""
        time_start = time.time()

        J_tot_all = []
        for _ in range(self.eval_num_traj):
            traj_and_J = trajectory_sampler(self.env_eval,self.actor,
                self.env_horizon,eval=True,deterministic=True)
            J_tot = traj_and_J[-1]
            J_tot_all.append(J_tot)
        
        J_tot_ave = np.mean(J_tot_all)

        time_end = time.time()
        time_eval = time_end - time_start
        
        log_env_eval_data = {
            'J_tot_eval':   J_tot_ave,
            'steps_eval':   num_timesteps - self.last_eval,
            'time_eval':    time_eval    
        }
        self.logger.log_train(log_env_eval_data)

        self.last_eval = num_timesteps

    def _set_rms(self):
        """Shares normalizers with all relevant classes."""
        self.actor.set_rms(self.normalizer)
        for critic in self.critics:
            critic.set_rms(self.normalizer)
        self.corruptor.set_rms(self.normalizer)

    def _update(self):
        """Updates actor, critic, and models (if applicable)."""
        raise NotImplementedError
    
    def _update_actor(self,rollout_data):
        """Updates actor.
        
        Args:
            rollout_data (tuple): tuple of data collected to use for updates
        """
        log_actor = self.mf_algo.update(rollout_data)
        return log_actor

    def _update_critics(self,rollout_data_all):
        """Updates critics.
        
        Args:
            rollout_data_all (tuple): tuple of data collected to use for updates
        """
        
        s_all_list, _, _, rtg_all_list, _, _ = rollout_data_all
        if self.num_critics != self.B:
            s_all_list = [aggregate_data(s_all_list)]
            rtg_all_list = [aggregate_data(rtg_all_list)]

        n_samples = np.min([len(rtg_all) for rtg_all in rtg_all_list])
        n_batch = int(n_samples / self.critic_nminibatch)

        # Minibatch update loop for critics
        for _ in range(self.critic_update_it):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,n_batch)[1:]

            batches = np.array_split(idx,sections)
            if (n_samples % n_batch != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                # Active data
                s_active_list = [s_all[batch_idx] for s_all in s_all_list]
                rtg_active_list = [rtg_all[batch_idx] 
                    for rtg_all in rtg_all_list]

                # Critic update
                self._apply_critic_grads(s_active_list,rtg_active_list)
        
        log_critic_all = []
        for idx in range(len(self.critics)):
            critic = self.critics[idx]
            s_all = s_all_list[idx]
            rtg_all = rtg_all_list[idx]

            v_loss = critic.get_loss(s_all,rtg_all)

            log_critic = {
                'critic_loss':  v_loss.numpy(),
            }
            log_critic_all.append(log_critic)

        return log_critic_all

    # @tf.function
    def _apply_critic_grads(self,s_active_list,rtg_active_list):
        """Applies critic gradients."""

        with tf.GradientTape() as tape:
            v_loss_all = 0.0
            for idx in range(len(self.critics)):
                critic = self.critics[idx]
                s_active = s_active_list[idx]
                rtg_active = rtg_active_list[idx]

                v_loss = critic.get_loss(s_active,rtg_active)
                v_loss_all += v_loss
        
        vg = tape.gradient(v_loss_all,self.critic_trainable)
        self.critic_optimizer.apply_gradients(zip(vg,self.critic_trainable))

    def train(self,total_timesteps,params):
        """Training loop.

        Args:
            total_timesteps (int): number of true environment steps for training

        Returns:
            Name of checkpoint file.
        """

        self._set_rms()
        
        self._collect_expert_data()
        
        checkpt_idx = 0
        if self.save_freq is None:
            checkpoints = np.array([total_timesteps])
        else:
            checkpoints = np.concatenate(
                (np.arange(0,total_timesteps,self.save_freq)[1:],
                [total_timesteps]))

        eval_idx = 0
        report_idx = 0
        if self.eval_freq is None:
            evaluate = False 
        else:
            evaluate = True
            eval_points = np.concatenate(
                (np.arange(0,total_timesteps,self.eval_freq)[1:],
                [total_timesteps]))
            report_points = np.concatenate(
                (np.arange(0,total_timesteps,self.eval_freq/2)[1:],
                [total_timesteps]))

        # Training loop
        num_timesteps = 0
        if evaluate:
            self._evaluate(num_timesteps)
        while num_timesteps < total_timesteps:
            # Collect and store data from true environment
            steps_new = self._collect_env_data(num_timesteps)
            num_timesteps += steps_new

            # Update
            self.exp_state_prob=max(self.exp_state_ratio*(num_timesteps/total_timesteps),0.01)
            self._update()

            # Evaluate
            if evaluate:
                if num_timesteps >= eval_points[eval_idx]:
                    self._evaluate(num_timesteps)
                    eval_idx += 1

                if num_timesteps >= report_points[report_idx]:    
                    self.report_train(num_timesteps)
                    report_idx += 1

            # Save training data to checkpoint file
            #self.report_train(num_timesteps)
            if num_timesteps >= checkpoints[checkpt_idx]:
                self._dump_and_save(params)
                checkpt_idx += 1
        self._dump_and_save(params)
        return self.checkpoint_name

    def _dump_stats(self):
        """Returns dictionary of NN weights and normalization stats."""
        
        final = dict()

        # Actor and critic weights
        final['actor_weights'] = self.actor.get_weights()
        final['critic_weights'] = [critic.get_weights() 
            for critic in self.critics]
        
        # Normalization stats
        final['rms_stats'] = self.normalizer.get_rms_stats()

        return final

    def _dump_and_save(self,params):
        """Saves training data to checkpoint file and resets logger."""
        self.logger.log_params(params)

        final = self._dump_stats()
        self.logger.log_final(final)

        self.logger.dump_and_save(self.save_path,self.checkpoint_name)
        self.logger.reset()
        
    def report_train(self, num_timesteps):
        
        try:
            print('timesteps: %d    J_tot: %.5f    MSE_on_expert: %.5f    MSE_on_ca: %.5f'%(num_timesteps,self.logger.dump()['train']['J_tot'][-1],self.logger.dump()['train']['model_MSE_on_expert_data'][-1],self.logger.dump()['train']['model_MSE_on_expert_counterfactual_action'][-1]),flush=True)

        except:
            #try:
                
            print('timesteps: %d    J_tot: %.5f'%(num_timesteps,self.logger.dump()['train']['J_tot'][-1]),flush=True)
            
            #except:
            #    print('a: %d'%(a))
        
        
        