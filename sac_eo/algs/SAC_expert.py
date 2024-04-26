import numpy as np
import tensorflow as tf
import time

#from sac_eo.common.samplers import batch_simtrajectory_sampler,trajectory_sampler,mbpo_sampler
from sac_eo.common.normalizer import RunningNormalizers
from sac_eo.common.samplers import batch_simtrajectory_sampler,trajectory_sampler
from sac_eo.common.buffers import TrajectoryBuffer

from sac_eo.algs.mbrl_onpolicy_alg import MBRLOnPolicyAlg

def td_target(reward, discount, next_value):
    return reward + discount * next_value




class SAC_exp(MBRLOnPolicyAlg):
    """Base class for MBRL algorithms with off-policy updates."""

    def __init__(self,idx,env,env_eval,env_expert,actor,expert,init_expert_rms_stats,
                 v_critic,q_targets,q_critics,models,alg_kwargs,mf_update_kwargs):
        """Initializes MBRLOnPolicyAlg class. See BaseOnPolicyAlg for detais."""
        super(SAC_exp,self).__init__(idx,env,env_eval,actor,v_critic,models,
            alg_kwargs,mf_update_kwargs)
        
        
        self.q_critics=q_critics
        self.q_targets=q_targets
        
        
        
        
        self.qcritic_trainable = []
        for critic in self.q_critics:
            self.qcritic_trainable = self.qcritic_trainable + critic.trainable
            
        
        self.qtarget_trainable = []
        for critic in self.q_targets:
            self.qtarget_trainable = self.qtarget_trainable + critic.trainable
            
            
        
        #self.target_entropy=-np.prod(env.action_space.shape)  
        self.target_entropy=-len(env.action_space.sample())
        self.s=None #use this variable to modify the current state while collecting real data.
        
        if self.update_normalizers == True:
            self.new_traj= TrajectoryBuffer(self.s_dim,self.a_dim,self.gamma,
                        self.lam,self.env_buffer_size)
        
        self.model_normalizer = RunningNormalizers(self.s_dim,self.a_dim,self.gamma,
            self.init_rms_stats)
        
        
        self.model_buffer_size=int(alg_kwargs['model_buffer_size'])
        self.model_data = TrajectoryBuffer(self.s_dim,self.a_dim,self.gamma,
            self.lam,self.model_buffer_size)
        
        self.B=len(self.models)
        
        
        ###Expert Algorithm
        self.env_expert = env_expert
        self.expert=expert
        self.expert_data = TrajectoryBuffer(self.s_dim,self.a_dim,self.gamma,
            self.lam,self.expert_buffer_size)
        self.init_expert_rms_stats=init_expert_rms_stats
        self.expert_normalizer = RunningNormalizers(self.s_dim,self.a_dim,self.gamma,
            self.init_expert_rms_stats)
        
        
        self.model_MSE_on_expert_data=[] # we are going to use adaptive epsilon by using this list.
                                 # set epsilon as 1/(x+1)

        self.model_MSE_on_expert_counterfactual_action=[]#calculate model mse without expert actions.
                                                 #We use this value during policy update.

    def _setup(self,alg_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(SAC_exp,self)._setup(alg_kwargs)
        
        
        
        #MBPO related parameters
        self.init_temperature=alg_kwargs['init_temperature']
        self.mbpo_lr=alg_kwargs['q_crit_lr']
        self.mbpo_actor_lr=alg_kwargs['mbpo_actor_lr']
        self.mbpo_alpha_lr=alg_kwargs['mbpo_alpha_lr']
        self.E=alg_kwargs['mbpo_E']
        self.G=alg_kwargs['mbpo_G']
        self.sac_batch_size=alg_kwargs['sac_batch_size']
        self.soft_tau=alg_kwargs['soft_tau']
        self.target_update_int=alg_kwargs['target_update_int']
        self.repeat_after_real_steps=alg_kwargs['real_step_mod']
        self.random_act=alg_kwargs['random_act']
        self.update_normalizers=alg_kwargs['update_normalizers']
        self.only_model_normalizer=alg_kwargs['only_model_normalizer']
        

        
        
        
        
        self.alpha=tf.Variable(np.log(self.init_temperature),dtype=tf.float32)
        
        self.q_critic1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.mbpo_lr)
        self.q_critic2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.mbpo_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.mbpo_actor_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.mbpo_alpha_lr)
        self._max_episode_steps=1000
        
        
        self.expert_buffer_size = alg_kwargs['expert_buffer_size']
        if self.expert_buffer_size:
            self.expert_buffer_size = int(self.expert_buffer_size)
        self.epsilon=alg_kwargs['epsilon']
        self.scale_epsilon_by_true_MSE=alg_kwargs['scale_epsilon_by_true_MSE']
        self.use_expert_actions=alg_kwargs['use_expert_actions']
        self.scale_max_disc=alg_kwargs['scale_max_disc']
        self.scale_median_disc=alg_kwargs['scale_median_disc']
        self.scale_total_disc=alg_kwargs['scale_total_disc']
        self.exp_mult=alg_kwargs['exp_mult']
        self.min_mult=alg_kwargs['min_mult']
        self.mult_coeff=alg_kwargs['mult_coeff']
        self.exp_batch_type = alg_kwargs['exp_batch_type']
        self.expert_batch_size= alg_kwargs['expert_batch_size']
        
    
    def _set_rms(self):

        """Shares normalizers with all relevant classes."""
        super(MBRLOnPolicyAlg,self)._set_rms()
        for model in self.models:
            
            if self.only_model_normalizer==True:
                model.set_rms(self.model_normalizer)
            else:
                model.set_rms(self.normalizer)
        
        for qcrit in self.q_critics:
            qcrit.set_rms(self.normalizer)
            
        for qtarg in self.q_targets:
            qtarg.set_rms(self.normalizer)
        
        
        self.expert.set_rms(self.expert_normalizer)


    def _collect_expert_data(self):
        
        time_start = time.time()
        traj_start=0
        steps_start=0
        batch_size_cur = 0
        J_tot_all = []
        
        while batch_size_cur < self.expert_buffer_size:
            if self.exp_batch_type == 'steps':
                horizon = np.minimum(self.expert_buffer_size-batch_size_cur,self.env_horizon)
            else:
                horizon = self.env_horizon
                
            traj_and_J = trajectory_sampler(
                self.env_expert,self.expert,horizon,eval=True,deterministic=True,corruptor=self.corruptor)
            s_traj, a_traj, r_traj, sp_traj, d_traj, J_tot = traj_and_J
            #self.normalizer.update_rms(s_traj,a_traj,r_traj,sp_traj)
            self.expert_data.add(s_traj,a_traj,r_traj,sp_traj,d_traj)
            
            
            if horizon == self.env_horizon:
                J_tot_all.append(J_tot)
                
            if self.exp_batch_type == 'steps':
                steps_cur = self.expert_data.steps_total
                batch_size_cur = steps_cur - steps_start
            else:
                traj_cur = self.expert_data.traj_total
                batch_size_cur = traj_cur - traj_start
                
            
        steps_end = self.expert_data.steps_total
        traj_end = self.expert_data.traj_total
        
        J_tot_ave = np.mean(J_tot_all)
        steps_new = steps_end - steps_start
        traj_new = traj_end - traj_start
        
        
        time_end = time.time()
        time_expert_data = time_end - time_start
        
        log_expert_data = {
            'expert_J_tot':            J_tot_ave,
            'expert_steps':            steps_new,
            'expert_traj':             traj_new,
            'expert_time':    time_expert_data
        }
        self.expert_reward=J_tot_ave
        
        self.logger.log_train(log_expert_data)
            


    def _get_Q_target(self,sp,r,done):
        next_Actions,neglog_p = self.actor.evaluate(sp)
        
        
        next_Qs_values=tuple(
            Q.value(sp, next_Actions)
            for Q in self.q_targets)
        
        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        
        #Please see Equation 3, we already returned neglog_p instead of logp.
        next_value = min_next_Q + self.alpha * neglog_p
        
        Q_target=td_target(
            reward=1.0 * r[:,0],
            discount=self.gamma,
            next_value=(1-done)*next_value)
        
        return Q_target
    
    
    def _update_critic(self,s,a,sp,r,done):
    
        #start critic update # based on Eq 5.
        Q_target=self._get_Q_target(sp,r,done)
        Q_target=tf.expand_dims(Q_target,axis=-1)
        
        with tf.GradientTape() as tape:
            predicted_q_value1=self.q_critics[0]._forward(s,a)
            q1_loss= 0.5 * tf.reduce_sum(tf.square(predicted_q_value1 - Q_target),axis=-1)
            q1_loss= tf.reduce_mean(q1_loss)
            grads = tape.gradient(q1_loss,self.q_critics[0].trainable)
        self.q_critic1_optimizer.apply_gradients(zip(grads, self.q_critics[0].trainable))
        
        with tf.GradientTape() as tape:
            predicted_q_value2=self.q_critics[1]._forward(s,a)
            q2_loss= 0.5 * tf.reduce_sum(tf.square(predicted_q_value2 - Q_target),axis=-1)
            q2_loss= tf.reduce_mean(q2_loss)
            grads = tape.gradient(q2_loss,self.q_critics[1].trainable)
        self.q_critic2_optimizer.apply_gradients(zip(grads, self.q_critics[1].trainable))
        #end critic update
        
        
        log_env_data = {
            'q1_loss':            q1_loss.numpy(),
            'q2_loss':            q2_loss.numpy()
        }

        #self.logger.log_train(log_env_data)
    
    
    def _update_actor_and_alpha(self,s,expert_reg):
        
        s_expert=expert_reg[0]
        a_expert=expert_reg[1]
        sp_expert=expert_reg[2]
        epsilon=expert_reg[3]
        use_expert_actions=expert_reg[4]
        
        
        if len(self.models) == 1:
            
            with tf.GradientTape() as tape:
                new_action,neglog_p=self.actor.evaluate(s)
                neglog_p=tf.expand_dims(neglog_p,axis=-1)
                Q_log_targets = tuple(
                    Q._forward(s, new_action)
                    for Q in self.q_critics)
                min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)
                """
                Notice:
                        Prior log prob is ignored.
                        
                        Please check _init_actor_update in 
                        https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/softlearning/algorithms/sac.py#L11
                """
                p_loss= tf.reduce_mean(-self.alpha * neglog_p-min_Q_log_target)
                #loss= -self.alpha * neglog_p-min_Q_log_target-policy_prior_log_probs
                
                counterfactual_a=self.actor.sample(s_expert,deterministic=False)
                sp_pred = self.models[0].sample(s_expert,counterfactual_a,deterministic=True)
                
                delta_loss = 0.5 * tf.reduce_sum(
                            tf.square(sp_expert - sp_pred),axis=-1)
                MSE_loss=tf.reduce_mean(delta_loss)
                p_loss = (1-epsilon)*p_loss + epsilon * MSE_loss
            grads_policy = tape.gradient(p_loss,self.actor.trainable)
        
        else:
                
            idx = np.arange(len(s_expert))
            self.rng.shuffle(idx)
            sections=np.array_split(idx, len(self.models))
            
            s_expert_one=s_expert[sections[0]]
            s_expert_two=s_expert[sections[1]]
                                  
            sp_expert_one=sp_expert[sections[0]]
            sp_expert_two=sp_expert[sections[1]]
            
            with tf.GradientTape() as tape:
                new_action,neglog_p=self.actor.evaluate(s)
                neglog_p=tf.expand_dims(neglog_p,axis=-1)
                Q_log_targets = tuple(
                    Q._forward(s, new_action)
                    for Q in self.q_critics)
                min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)
                
                p_loss= tf.reduce_mean(-self.alpha * neglog_p-min_Q_log_target)
                
                c_a_one=self.actor.sample(s_expert_one,deterministic=False)
                c_a_two=self.actor.sample(s_expert_two,deterministic=False)
                
                
                sp_pred_one = self.models[0].sample(s_expert_one,c_a_one,deterministic=True)
                sp_pred_two = self.models[1].sample(s_expert_two,c_a_two,deterministic=True)
                
                
                delta_loss = tf.reduce_sum(
                            tf.square(sp_expert_one - sp_pred_one),axis=-1) +\
                            tf.reduce_sum(
                                        tf.square(sp_expert_two - sp_pred_two),axis=-1)
                delta_loss=0.5*delta_loss
                MSE_loss=tf.reduce_mean(delta_loss)
                p_loss = (1-epsilon)*p_loss + epsilon * MSE_loss
            grads_policy = tape.gradient(p_loss,self.actor.trainable)
            
        self.actor_optimizer.apply_gradients(zip(grads_policy, self.actor.trainable))
        
        #alpha update
        with tf.GradientTape() as tape: #based on Eq 17.
            new_action,neglog_p=self.actor.evaluate(s)
            neglog_p=tf.expand_dims(neglog_p,axis=-1)
            
            alpha_loss = -self.alpha*tf.reduce_mean(-neglog_p + self.target_entropy)
        grads_alpha = tape.gradient(alpha_loss,self.alpha)
        self.alpha_optimizer.apply_gradients(zip([grads_alpha], [self.alpha]))
        self.alpha.assign(np.maximum(self.alpha.numpy(),1e-5))
        
        
        log_env_data = {
            'alpha_loss':            alpha_loss.numpy(),
            'p_loss':            p_loss.numpy(),
            'epsilon':          epsilon
        }
        self.logger.log_train(log_env_data)
 
    
    
    
    
    def _update_q_target(self):
        
        
        tmp=[]
        for l in range (len(self.q_targets[0].get_weights())):
            tmp.append(self.q_targets[0].get_weights()[l] * (1.0 - self.soft_tau) + self.q_critics[0].get_weights()[l] * self.soft_tau)
        self.q_targets[0].set_weights(tmp)
        
        tmp=[]
        for l in range (len(self.q_targets[1].get_weights())):
            tmp.append(self.q_targets[1].get_weights()[l] * (1.0 - self.soft_tau) + self.q_critics[1].get_weights()[l] * self.soft_tau)
        self.q_targets[1].set_weights(tmp)

    def _expert_preprocess(self):
        """Updates actor by using expert data
        
        Args:
            rollout_data (tuple): tuple of data collected to use for updates
        """
        epsilon_coef=self.epsilon
        s_expert, a_expert, sp_expert, r_expert = self.expert_data.get_model_info()
        if self.scale_epsilon_by_true_MSE == True:
            #please notice that inside this if statement smaller self.epsilon values lead to
            #greater epsilon coef. Thus, more weight is given to MSE term 
            #even if the true MSE is large.
            epsilon_coef=1/(self.epsilon*(self.model_MSE_on_expert_counterfactual_action[-1]) + 1)
            print('epsilon coef:    %.5f '%(epsilon_coef),flush=True)
            cur_reward=self.current_reward
            #exp_reward=self.logger.train_dict['expert_J_tot'][-1]
            if cur_reward>0:

                if self.min_mult ==True:

                    epsilon_coef=epsilon_coef* (-min(self.mult_coeff*(cur_reward/self.expert_reward)-1, 0))
                    #mult coeff must be between 0 and 1. 
                    #1, 0.75, 0.5, 0.25 are some candidates.
                if self.exp_mult ==True:

                    epsilon_coef=epsilon_coef* np.exp(-self.mult_coeff*cur_reward/self.expert_reward)
                    #mult coeff must be positive integers.
                    #1, 2, 4 are some candidates.
        
        elif self.scale_max_disc==True:
            disc_ratio,max_disc,median_disc,s_disc_total=self._calc_disc(s_expert,a_expert,sp_expert)
            epsilon_coef=1/(self.epsilon*(max_disc) + 1)
            print('epsilon coef:    %.5f '%(epsilon_coef),flush=True)
        elif self.scale_median_disc==True:
            disc_ratio,max_disc,median_disc,s_disc_total=self._calc_disc(s_expert,a_expert,sp_expert)
            epsilon_coef=1/(self.epsilon*(median_disc) + 1)
            print('epsilon coef:    %.5f '%(epsilon_coef),flush=True)
        elif self.scale_total_disc==True:
            disc_ratio,max_disc,median_disc,s_disc_total=self._calc_disc(s_expert,a_expert,sp_expert)
            epsilon_coef=1/(self.epsilon*(s_disc_total) + 1)
            print('epsilon coef:    %.5f '%(epsilon_coef),flush=True)
        
        else:
            epsilon_coef=self.epsilon
        
        
        if self.expert_batch_size:
            s_expert, a_expert, sp_expert, r_expert = self.expert_data.get_model_info(batch_size=self.expert_batch_size)
        expert_reg=(s_expert,a_expert,sp_expert,epsilon_coef,self.use_expert_actions)
        return expert_reg


    def _calc_disc(self,s_expert,a_expert,sp_expert):
        
        #s_expert, a_expert, sp_expert, r_expert = self.expert_data.get_model_info()
        
        sp_pred=[]
        if self.use_expert_actions==True:
            
            
            for model in self.models:
                
                    sp_pred.append(model.sample(s_expert,a_expert,deterministic=False))
            
        
        else:
            
            counterfactual_a=self.actor.sample(s_expert,deterministic=False)
            counterfactual_a=self.actor.tf_clip(counterfactual_a)
            
            for model in self.models:
                sp_pred.append(model.sample(s_expert,counterfactual_a,deterministic=False))
        
        
        diff=sp_pred[0] - sp_pred[1]
        s_disc=tf.math.reduce_euclidean_norm(diff,axis=1).numpy()
        s_disc_total=np.sum(s_disc)
        disc_ratio=s_disc/s_disc_total
        max_disc=np.max(s_disc)
        median_disc=np.median(s_disc)
        
            
        
        
        
        return disc_ratio,max_disc,median_disc,s_disc_total
    
    
    def _update(self,num_timesteps,expert_reg):
        
        s,a,sp,r,done =self.env_data.get_offmodel_info(batch_size=self.sac_batch_size)
        r=np.expand_dims(r,axis=1)
        #done=np.expand_dims(done,axis=1)

        ###Updates
        self._update_critic(s,a,sp,r,done)
        self._update_actor_and_alpha(s,expert_reg)
        
        
        
        if num_timesteps % self.target_update_int== 0:
            """q_target is updated after each self.target_update_int iterations"""
            self._update_q_target()


    def _update_models(self):
        """Updates learned dynamics models."""
        time_start = time.time()
        
        s_all, a_all, sp_all, r_all = self.model_data.get_model_info()

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

            cntr=0
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

                        
        if self.reset_model_optimizer == True:
            self.model_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_lr)
        
        # log_model_fit_all = []
        # for model in self.models:
        #     loss_train = model.get_loss(s_train,sp_train,a_train,r_train)
        #     if self.model_holdout_ratio > 0.0:
        #         loss_holdout = model.get_loss(
        #             s_holdout,sp_holdout,a_holdout,r_holdout)
        #     else:
        #         loss_holdout = tf.constant(np.inf)
        #     loss_all = model.get_loss(s_all,sp_all,a_all,r_all)
        #     loss_mse, loss_reward = model.get_losses_eval(
        #         s_all,sp_all,a_all,r_all)

        #     log_model_fit = {
        #         'model_loss':           loss_all.numpy(),
        #         'model_loss_train':     loss_train.numpy(),
        #         'model_loss_holdout':   loss_holdout.numpy(),
        #         'model_loss_mse':       loss_mse.numpy(),
        #         'model_loss_reward':    loss_reward.numpy(),
        #     }
        #     log_model_fit_all.append(log_model_fit)
        
        
        #Evaluate the model performance on expert data
        model_MSE_on_expert_data=[]
        for model in self.models:
            s_expert, a_expert, sp_expert, r_expert = self.expert_data.get_model_info()
            sp_pred = model.sample(s_expert,a_expert,deterministic=True)
            
            delta_loss = 0.5 * tf.reduce_sum(
                    tf.square(sp_pred - sp_expert),axis=-1)
            model_MSE_on_expert_data.append(tf.reduce_mean(delta_loss).numpy())
        
        model_MSE_on_expert_data=np.mean(np.array(model_MSE_on_expert_data))
        self.model_MSE_on_expert_data.append(model_MSE_on_expert_data)
        
        
        model_MSE_on_expert_counterfactual_action=[]
        if self.use_expert_actions==True:
            self.model_MSE_on_expert_counterfactual_action.append(model_MSE_on_expert_data)
        else:
            
            counterfactual_a=self.actor.sample(s_expert,deterministic=False)
            for model in self.models:
                s_expert, a_expert, sp_expert, r_expert = self.expert_data.get_model_info()
                sp_pred = model.sample(s_expert,counterfactual_a,deterministic=True)
                
                delta_loss = 0.5 * tf.reduce_sum(
                        tf.square(sp_pred - sp_expert),axis=-1)
                model_MSE_on_expert_counterfactual_action.append(tf.reduce_mean(delta_loss).numpy())
            
            model_MSE_on_expert_counterfactual_action=np.mean(np.array(model_MSE_on_expert_counterfactual_action))
            self.model_MSE_on_expert_counterfactual_action.append(model_MSE_on_expert_counterfactual_action)
            
        time_end = time.time()
        time_model_fit = time_end - time_start

        log_model_fit_agg = {
            'time_model_fit':       time_model_fit,
            'model_ent':            ent_all,
            'model_loss_epochs':    ep + 1,
            'model_MSE_on_expert_data': model_MSE_on_expert_data,
            'model_MSE_on_expert_counterfactual_action': model_MSE_on_expert_counterfactual_action
        }
            
        self.logger.log_train(log_model_fit_agg)
        #self.logger.log_train_ensemble(log_model_fit_all)
    

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
            self.model_data.add(s_traj,a_traj,r_traj,sp_traj,d_traj)

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
        if self.eval_freq is None:
            evaluate = False 
        else:
            evaluate = True
            eval_points = np.concatenate(
                (np.arange(0,total_timesteps,self.eval_freq)[1:],
                [total_timesteps]))
        
        # Training loop
        num_timesteps = 0
        if evaluate:
            self._evaluate(num_timesteps)
        
        
        
        
        #Collect certain amount of real_data before training.
        
        steps_new=self._collect_env_data(num_timesteps,update_normalizers=self.update_normalizers,only_model_normalizer=self.only_model_normalizer)
        num_timesteps=num_timesteps+steps_new
        
        
        episode_step, episode, episode_reward, done = 0, 0, 0, True
        time_start = time.time()
        while num_timesteps< total_timesteps:
            
            
            if done == True:
                #s,a,r,sp
                

                
                if self.update_normalizers==True and episode>0:
                    if self.only_model_normalizer == True:
                        self.model_normalizer.update_rms(self.new_traj.get_model_info()[0],self.new_traj.get_model_info()[1],self.new_traj.get_model_info()[3],self.new_traj.get_model_info()[2])
                    else:
                        self.normalizer.update_rms(self.new_traj.get_model_info()[0],self.new_traj.get_model_info()[1],self.new_traj.get_model_info()[3],self.new_traj.get_model_info()[2])
                        self.model_normalizer.update_rms(self.new_traj.get_model_info()[0],self.new_traj.get_model_info()[1],self.new_traj.get_model_info()[3],self.new_traj.get_model_info()[2])
                    self.new_traj.reset()
                
                time_end = time.time()
                time_env_data=time_end-time_start
                if episode > 0:        
                    log_env_data = {
                        'J_tot':            episode_reward,
                        'steps':            episode_step,
                        'traj':             1,
                        'time_env_data':    time_env_data
                    }
                    
                    if episode % 5 == 0:
                        print('timesteps: %d    J_tot: %.5f    MSE_on_expert: %.5f    MSE_on_ca: %.5f'%(num_timesteps,self.logger.dump()['train']['J_tot'][-1],self.logger.dump()['train']['model_MSE_on_expert_data'][-1],self.logger.dump()['train']['model_MSE_on_expert_counterfactual_action'][-1]),flush=True)
                    self.logger.log_train(log_env_data)

                
                obs = self.env.reset()
                #self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                #self._global_episode += 1
                
                
                #update environment models and get expert data
                self._update_models()
                expert_reg=self._expert_preprocess()
                
                time_start = time.time()
            
            
            a = self.actor.sample(obs,deterministic=not self.random_act).numpy()
            self._update(num_timesteps,expert_reg)
            
            #if episode_step % int(self.repeat_after_real_steps)==0:
                
            #    for g_dx in range (self.G):
            #        self._update(num_timesteps)
            
            next_obs, r, done, _ = self.env.step(self.actor.clip(a))
            done_no_max=False if episode_step + 1 == self._max_episode_steps else done
            
            
            episode_reward += r
            
            self.env_data.add(np.array([obs]), np.array([a]), np.array([r]), \
                              np.array([next_obs]), np.array([done_no_max]))
            
            self.model_data.add(np.array([obs]), np.array([a]), np.array([r]), \
                              np.array([next_obs]), np.array([done_no_max]))
            
            if self.update_normalizers==True:
                self.new_traj.add(np.array([obs]), np.array([a]), np.array([r]), \
                                  np.array([next_obs]), np.array([done_no_max]))
            
            obs = next_obs
            episode_step += 1
            num_timesteps += 1
            

                
            # Evaluate
            if evaluate:
                if num_timesteps >= eval_points[eval_idx]:
                    self._evaluate(num_timesteps)
                    eval_idx += 1

            # Save training data to checkpoint file
            if num_timesteps >= checkpoints[checkpt_idx]:
                self._dump_and_save(params)
                checkpt_idx += 1
    
    
            #self.s=None#time to reset current state. Real data collection is completed in this epoch.
    
        self._dump_and_save(params)
        return self.checkpoint_name
    



