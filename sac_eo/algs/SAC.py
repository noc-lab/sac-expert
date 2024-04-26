import numpy as np
import tensorflow as tf
import time

from sac_eo.common.normalizer import RunningNormalizers
from sac_eo.common.samplers import batch_simtrajectory_sampler,trajectory_sampler
from sac_eo.common.buffers import TrajectoryBuffer

from sac_eo.algs.mbrl_onpolicy_alg import MBRLOnPolicyAlg

def td_target(reward, discount, next_value):
    return reward + discount * next_value




class SAC(MBRLOnPolicyAlg):
    """Base class for MBRL algorithms with off-policy updates."""

    def __init__(self,idx,env,env_eval,actor,v_critic,q_targets,q_critics,models,alg_kwargs,mf_update_kwargs):
        """Initializes MBRLOnPolicyAlg class. See BaseOnPolicyAlg for detais."""
        super(SAC,self).__init__(idx,env,env_eval,actor,v_critic,models,
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

    def _setup(self,alg_kwargs):
        """Sets up hyperparameters as class attributes."""
        super(SAC,self)._setup(alg_kwargs)
        
        
        
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
    
    def _init_sim_data_all(self):
        """"Initilizates replay buffer."""
        for idx in range(self.B):
            buffer = self.sim_data[idx]
            buffer.reset() 

    def _collect_sim_data_all(self,models):
        """"Collects simulated data from all learned dynamics models."""
        for idx in range(self.B):
            model = models[idx]
            buffer = self.sim_data[idx]
            #buffer.reset() # no need to reset buffer.
            self._collect_sim_data(model,buffer)
    
            


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
    
    
    def _update_actor_and_alpha(self,s):
        #actor update
        #start actor update # based on Equation 9.
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
            'p_loss':            p_loss.numpy()
        }

        #self.logger.log_train(log_env_data)
    
    
    
    
    
    def _update_q_target(self):
        
        tmp=[]
        for l in range (len(self.q_targets[0].get_weights())):
            tmp.append(self.q_targets[0].get_weights()[l] * (1.0 - self.soft_tau) + self.q_critics[0].get_weights()[l] * self.soft_tau)
        self.q_targets[0].set_weights(tmp)
        
        tmp=[]
        for l in range (len(self.q_targets[1].get_weights())):
            tmp.append(self.q_targets[1].get_weights()[l] * (1.0 - self.soft_tau) + self.q_critics[1].get_weights()[l] * self.soft_tau)
        self.q_targets[1].set_weights(tmp)
    
    
    def _update(self,num_timesteps):
        
        s,a,sp,r,done =self.env_data.get_offmodel_info(batch_size=self.sac_batch_size)
        r=np.expand_dims(r,axis=1)
        #done=np.expand_dims(done,axis=1)

        ###Updates
        self._update_critic(s,a,sp,r,done)
        self._update_actor_and_alpha(s)
        
        
        
        if num_timesteps % self.target_update_int== 0:
            """q_target is updated after each self.target_update_int iterations"""
            self._update_q_target()
    

    
    def train(self,total_timesteps,params):
        """Training loop.
        Args:
            total_timesteps (int): number of true environment steps for training
        Returns:
            Name of checkpoint file.
        """
        
        self._set_rms()
        #self._collect_expert_data()
        
        
        
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
        #E=1000#denotes realstep per model update. In paper, it is given as 1000 for all environments!
        num_timesteps = 0
        if evaluate:
            self._evaluate(num_timesteps)
        
        
        
        
        #Collect certain amount of real_data before training.
        
        
        steps_new=self._collect_env_data(num_timesteps,update_normalizers=self.update_normalizers,only_model_normalizer=self.only_model_normalizer)
        #self._init_sim_data_all()
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
                        print('timesteps: %d    J_tot: %.5f' %(num_timesteps,episode_reward),flush=True) 
                    self.logger.log_train(log_env_data)

                
                obs = self.env.reset()
                #self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                #self._global_episode += 1
                time_start = time.time()
            
            
            a = self.actor.sample(obs,deterministic=not self.random_act).numpy()
            
            
            if episode_step % int(self.repeat_after_real_steps)==0:
                
                for g_dx in range (self.G):
                    self._update(num_timesteps)
            
            next_obs, r, done, _ = self.env.step(self.actor.clip(a))
            done_no_max=False if episode_step + 1 == self._max_episode_steps else done
            
            
            episode_reward += r
            
            self.env_data.add(np.array([obs]), np.array([a]), np.array([r]), \
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
    



