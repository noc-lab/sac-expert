import numpy as np
import tensorflow as tf

from sac_eo.algs.model_free.base_mfrl_updates import BaseOnPolicyUpdate

class PPO(BaseOnPolicyUpdate):
    """Algorithm class for PPO actor and critic updates."""

    def __init__(self,actor,update_kwargs):
        """Initializes PPO class."""
        super(PPO,self).__init__(actor,update_kwargs)
    
    def _setup(self,update_kwargs):
        """Sets up hyperparameters as class attributes."""

        self.actor_lr = update_kwargs['actor_lr']
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.actor_lr)

        self.actor_update_it = update_kwargs['actor_update_it']
        self.actor_nminibatch = update_kwargs['actor_nminibatch']

        self.adv_center = update_kwargs['adv_center']
        self.adv_scale = update_kwargs['adv_scale']

        self.eps = update_kwargs['eps_ppo']
        self.max_grad_norm = update_kwargs['max_grad_norm']
        
        self.adaptlr = update_kwargs['adaptlr']
        self.adapt_factor = update_kwargs['adapt_factor']
        self.adapt_minthresh = update_kwargs['adapt_minthresh']
        self.adapt_maxthresh = update_kwargs['adapt_maxthresh']

        self.ent_reg = update_kwargs['ent_reg']
        self.ent_targ = update_kwargs['ent_targ']
        self.alpha = tf.Variable(0.0,dtype=tf.float32)
        self.alpha_lr = update_kwargs['alpha_lr']
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.alpha_lr)

    def update(self,rollout_data,expert_reg=None):
        """Updates actor."""
        s_all, a_all, adv_all, _, _, _ = rollout_data
        neglogp_old_all = self.actor.neglogp(s_all,a_all).numpy()
        kl_info_ref = self.actor.get_kl_info(s_all)
        ent = tf.reduce_mean(self.actor.entropy(s_all))

        n_samples = s_all.shape[0]
        n_batch = int(n_samples / self.actor_nminibatch)

        grad_norm_pre_all = 0.0
        grad_norm_post_all = 0.0

        # Minibatch update loop for actor
        for _ in range(self.actor_update_it):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,n_batch)[1:]

            batches = np.array_split(idx,sections)
            if (n_samples % n_batch != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                # Active data
                s_active = s_all[batch_idx]
                a_active = a_all[batch_idx]
                adv_active = adv_all[batch_idx]
                neglogp_old_active = neglogp_old_all[batch_idx]

                adv_mean = np.mean(adv_active)
                adv_std = np.std(adv_active) + 1e-8

                if self.adv_center:
                    adv_active = adv_active - adv_mean 
                if self.adv_scale:
                    adv_active = adv_active / adv_std

                # Actor update
                grad_norm_pre, grad_norm_post = self._apply_actor_grad(
                    s_active,a_active,adv_active,neglogp_old_active,expert_reg)
                
                grad_norm_pre_all += grad_norm_pre
                grad_norm_post_all += grad_norm_post

        grad_norm_pre_ave = grad_norm_pre_all.numpy() / (
            self.actor_update_it * len(batches))
        grad_norm_post_ave = grad_norm_post_all.numpy() / (
            self.actor_update_it * len(batches))
                
        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        ratio_diff = tf.abs(ratio - 1.)
        tv = 0.5 * tf.reduce_mean(ratio_diff)
        kl = tf.reduce_mean(self.actor.kl(s_all,kl_info_ref))

        log_actor = {
            'ent':                  ent.numpy(),
            'tv':                   tv.numpy(),
            'kl':                   kl.numpy(),
            'alpha':                self.alpha.numpy(),
            'actor_lr':             self.actor_optimizer.learning_rate.numpy(),
            'outside_clip':         np.mean(ratio_diff > self.eps),
            'actor_grad_norm_pre':  grad_norm_pre_ave,
            'actor_grad_norm':      grad_norm_post_ave
        }

        # Adapt learning rate
        if self.adaptlr:
            if tv > (self.adapt_maxthresh * 0.5 * self.eps):
                lr_new = (self.actor_optimizer.learning_rate.numpy() / 
                    (1+self.adapt_factor))
                self.actor_optimizer.learning_rate.assign(lr_new)
            elif tv < (self.adapt_minthresh * 0.5 * self.eps):
                lr_new = (self.actor_optimizer.learning_rate.numpy() * 
                    (1+self.adapt_factor))
                self.actor_optimizer.learning_rate.assign(lr_new)
        
        return log_actor

    # @tf.function
    def _apply_actor_grad(self,s_active,a_active,adv_active,neglogp_old_active,expert_reg):
        """Performs single actor update.

        Args:
            s_active (np.ndarray): states
            a_active (np.ndarray): actions
            adv_active (np.ndarray): advantages
            neglogp_old_active (np.ndarray): negative log probabilities
        """

        if expert_reg is None:
            with tf.GradientTape() as tape:
                neglogp_cur_active = self.actor.neglogp(s_active,a_active)
                ratio = tf.exp(neglogp_old_active - neglogp_cur_active)
                ratio_clip = tf.clip_by_value(ratio,1.-self.eps,1.+self.eps)
                
                pg_loss_surr = ratio * adv_active * -1
                pg_loss_clip = ratio_clip * adv_active * -1
                pg_loss = tf.reduce_mean(tf.maximum(pg_loss_surr,pg_loss_clip))
    
                ent_loss = tf.reduce_mean(self.actor.entropy(s_active))
                pg_loss = pg_loss - self.alpha * (ent_loss - self.ent_targ)
    
            neg_pg, alpha_grad = tape.gradient(pg_loss,
                [self.actor.trainable,self.alpha])
        
        else:
            s_expert=expert_reg[0]
            a_expert=expert_reg[1]
            sp_expert=expert_reg[2]
            epsilon=expert_reg[3]
            models=expert_reg[4]
            use_expert_actions=expert_reg[5]
            
            with tf.GradientTape() as tape:
                neglogp_cur_active = self.actor.neglogp(s_active,a_active)
                ratio = tf.exp(neglogp_old_active - neglogp_cur_active)
                ratio_clip = tf.clip_by_value(ratio,1.-self.eps,1.+self.eps)
                
                pg_loss_surr = ratio * adv_active * -1
                pg_loss_clip = ratio_clip * adv_active * -1
                pg_loss = tf.reduce_mean(tf.maximum(pg_loss_surr,pg_loss_clip))
    
                ent_loss = tf.reduce_mean(self.actor.entropy(s_active))
                pg_loss = pg_loss - self.alpha * (ent_loss - self.ent_targ)
                
                """
                #REGULARIZE pg_loss by using expert data.
                counterfactual_a=self.actor._forward(s_expert)#consider using sample here.
                
                delta_pred, r_pred = models[0]._forward(s_expert,counterfactual_a,clip=False)
                #Again use sample to obtain next state.
                delta = sp_expert - s_expert
                delta_norm = models[0].delta_rms.normalize(delta)
                
                if models[0].delta_clip_loss:
                    delta_norm = tf.clip_by_value(delta_norm,
                        models[0].delta_clip_loss*-1,models[0].delta_clip_loss)
                    
                delta_loss = 0.5 * tf.reduce_sum(
                                tf.square(delta_norm - delta_pred),axis=-1)
                
                MSE_loss=tf.reduce_mean(delta_loss)
                """
                #REGULARIZE pg_loss by using expert data.
                counterfactual_a=self.actor.sample(s_expert,deterministic=False)
                counterfactual_a=self.actor.tf_clip(counterfactual_a)
                #Do we want to sample deterministic actions here ?
                
                if use_expert_actions == True:
                    
                    delta_loss = 0.5 * tf.reduce_sum(
                                    tf.square(a_expert - counterfactual_a),axis=-1)
                
                else:
                    
                    sp_pred = models[0].sample(s_expert,counterfactual_a,deterministic=True)
                    
                    delta_loss = 0.5 * tf.reduce_sum(
                                tf.square(sp_expert - sp_pred),axis=-1)
                
                #sp_pred = models[0].sample(s_expert,counterfactual_a)
                #Again use sample to obtain next state.
                #delta = sp_expert - s_expert
                #delta_norm = models[0].delta_rms.normalize(delta)
                
                #if models[0].delta_clip_loss:
                #    delta_norm = tf.clip_by_value(delta_norm,
                #        models[0].delta_clip_loss*-1,models[0].delta_clip_loss)
                    
                delta_loss = 0.5 * tf.reduce_sum(
                                tf.square(sp_expert - sp_pred),axis=-1)
                
                MSE_loss=tf.reduce_mean(delta_loss)
                
                
                pg_loss = (1-epsilon)*pg_loss + epsilon * MSE_loss
                #raise NotImplementedError
                #need to use expert data here!
    
            neg_pg, alpha_grad = tape.gradient(pg_loss,
                [self.actor.trainable,self.alpha])

        if self.ent_reg:
            self.alpha_optimizer.apply_gradients(
                zip([alpha_grad*-1],[self.alpha]))
            self.alpha.assign(tf.maximum(self.alpha,0.0))

        if self.max_grad_norm is not None:
            neg_pg, grad_norm_pre = tf.clip_by_global_norm(
                neg_pg,self.max_grad_norm)
        else:
            grad_norm_pre = tf.linalg.global_norm(neg_pg)
        grad_norm_post = tf.linalg.global_norm(neg_pg)

        self.actor_optimizer.apply_gradients(zip(neg_pg,self.actor.trainable))

        return grad_norm_pre, grad_norm_post