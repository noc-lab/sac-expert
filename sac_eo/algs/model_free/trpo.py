import numpy as np
import tensorflow as tf

from sac_eo.algs.model_free.base_mfrl_updates import BaseOnPolicyUpdate
from sac_eo.common.nn_utils import list_to_flat
from sac_eo.common.update_utils import cg

class TRPO(BaseOnPolicyUpdate):
    """Algorithm class for TRPO actor and critic updates."""

    def __init__(self,actor,update_kwargs):
        """Initializes TRPO class."""
        super(TRPO,self).__init__(actor,update_kwargs)
    
    def _setup(self,update_kwargs):
        """Sets up hyperparameters as class attributes."""

        self.adv_center = update_kwargs['adv_center']
        self.adv_scale = update_kwargs['adv_scale']
        
        self.delta = update_kwargs['delta_trpo']
        self.cg_it = update_kwargs['cg_it']
        self.trust_sub = update_kwargs['trust_sub']
        self.trust_damp = update_kwargs['trust_damp']
        self.kl_maxfactor = update_kwargs['kl_maxfactor']

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

        # Compute surrogate objective gradient
        adv_mean = np.mean(adv_all)
        adv_std = np.std(adv_all) + 1e-8
        

        if self.adv_center:
            adv_all = adv_all - adv_mean 
        if self.adv_scale:
            adv_all = adv_all / adv_std

        
        if expert_reg is None:

            with tf.GradientTape() as tape:
                neglogp_cur_all = self.actor.neglogp(s_all,a_all)
                ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
                
                pg_loss_surr = ratio * adv_all * -1
                pg_loss = tf.reduce_mean(pg_loss_surr)
    
                ent_loss = tf.reduce_mean(self.actor.entropy(s_all))
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
            rng=expert_reg[6]
            
            
            if len(models) == 1:
                
                with tf.GradientTape() as tape:
                    neglogp_cur_all = self.actor.neglogp(s_all,a_all)
                    ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
                    
                    pg_loss_surr = ratio * adv_all * -1
                    pg_loss = tf.reduce_mean(pg_loss_surr)
        
                    ent_loss = tf.reduce_mean(self.actor.entropy(s_all))
                    pg_loss = pg_loss - self.alpha * (ent_loss - self.ent_targ)
                neg_pg, alpha_grad = tape.gradient(pg_loss,
                    [self.actor.trainable,self.alpha])
                
                with tf.GradientTape() as tape:
                    
                    
                    #REGULARIZE pg_loss by using expert data.
                    
                    
                    counterfactual_a=self.actor.sample(s_expert,deterministic=False)
                    counterfactual_a=self.actor.tf_clip(counterfactual_a)
                    
                    sp_pred = models[0].sample(s_expert,counterfactual_a,deterministic=True)
                        
                    delta_loss = 0.5 * tf.reduce_sum(
                                     tf.square(sp_expert - sp_pred),axis=-1)
                    MSE_loss=tf.reduce_mean(delta_loss)
                
                MSE_loss_grads,MSE_alpha_grad = tape.gradient(MSE_loss,[self.actor.trainable,self.alpha])
                
                grad_final=[]
                for i in range(len(MSE_loss_grads)):
                    final=(1-epsilon)*neg_pg[i] + epsilon*MSE_loss_grads[i]
                    grad_final.append(final)
                    
                alpha_grad=(1-epsilon)*alpha_grad + epsilon*MSE_alpha_grad
            
            else:
                
                idx = np.arange(len(s_expert))
                rng.shuffle(idx)
                sections=np.array_split(idx, len(models))
                
                s_expert_one=s_expert[sections[0]]
                s_expert_two=s_expert[sections[1]]
                                      
                sp_expert_one=sp_expert[sections[0]]
                sp_expert_two=sp_expert[sections[1]]
                
                with tf.GradientTape() as tape:
                    neglogp_cur_all = self.actor.neglogp(s_all,a_all)
                    ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
                    
                    pg_loss_surr = ratio * adv_all * -1
                    pg_loss = tf.reduce_mean(pg_loss_surr)
        
                    ent_loss = tf.reduce_mean(self.actor.entropy(s_all))
                    pg_loss = pg_loss - self.alpha * (ent_loss - self.ent_targ)
                neg_pg, alpha_grad = tape.gradient(pg_loss,
                    [self.actor.trainable,self.alpha])
                
                with tf.GradientTape() as tape:

                    
                    c_a_one=self.actor.sample(s_expert_one,deterministic=False)
                    c_a_two=self.actor.sample(s_expert_two,deterministic=False)
                    
                    
                    sp_pred_one = models[0].sample(s_expert_one,c_a_one,deterministic=True)
                    sp_pred_two = models[1].sample(s_expert_two,c_a_two,deterministic=True)
                    
                    
                    delta_loss = tf.reduce_sum(
                                tf.square(sp_expert_one - sp_pred_one),axis=-1) +\
                                tf.reduce_sum(
                                            tf.square(sp_expert_two - sp_pred_two),axis=-1)
                    delta_loss=0.5*delta_loss
                    MSE_loss=tf.reduce_mean(delta_loss)
                MSE_loss_grads,MSE_alpha_grad = tape.gradient(MSE_loss,[self.actor.trainable,self.alpha])
                    
                grad_final=[]
                norm_pg=0
                norm_MSE=0
                for i in range(len(MSE_loss_grads)):
                    final=(1-epsilon)*neg_pg[i] + epsilon*MSE_loss_grads[i]
                    grad_final.append(final)
                    
                    #norm_calculation
                    norm_pg+=tf.norm(neg_pg[i])
                    norm_MSE+=tf.norm(MSE_loss_grads[i])
                    
                #alpha_grad=(1-epsilon)*alpha_grad + epsilon*MSE_alpha_grad
                

        if self.ent_reg:
            self.alpha_optimizer.apply_gradients(
                zip([alpha_grad*-1],[self.alpha]))
            self.alpha.assign(tf.maximum(self.alpha,0.0))
        
        #pg_vec = list_to_flat(final) * -1
        pg_vec = list_to_flat(grad_final) * -1

        # Compute policy update
        if np.allclose(pg_vec,0) or self.delta==0.0:
            eta_v_flat = np.zeros_like(pg_vec)
        else:
            F = self._make_F(s_all)
            v_flat = cg(F,pg_vec,cg_iters=self.cg_it)

            vFv = np.dot(v_flat,F(v_flat))
            eta = np.sqrt(2*self.delta/vFv)
            eta_v_flat = eta * v_flat

        # Correct policy update with backtracking line search
        log_actor = self._backtrack(eta_v_flat,s_all,a_all,adv_all,
            neglogp_old_all)
        
        log_actor['alpha'] = self.alpha.numpy()
        log_actor['epsilon'] = epsilon
        log_actor['norm_pg'] = norm_pg
        log_actor['norm_MSE'] = norm_MSE
        
        return log_actor

    def _make_F(self,s_all):
        """Creates matrix-vector product function for average FIM.

        Args:
            s_all (np.ndarray): states
        
        Returns:
            Matrix-vector product function for average FIM.
        """
        s_sub = s_all[::self.trust_sub]
        kl_info_ref = self.actor.get_kl_info(s_sub)

        def F(x):
            with tf.GradientTape() as outtape:
                with tf.GradientTape() as intape:
                    kl_loss = tf.reduce_mean(self.actor.kl(s_sub,kl_info_ref))
                grads = intape.gradient(kl_loss,self.actor.trainable)
                grad_flat = tf.concat(
                    [tf.reshape(grad,[-1]) for grad in grads],-1)
                output = tf.reduce_sum(grad_flat * x)
            result = outtape.gradient(output,self.actor.trainable)
            result_flat = tf.concat(
                [tf.reshape(grad,[-1]) for grad in result],-1)
            result_flat += self.trust_damp * x

            return result_flat

        return F

    def _backtrack(self,eta_v_flat,s_all,a_all,adv_all,neglogp_old_all):
        """Performs backtracking line search and updates policy.

        Args:
            eta_v_flat (np.ndarray): pre backtrack flattened policy update
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
        """
        # Current policy info
        ent = tf.reduce_mean(self.actor.entropy(s_all))
        kl_info_ref = self.actor.get_kl_info(s_all)
        actor_weights_pik = self.actor.get_weights()

        adv_mean = np.mean(adv_all)
        adv_std = np.std(adv_all) + 1e-8

        if self.adv_center:
            adv_all = adv_all - adv_mean 
        if self.adv_scale:
            adv_all = adv_all / adv_std

        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr_before = tf.reduce_mean(ratio * adv_all)

        # Update
        self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
                
        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr = tf.reduce_mean(ratio * adv_all)
        improve = surr - surr_before

        tv = 0.5 * tf.reduce_mean(tf.abs(ratio - 1.))
        kl = tf.reduce_mean(self.actor.kl(s_all,kl_info_ref))

        tv_pre = tv.numpy()
        kl_pre = kl.numpy()

        adj = 1
        for _ in range(10):
            if kl > (self.kl_maxfactor * self.delta):
                pass
            elif improve < 0:
                pass
            else:
                break
            
            # Scale policy update
            factor = np.sqrt(2)
            adj = adj / factor
            eta_v_flat = eta_v_flat / factor

            self.actor.set_weights(actor_weights_pik)
            self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
            
            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(self.actor.kl(s_all,kl_info_ref))
        else:
            # No policy update
            adj = 0
            self.actor.set_weights(actor_weights_pik)

            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(self.actor.kl(s_all,kl_info_ref))
        
        tv = 0.5 * tf.reduce_mean(tf.abs(ratio - 1.))
        
        log_actor = {
            'ent':      ent.numpy(),
            'tv_pre':   tv_pre,
            'kl_pre':   kl_pre,
            'tv':       tv.numpy(),
            'kl':       kl.numpy(),
            'adj':      adj,
            'improve':  improve.numpy()
        }

        return log_actor