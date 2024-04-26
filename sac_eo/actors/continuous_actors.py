import gym
import numpy as np
import tensorflow as tf

from sac_eo.actors.base_actor import BaseActor
from sac_eo.common.nn_utils import transform_features, create_nn,flat_to_list, list_to_flat


class GaussianActor(BaseActor):
    """Multivariate Gaussian policy with diagonal covariance.

    Mean action for a given state is parameterized by a neural network.
    Diagonal covariance is state-independent and parameterized separately.
    """

    def __init__(self,env,layers,activations,gain,init_type,layer_norm,
                 std_mult=1.0,per_state_std=False,output_norm=False):
        """Initializes multivariate Gaussian policy with diagonal covariance.

        Args:
            std_mult (float): multiplicative factor for diagonal covariance 
                initialization
        """

        assert isinstance(env.action_space,gym.spaces.Box), (
            'Only Box action space supported')
        
        super(GaussianActor,self).__init__(env,layers,activations,gain,init_type,layer_norm)
        
        
        self.per_state_std = per_state_std
        self.output_norm = output_norm
        
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        
        
        
        if self.per_state_std:
            self.logstd_init = np.ones((1,)+env.action_space.shape,
                dtype='float32') * (np.log(std_mult) - np.log(np.log(2)))
        else:
            self.logstd_init = np.ones((1,)+env.action_space.shape,
                dtype='float32') * np.log(std_mult)
        
        if self.per_state_std:
            self._nn = create_nn(self.s_dim,2*self.a_dim,
                layers,activations,gain,init_type,layer_norm)
            self.trainable = self._nn.trainable_variables
        
            #self._nn_targ = create_nn(self.s_dim,2*self.a_dim,
                #layers,activations,init_type,gain,layer_norm)
        else:
            self._nn = create_nn(self.s_dim,self.a_dim,
                layers,activations,gain,init_type,layer_norm)
            self.logstd = tf.Variable(np.zeros_like(self.logstd_init),
                dtype=tf.float32)
            self.trainable = self._nn.trainable_variables + [self.logstd]

        #logstd_init = np.ones((1,)+env.action_space.shape) * np.log(std_mult)
        #self.logstd = tf.Variable(logstd_init,dtype=tf.float32,name='logstd')

        #self.trainable = self._nn.trainable_variables + [self.logstd]
        self.d = np.sum([np.prod(x.shape) for x in self.trainable])
        
    
    
    def _output_normalization(self,out):
        """Normalizes output of neural network."""
        out_max = tf.reduce_mean(tf.abs(out),axis=-1,keepdims=True)
        out_max = tf.maximum(out_max,1.0)
        return out / out_max

    def _forward(self,s,targ=False,adversary=False):
        """Returns output of neural network."""
        s_feat = self._transform_state(s)
        
        if targ:
            a_out = self._nn_targ(s_feat)
        else:
            a_out = self._nn(s_feat)
        
        if self.per_state_std:
            a_mean, a_std_out = tf.split(a_out,num_or_size_splits=2,axis=-1)
            a_std = tf.math.softplus(a_std_out)
            a_logstd = tf.math.log(a_std)
        else:
            a_mean = a_out
            if targ:
                a_logstd = self.logstd_targ * tf.ones_like(a_mean)
            else:
                a_logstd = self.logstd * tf.ones_like(a_mean)
        
        a_logstd = a_logstd + self.logstd_init
        a_logstd = tf.maximum(a_logstd,tf.math.log(1e-3))
    
        if self.output_norm:
            a_mean = self._output_normalization(a_mean)
    
        return a_mean, a_logstd
    

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns mean action
        
        Returns:
            Action sampled from current policy.
        """

        a, a_logstd = self._forward(s)

        if not deterministic:
            u = np.random.normal(size=np.shape(a))
            a = a + tf.exp(a_logstd) * u

        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)

        return a

    def clip(self,a):
        return np.clip(a,self.act_low,self.act_high)
    
    def tf_clip(self,a):
        return tf.clip_by_value(a,self.act_low,self.act_high)
    
    
    def neglogp(self,s,a,targ=False,adversary=False):
        a_mean, a_logstd = self._forward(s,targ=targ,adversary=adversary)
    
        neglogp_vec = (tf.square((a - a_mean) / tf.exp(a_logstd)) 
            + 2*a_logstd + tf.math.log(2*np.pi))
    
        return 0.5 * tf.squeeze(tf.reduce_sum(neglogp_vec,axis=-1))
    
    def entropy(self,s,adversary=False):
        _, a_logstd = self._forward(s,adversary=adversary)
        ent_vec = 2*a_logstd + tf.math.log(2*np.pi) + 1
        return 0.5 * tf.reduce_sum(ent_vec,axis=-1)
    
    
    # def neglogp(self,s,a):
    #     a_mean = self._forward(s)

    #     a_vec = (tf.square((a - a_mean) / tf.exp(self.logstd)) 
    #         + 2*self.logstd + tf.math.log(2*np.pi))

    #     return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    # def entropy(self,s):
    #     vec = 2*self.logstd + tf.math.log(2*np.pi) + 1
    #     ent = 0.5 * tf.reduce_sum(vec)
    #     return ent * tf.ones((transform_features(s).shape[0]),dtype=tf.float32)

    def kl(self,s,kl_info_ref,direction='forward'):
        """Calculates KL divergence between current and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (tuple): mean actions and log std. deviation for 
                reference policy
            direction (string): forward or reverse
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        mean_ref, logstd_ref = np.moveaxis(kl_info_ref,-1,0)
        #a_mean = self._forward(s)
        a_mean, a_logstd = self._forward(s)

        if direction == 'forward':
            num = tf.square(a_mean-mean_ref) + tf.exp(2*logstd_ref)
            #vec = num / tf.exp(2*self.logstd) + 2*self.logstd - 2*logstd_ref - 1
            vec = num / tf.exp(2*a_logstd) + 2*a_logstd - 2*logstd_ref - 1
        else:
            num = tf.square(a_mean-mean_ref) + tf.exp(2*self.logstd)
            vec = num / tf.exp(2*logstd_ref) + 2*logstd_ref - 2*self.logstd - 1

        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def get_kl_info(self,s):
        #mean_ref = self._forward(s).numpy()
        #logstd_ref = np.ones_like(mean_ref) * self.logstd.numpy()
        #return np.stack((mean_ref,logstd_ref),axis=-1)
    
        ref_mean, ref_logstd = self._forward(s)
        return np.stack((ref_mean,ref_logstd),axis=-1)

    #def get_weights(self,flat=False):
    #    weights = self._nn.get_weights() + [self.logstd.numpy()]
    #    if flat:
    #        weights = list_to_flat(weights)
    #    
    #    return weights
    
    def get_weights(self,flat=False):
        weights = self._nn.get_weights()
        if not self.per_state_std:
            weights = weights + [self.logstd.numpy()]
        
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False,trpo_backtrack=False):
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        #model_weights = weights[:-1]
        #logstd_weights = weights[-1]
        #logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
        
        #self._nn.set_weights(model_weights)
        #self.logstd.assign(logstd_weights)
        
        
        if self.per_state_std:
            self._nn.set_weights(weights)
        else:
            self._nn.set_weights(weights[:-1])
            
            logstd_weights = weights[-1]
            logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
            self.logstd.assign(logstd_weights)


class SquashedGaussianActor(GaussianActor):
    """Squashed multivariate Gaussian policy with diagonal covariance."""

    def __init__(self,env,layers,activations,init_type,gain,layer_norm,
        std_mult=1.0,per_state_std=False,output_norm=False):
        """Initializes squashed multivariate Gaussian policy with diagonal 
        covariance.
        """       
        super(SquashedGaussianActor,self).__init__(env,layers,activations,
            init_type,gain,layer_norm,std_mult,per_state_std,output_norm)

        self.threshold = 1-1e-3
        self.gaussian_threshold = tf.atanh(self.threshold).numpy()
        self.min_log_std=-5
        self.max_log_std=2
        self.act_limit=env.action_space.high

    def _squash(self,a_gaussian):
        """Squashes Gaussian action."""
        a_gaussian = tf.clip_by_value(a_gaussian,
            -self.gaussian_threshold,self.gaussian_threshold)
        return tf.tanh(a_gaussian)

    def _unsquash(self,a):
        """Unsquashes action."""
        a = tf.clip_by_value(a,-self.threshold,self.threshold)
        return tf.atanh(a)

    def sample_old(self,s,deterministic=False,targ=False):
        a_gaussian = super(SquashedGaussianActor,self).sample(s,deterministic)
        return self._squash(a_gaussian)

    
    def sample(self,s,deterministic=False):
        
        s_feat = self._transform_state(s)
        a_out = self._nn(s_feat)
        
        if self.per_state_std:
            a_mean, a_logstd = tf.split(a_out,num_or_size_splits=2,axis=-1)
            #a_std = tf.math.softplus(a_std_out)
            #a_logstd = tf.math.log(a_std)
        else:
            a_mean = a_out

            a_logstd = self.logstd * tf.ones_like(a_mean)
        
        
        a_logstd =   tf.clip_by_value(a_logstd,
              self.min_log_std,self.max_log_std)
        std = tf.exp(a_logstd)
        
        
        if deterministic == True:
            
            a = a_mean
        else:
            
            u = np.random.normal(size=np.shape(a_mean))
            a = a_mean + std * u
        
        
        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)
            
        
        pi_action = tf.tanh(a)
        pi_action = self.act_limit * pi_action
        
        return pi_action
            
            
            
            
        

    def neglogp(self,s,a,targ=False):
        
        
        a_gaussian = self._unsquash(a)

        neglogp_gaussian = super(SquashedGaussianActor,self).neglogp(s,
            a_gaussian)

        neglogp_correction_dim = 2. * (
            np.log(2.) - a_gaussian - tf.nn.softplus(-2. * a_gaussian))
        neglogp_correction = tf.reduce_sum(neglogp_correction_dim,axis=-1)

        return neglogp_gaussian + neglogp_correction
    
    def evaluate(self,s):
        
        s_feat = self._transform_state(s)
        
        a_out = self._nn(s_feat)
        
        
        if self.per_state_std:
            a_mean, a_logstd = tf.split(a_out,num_or_size_splits=2,axis=-1)
            #a_std = tf.math.softplus(a_std_out)
            #a_logstd = tf.math.log(a_std)
        else:
            a_mean = a_out

            a_logstd = self.logstd * tf.ones_like(a_mean)
        
        
        
        a_logstd =   tf.clip_by_value(a_logstd,
              self.min_log_std,self.max_log_std)
        
        std = tf.exp(a_logstd)
        
        
        u = np.random.normal(size=np.shape(a_mean))
        a = a_mean + std * u

        if np.shape(a)[0] == 1:
            a = tf.squeeze(a,axis=0)
            
        
        
        neglogp_vec = (tf.square((a - a_mean) / tf.exp(a_logstd)) 
            + 2*a_logstd + tf.math.log(2*np.pi))
    
        neglogp= 0.5 * tf.squeeze(tf.reduce_sum(neglogp_vec,axis=-1))
        #neglogp=self.neglogp(s,a)
        
        
        neglogp_correction_dim = 2. * (
            np.log(2.) - a - tf.nn.softplus(-2. * a))
        neglogp_correction = tf.reduce_sum(neglogp_correction_dim,axis=-1)
        
        neglogp_adjusted=neglogp+neglogp_correction
        
        
        
        
        pi_action = tf.tanh(a)
        pi_action = self.act_limit * pi_action
        
        
        return pi_action,neglogp_adjusted
        
        
        #previously, we used the code below, and we get numeric errors.  
        """
        sampled_action=self.sample(s,deterministic=False)
        neglogp=self.neglogp(s,sampled_action)
        
        
        #a_mean,a_logstd = self._forward(s)
        #u = np.random.normal(size=np.shape(a_mean))
        #act = a_mean + tf.exp(a_logstd) * u
        
        #a_vec = (tf.square((act - a_mean) / tf.exp(a_logstd)) 
        #    + 2*a_logstd + tf.math.log(2*np.pi)) #+tf.math.log(1 - action.pow(2) + epsilon)
        #please check : evaluate function of Policy Network in the following link:
            #https://github.com/vaishak2future/sac/blob/master/sac.ipynb
            #I do not have any explanation for having that additional term ?
        #a_vec=0.5 * tf.reduce_sum(a_vec,axis=-1)
        return sampled_action,neglogp
        """

    def entropy(self,s):
        """No closed-form solution. Returns data for sample average estimate."""
        ent_gaussian = super(SquashedGaussianActor,self).entropy(s)
        a = self.sample(s,deterministic=False)
        a_gaussian = self._unsquash(a)

        neglogp_correction_dim = 2. * (
            np.log(2.) - a_gaussian - tf.nn.softplus(-2. * a_gaussian))
        neglogp_correction = tf.reduce_sum(neglogp_correction_dim,axis=-1)

        return ent_gaussian + neglogp_correction
            
            

            
