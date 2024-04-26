import numpy as np
import tensorflow as tf

from sac_eo.common.nn_utils import transform_features, flat_to_list
from sac_eo.models.base_world_model import BaseWorldModel

class GaussianModel(BaseWorldModel):
    """Class for learned Gaussian dynamics models."""

    def __init__(self,env,model_layers,model_activations,model_gain,
        reward_layers,reward_activations,reward_gain,model_setup_kwargs,
        std_mult):
        """Initializes Gaussian world model.

        Args:
            std_mult (float): multiplicative factor for diagonal covariance 
                initialization
        """
        
        super(GaussianModel,self).__init__(env,model_layers,model_activations,
            model_gain,reward_layers,reward_activations,reward_gain,
            model_setup_kwargs)
        
        logstd_init = np.ones((1,self.s_dim)) * np.log(std_mult)
        self.logstd = tf.Variable(logstd_init,dtype=tf.float32,name='logstd')
        
        self.model_trainable = self._nn.trainable_variables+[self.logstd]
        if self.separate_reward_nn:
            self.reward_trainable = self._nn_reward.trainable_variables
            self.trainable = self.model_trainable + self.reward_trainable
        else:
            self.trainable = self.model_trainable
        
        self.d = np.sum([np.prod(x.shape) for x in self.trainable])
    
    def step(self,a):
        """Takes step in learned model."""
        delta_norm_mean, r_norm = self._forward(self.s,a)

        u = np.random.normal(size=np.shape(delta_norm_mean))
        delta_norm = delta_norm_mean + tf.exp(self.logstd) * u
        delta_norm = tf.squeeze(delta_norm).numpy()

        delta = self.delta_rms.denormalize(delta_norm)
        self.s = self.s + delta

        d = tf.squeeze(tf.ones_like(r_norm) == 0).numpy()
        
        r_norm = tf.squeeze(r_norm).numpy()
        r = self.r_rms.denormalize(r_norm)
        
        info = {}

        return self.s, r, d, info

    def sample(self,s,a,deterministic=False):
        """Samples next state from learned model."""
        delta_norm, _ = self._forward(s,a)

        if not deterministic:
            u = np.random.normal(size=np.shape(delta_norm))
            delta_norm = delta_norm + tf.exp(self.logstd) * u

        if np.shape(delta_norm)[0] == 1:
            delta_norm = tf.squeeze(delta_norm,axis=0)

        delta = self.delta_rms.denormalize(delta_norm)
        sp = s + delta

        return sp

    def reset(self,s):
        """Resets learned model to given state and returns state."""
        self.s=s
        return s
    
    def seed(self,seed):
        """Sets random seed."""
        raise NotImplementedError

    def get_weights(self):
        """Returns parameter weights for model network."""
        weights = self._nn.get_weights() + [self.logstd.numpy()]
        return weights

    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights for model network."""
        if from_flat:
            weights = flat_to_list(self.model_trainable,weights)

        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights()))

        nn_weights = weights[:-1]
        logstd_weights = weights[-1]
        
        self._nn.set_weights(nn_weights)
        self.logstd.assign(logstd_weights)

    def get_loss(self,s,sp,a,r):
        """Constructs loss for model fitting."""
        delta_pred_mean, r_pred = self._forward(s,a,clip=False)

        delta = sp - s
        delta_norm = self.delta_rms.normalize(delta)
        if self.delta_clip_loss:
            delta_norm = tf.clip_by_value(delta_norm,
                self.delta_clip_loss*-1,self.delta_clip_loss)

        delta_vec = (tf.square((delta_norm - delta_pred_mean) 
            / tf.exp(self.logstd)) + 2*self.logstd + tf.math.log(2*np.pi))
        delta_neglogp = 0.5 * tf.reduce_sum(delta_vec,axis=-1)

        r_norm = self.r_rms.normalize(r)
        if self.reward_clip_loss:
            r_norm = tf.clip_by_value(r_norm,
                self.reward_clip_loss*-1,self.reward_clip_loss)

        r_loss = 0.5 * tf.square(r_norm - r_pred)

        if self.scale_model_loss:
            delta_scale = tf.stop_gradient(tf.reduce_mean(
                tf.square(tf.exp(self.logstd))
            ))
        else:
            delta_scale = 1.0

        loss_all = delta_scale * delta_neglogp + self.reward_loss_coef * r_loss

        return tf.reduce_mean(loss_all)

    def get_losses_eval(self,s,sp,a,r):
        """Returns MSE and reward loss for evaluation purposes."""
        delta_pred_mean, r_pred = self._forward(s,a,clip=False)

        delta = sp - s
        delta_norm = self.delta_rms.normalize(delta)

        mse_loss_all = 0.5 * tf.reduce_sum(
            tf.square(delta_norm - delta_pred_mean),axis=-1)
        mse_loss = tf.reduce_mean(mse_loss_all)

        r_norm = self.r_rms.normalize(r)
        r_loss_all = 0.5 * tf.square(r_norm - r_pred)
        r_loss = tf.reduce_mean(r_loss_all)

        return mse_loss, r_loss

    def neglogp(self,s,a,sp):
        """Calculates negative log prob for state-action-next state pairs."""
        delta_mean, _ = self._forward(s,a)

        delta = sp - s
        delta_norm = transform_features(self.delta_rms.normalize(delta))

        vec = (tf.square((delta_norm - delta_mean) / tf.exp(self.logstd)) 
            + 2*self.logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(vec,axis=-1))

    def entropy(self,s,a):
        """Returns entropy for each state-action pair."""
        vec = 2*self.logstd + tf.math.log(2*np.pi) + 1
        ent = 0.5 * tf.reduce_sum(vec)
        return ent * tf.ones((transform_features(s).shape[0]),dtype=tf.float32)

    def kl(self,s,a,kl_info_ref,direction='reverse'):
        """Calculates KL divergence between current and reference model. Note
        that calculations are done in normalized space since KL divergence is 
        shift and scale invariant.
        
        Args:
            s (np.ndarray): states
            a (np.ndarray): actions
            kl_info_ref (tuple): mean next states and log std. deviation for 
                reference model
            direction (string): forward or reverse
        
        Returns:
            np.ndarray of KL divergences between current model and reference 
            model at every input state-action pair.
        """
        delta_ref, logstd_ref = np.moveaxis(kl_info_ref,-1,0)
        delta_mean, _ = self._forward(s,a)

        if direction == 'forward':
            num = tf.square(delta_mean - delta_ref) + tf.exp(2*logstd_ref)
            vec = num / tf.exp(2*self.logstd) + 2*self.logstd - 2*logstd_ref - 1
        else:
            num = tf.square(delta_mean - delta_ref) + tf.exp(2*self.logstd)
            vec = num / tf.exp(2*logstd_ref) + 2*logstd_ref - 2*self.logstd - 1

        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def get_kl_info(self,s,a):
        """Returns info needed to calculate KL divergence."""
        delta_ref, _ = self._forward(s,a)
        delta_ref = delta_ref.numpy()
        logstd_ref = np.ones_like(delta_ref) * self.logstd.numpy()
        return np.stack((delta_ref,logstd_ref),axis=-1)



class MSEModel(BaseWorldModel):
    """Class for learned deterministic dynamics model."""

    def __init__(self,env,model_layers,model_activations,model_gain,
        reward_layers,reward_activations,reward_gain,model_setup_kwargs):
        """Initializes MSE world model."""
       
        super(MSEModel,self).__init__(env,model_layers,model_activations,
            model_gain,reward_layers,reward_activations,reward_gain,
            model_setup_kwargs)

        self.model_trainable = self._nn.trainable_variables
        if self.separate_reward_nn:
            self.reward_trainable = self._nn_reward.trainable_variables
            self.trainable = self.model_trainable + self.reward_trainable
        else:
            self.trainable = self.model_trainable
        
        self.d = np.sum([np.prod(x.shape) for x in self.trainable])

    def step(self,a):
        """Takes step in learned model."""
        delta_norm, r_norm = self._forward(self.s,a)

        delta_norm = tf.squeeze(delta_norm).numpy()
        delta = self.delta_rms.denormalize(delta_norm)
        self.s = self.s + delta

        d = tf.squeeze(tf.ones_like(r_norm) == 0).numpy()

        r_norm = tf.squeeze(r_norm).numpy()
        r = self.r_rms.denormalize(r_norm)
        
        info = {}
        """
        TO DO reward and termination condition check?
        """
        return self.s, r, d, info

    def sample(self,s,a,deterministic=True):
        """Samples next state from learned model."""
        delta_norm, _ = self._forward(s,a)

        if np.shape(delta_norm)[0] == 1:
            delta_norm = tf.squeeze(delta_norm,axis=0)

        delta = self.delta_rms.denormalize(delta_norm)
        sp = s + delta

        return sp

    def reset(self,s):
        """Resets learned model to given state and returns state."""
        self.s=s
        return s
    
    def seed(self,seed):
        """Sets random seed."""
        raise NotImplementedError

    def get_weights(self):
        """Returns parameter weights for model network."""
        return self._nn.get_weights()

    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights for model network."""
        if from_flat:
            weights = flat_to_list(self.model_trainable,weights)

        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights()))
        
        self._nn.set_weights(weights)

    def get_loss(self,s,sp,a,r):
        """Constructs loss for model fitting."""
        delta_pred, r_pred = self._forward(s,a,clip=False)

        delta = sp - s
        delta_norm = self.delta_rms.normalize(delta)
        if self.delta_clip_loss:
            delta_norm = tf.clip_by_value(delta_norm,
                self.delta_clip_loss*-1,self.delta_clip_loss)

        delta_loss = 0.5 * tf.reduce_sum(
            tf.square(delta_norm - delta_pred),axis=-1)

        r_norm = self.r_rms.normalize(r)
        if self.reward_clip_loss:
            r_norm = tf.clip_by_value(r_norm,
                self.reward_clip_loss*-1,self.reward_clip_loss)
        
        r_loss = 0.5 * tf.square(r_norm - r_pred)

        loss_all = delta_loss + self.reward_loss_coef * r_loss

        return tf.reduce_mean(loss_all)

    def get_losses_eval(self,s,sp,a,r):
        """Returns MSE and reward loss for evaluation purposes."""
        delta_pred, r_pred = self._forward(s,a,clip=False)

        delta = sp - s
        delta_norm = self.delta_rms.normalize(delta)

        mse_loss_all = 0.5 * tf.reduce_sum(
            tf.square(delta_norm - delta_pred),axis=-1)
        mse_loss = tf.reduce_mean(mse_loss_all)

        r_norm = self.r_rms.normalize(r)
        r_loss_all = 0.5 * tf.square(r_norm - r_pred)
        r_loss = tf.reduce_mean(r_loss_all)

        return mse_loss, r_loss

    def entropy(self,s,a):
        """Placeholder for logging."""
        return tf.zeros((transform_features(s).shape[0]),dtype=tf.float32)

    def kl(self,s,a,kl_info_ref,direction='reverse'):
        """MSE between current and reference model (name for compatibility).
        
        Args:
            s (np.ndarray): states
            a (np.ndarray): actions
            kl_info_ref (tuple): next states and log std. deviation placeholder 
                for reference model
            direction (string): unused in MSE model
        
        Returns:
            np.ndarray of MSE between current model and reference 
            model at every input state-action pair.
        """
        delta_ref, _ = np.moveaxis(kl_info_ref,-1,0)
        delta_mean, _ = self._forward(s,a)

        return 0.5 * tf.reduce_sum(tf.square(delta_mean - delta_ref),axis=-1)

    def get_kl_info(self,s,a):
        """Returns info needed to calculate MSE (name for compatibility)."""
        delta_ref, _ = self._forward(s,a)
        delta_ref = delta_ref.numpy()
        logstd_ref = np.ones_like(delta_ref)
        return np.stack((delta_ref,logstd_ref),axis=-1)