import tensorflow as tf
import gym

from sac_eo.common.nn_utils import transform_features, create_nn

class VCritic:
    """State value function."""

    def __init__(self,env,layers,activations,gain):
        """Initializes value function.

        Args:
            env (object): environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer 
                initialization
        """        

        in_dim = gym.spaces.utils.flatdim(env.observation_space)
        self._nn = create_nn(in_dim,1,layers,activations,gain,name='critic')

        self.trainable = self._nn.trainable_variables

    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, self.ret_rms = all_rms

    def _forward(self,s):
        """Returns output of neural network."""
        s_norm = self.s_rms.normalize(s)
        s_feat = transform_features(s_norm)
        return self._nn(s_feat)

    def value(self,s):
        """Calculates value given the state."""
        value = tf.squeeze(self._forward(s),axis=-1)
        value = self.ret_rms.denormalize(value,center=False)
        return value

    def get_loss(self,s,rtg):
        """Returns critic loss."""
        value = self.value(s)
        
        value_norm = self.ret_rms.normalize(value,center=False)
        rtg_norm = self.ret_rms.normalize(rtg,center=False)

        return 0.5 * tf.reduce_mean(tf.square(rtg_norm - value_norm))

    def get_weights(self):
        """Returns parameter weights."""
        return self._nn.get_weights()

    def set_weights(self,weights):
        """Sets parameter weights."""
        self._nn.set_weights(weights)
        
        
class QCritic:
    """State Action Q function."""

    def __init__(self,env,layers,activations,gain):
        """Initializes Q function.
        Args:
            env (NormEnv): normalized environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer 
                initialization
        """        

        in_dim = gym.spaces.utils.flatdim(env.observation_space) + gym.spaces.utils.flatdim(env.action_space)
        self._nn = create_nn(in_dim,1,layers,activations,gain,name='critic')

        self.trainable = self._nn.trainable_variables
        
        
    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, self.a_rms, _, _, self.ret_rms = all_rms

    def _forward(self,s,a):
        """Returns output of neural network."""
        #sa = np.concatenate([s,a],axis=-1) # we want to backpropogate through new action. We need to keep
        #tensor to keep gradients.
        
        s_norm = self.s_rms.normalize(s)
        a_norm = self.a_rms.normalize(a)
        
        sa = tf.concat([s_norm, a_norm], -1)
        sa_feat = transform_features(sa)
        return self._nn(sa_feat)

    def value(self,s,a):
        """Calculates value given the state."""
        #return tf.squeeze(self._forward(s,a),axis=-1)
    
    
        value = tf.squeeze(self._forward(s,a),axis=-1)
        value = self.ret_rms.denormalize(value,center=False)
        return value

    def get_weights(self):
        """Returns parameter weights."""
        return self._nn.get_weights()

    def set_weights(self,weights):
        """Sets parameter weights."""
        self._nn.set_weights(weights)