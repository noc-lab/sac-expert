import gym
import numpy as np
import tensorflow as tf

from sac_eo.actors.base_actor import BaseActor
from sac_eo.common.nn_utils import flat_to_list, list_to_flat

class SoftMaxActor(BaseActor):
    """Softmax policy for discrete action spaces."""

    def __init__(self,env,layers,activations,gain):
        """Initializes softmax policy."""

        assert isinstance(env.action_space,gym.spaces.Discrete), (
            'Only Discrete action space supported')
        
        super(SoftMaxActor,self).__init__(env,layers,activations,gain)
        
        self.trainable = self._nn.trainable_variables
        self.d = np.sum([np.prod(x.shape) for x in self.trainable])

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state.
        
        Args:
            s (np.ndarray): state
            deterministic (bool): if True, returns arg max
        
        Returns:
            Action sampled from current policy.
        """
        a_logits = self._forward(s)
        a_logits = a_logits - tf.reduce_max(a_logits,axis=-1,keepdims=True)

        if deterministic:
            act = tf.argmax(a_logits,axis=-1)
        else:
            # Gumbel trick
            u = np.random.random(size=np.shape(a_logits))
            act = tf.argmax(a_logits - np.log(-np.log(u)),axis=-1)
        
        return tf.squeeze(act)

    def clip(self,a):
        return a
    
    def neglogp(self,s,a):
        a_logits = self._forward(s)
        a_logits = a_logits - tf.reduce_max(a_logits,axis=-1,keepdims=True)

        a_labels = tf.one_hot(a,a_logits.shape[-1])

        neglogp = tf.nn.softmax_cross_entropy_with_logits(a_labels,a_logits)
        return tf.squeeze(neglogp)

    def entropy(self,s):
        logits_cur = self._forward(s)
        logits_cur = logits_cur - tf.reduce_max(
            logits_cur,axis=-1,keepdims=True)
        logsumexp_cur = tf.reduce_logsumexp(logits_cur,axis=-1,keepdims=True)

        prob_num = tf.exp(logits_cur)
        prob_den = tf.reduce_sum(prob_num,axis=-1,keepdims=True)
        prob = prob_num / prob_den

        return tf.reduce_sum(prob * (logits_cur - logsumexp_cur),axis=-1) * -1

    def kl(self,s,kl_info_ref):
        """Calculates KL divergence between current and reference policy.
        
        Args:
            s (np.ndarray): states
            kl_info_ref (tuple): logits for reference policy
        
        Returns:
            np.ndarray of KL divergences between current policy and reference 
            policy at every input state.
        """
        logits_ref = kl_info_ref
        
        logits_cur = self._forward(s)
        logits_cur = logits_cur - tf.reduce_max(
            logits_cur,axis=-1,keepdims=True)
        logsumexp_cur = tf.reduce_logsumexp(logits_cur,axis=-1,keepdims=True)

        logits_ref = logits_ref - tf.reduce_max(
            logits_ref,axis=-1,keepdims=True)
        logsumexp_ref = tf.reduce_logsumexp(logits_ref,axis=-1,keepdims=True)

        prob_num = tf.exp(logits_cur)
        prob_den = tf.reduce_sum(prob_num,axis=-1,keepdims=True)
        prob = prob_num / prob_den

        val = (logits_cur - logsumexp_cur) - (logits_ref - logsumexp_ref)

        return tf.reduce_sum(prob * val,axis=-1)

    def get_kl_info(self,s):
        logits_ref = self._forward(s).numpy()
        return logits_ref

    def get_weights(self,flat=False):
        weights = self._nn.get_weights()
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        self._nn.set_weights(weights)
