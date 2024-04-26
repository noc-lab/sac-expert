import numpy as np
import gym
import tensorflow as tf

from sac_eo.common.nn_utils import transform_features, create_nn 

class BaseWorldModel:
    """Base class for learned dynamics models."""

    def __init__(self,env,model_layers,model_activations,model_gain,
        reward_layers,reward_activations,reward_gain,model_setup_kwargs):
        """Initializes BaseWorldModel class.

        Args:
            env (object): environment
            model_layers (list): list of hidden layer sizes for model NN
            model_activations (list): list of activations for model NN
            model_gain (float): mult factor for final layer model NN init
            reward_layers (list): list of hidden layer sizes for reward NN
            reward_activations (list): list of activations for reward NN
            reward_gain (float): mult factor for final layer reward NN init
            model_setup_kwargs (dict): model setup parameters
        """
        
        self._setup(model_setup_kwargs)
        
        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)
        
        in_dim = self.s_dim + self.a_dim
        
        if self.separate_reward_nn:
            out_dim = self.s_dim
            self._nn = create_nn(in_dim,out_dim,model_layers,model_activations,
                model_gain,name='model')
            self._nn_reward = create_nn(in_dim,1,reward_layers,
                reward_activations,reward_gain,name='reward')
        else:
            out_dim = self.s_dim + 1 # +1 because of reward
            self._nn = create_nn(in_dim,out_dim,model_layers,model_activations,
                model_gain,name='model')

    def _setup(self,model_setup_kwargs):
        """Sets up hyperparameters as class attributes.
        
        Args:
            model_setup_kwargs (dict): dictionary of hyperparameters
        """

        self.separate_reward_nn = model_setup_kwargs['separate_reward_nn']
        self.reward_loss_coef = model_setup_kwargs['reward_loss_coef']
        self.scale_model_loss = model_setup_kwargs['scale_model_loss']
        
        self.delta_clip_loss = model_setup_kwargs['delta_clip_loss']
        self.reward_clip_loss = model_setup_kwargs['reward_clip_loss']

        self.delta_clip_pred = model_setup_kwargs['delta_clip_pred']
        self.reward_clip_pred = model_setup_kwargs['reward_clip_pred']
    
    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, self.a_rms, self.r_rms, self.delta_rms, _ = all_rms

    def _forward(self,s,a,clip=True):
        """Returns output of neural network."""
        s_norm = self.s_rms.normalize(s)
        a_norm = self.a_rms.normalize(a)
        sa_norm = tf.concat([s_norm,a_norm],axis=-1)
        sa_feat = transform_features(sa_norm)
        
        if self.separate_reward_nn:
            delta = self._nn(sa_feat)
            r = tf.squeeze(self._nn_reward(sa_feat),axis=-1)
        else:
            predictions = self._nn(sa_feat)
            delta = predictions[:,:-1]
            r = predictions[:,-1]

        if clip and self.delta_clip_pred:
            delta = tf.clip_by_value(delta,
                self.delta_clip_pred*-1,self.delta_clip_pred)
        if clip and self.reward_clip_pred:
            r = tf.clip_by_value(r,
                self.reward_clip_pred*-1,self.reward_clip_pred)
        
        return delta, r

    def step(self,a):
        """Takes step in learned model."""
        raise NotImplementedError
        return s, r, d, info

    def sample(self,s,a):
        """Samples next state from learned model."""
        raise NotImplementedError
        return sp

    def reset(self,s):
        """Resets learned model to given state and returns state."""
        raise NotImplementedError
        return s
    
    def seed(self,seed):
        """Sets random seed."""
        raise NotImplementedError

    def get_weights(self):
        """Returns parameter weights for model network."""
        raise NotImplementedError
        return weights

    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights for model network.

        Args:
            weights (list, np.ndarray): list or np.ndarray of parameter weights
            from_flat (bool): if True, weights are flattened np.ndarray
            increment (bool): if True, weights are incremental values
        """
        raise NotImplementedError

    def get_reward_weights(self):
        """Returns parameter weights for reward network."""
        if self.separate_reward_nn:
            return self._nn_reward.get_weights()
        else:
            return None

    def set_reward_weights(self,weights):
        """Sets parameter weights for reward network."""
        if self.separate_reward_nn:
            self._nn_reward.set_weights(weights)

    def get_loss(self,s,sp,a,r):
        """Constructs loss for model fitting."""
        raise NotImplementedError
        return loss