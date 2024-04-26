import gym

from sac_eo.common.nn_utils import transform_features, create_nn 

class BaseActor:
    """Base policy class."""

    def __init__(self,env,layers,activations,gain,init_type,layer_norm):
        """Initializes policy.

        Args:
            env (object): environment
            layers (list): list of hidden layer sizes for neural network
            activations (list): list of activations for neural network
            gain (float): multiplicative factor for final layer initialization
        """

        self.s_dim = gym.spaces.utils.flatdim(env.observation_space)
        self.a_dim = gym.spaces.utils.flatdim(env.action_space)

        #self._nn = create_nn(in_dim,out_dim,layers,activations,gain,init_type,
            #layer_norm,name='actor')

    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        self.s_rms, _, _, _, _ = all_rms

    #def _forward(self,s):
        """Returns output of neural network."""
        #s_norm = self.s_rms.normalize(s)
        #s_feat = transform_features(s_norm)
        #return self._nn(s_feat)
    
    def _transform_state(self,s):
        """Preprocesses state before passing data to neural network."""
        s_norm = self.s_rms.normalize(s)
        s_feat = transform_features(s_norm)
        return s_feat

    def sample(self,s,deterministic=False):
        """Samples an action from the current policy given the state."""
        raise NotImplementedError

    def clip(self,a):
        """Clips action to feasible range."""
        raise NotImplementedError
    
    def neglogp(self,s,a):
        """Calculates negative log probability for given state and action."""
        raise NotImplementedError

    def entropy(self,s):
        """Calculates entropy of current policy."""
        raise NotImplementedError

    def kl(self,s,kl_info_ref):
        """Calculates KL divergence between current and reference policy."""
        raise NotImplementedError

    def get_kl_info(self,s):
        """Returns info needed to calculate KL divergence."""
        raise NotImplementedError

    def get_weights(self,flat=False):
        """Returns parameter weights of current policy.
        
        Args:
            flat (bool): if True, returns weights as flattened np.ndarray
        
        Returns:
            list or np.ndarray of parameter weights.
        """
        raise NotImplementedError
    
    def set_weights(self,weights,from_flat=False,increment=False):
        """Sets parameter weights of current policy.
        
        Args:
            weights (list, np.ndarray): list or np.ndarray of parameter weights
            from_flat (bool): if True, weights are flattened np.ndarray
            increment (bool): if True, weights are incremental values
        """
        raise NotImplementedError
