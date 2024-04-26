import numpy as np

from sac_eo.common.buffer_utils import discounted_sum

class RunningNormalizer:
    """Class that tracks running statistics to use for normalization."""

    def __init__(self,dim):
        """"Initializes RunningNormalizer.

        Args:
            dim (int): dimension of statistic
        """
        self.dim = dim
        self.t_last = 0

        if dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(dim,dtype=np.float32)
            self.var = np.zeros(dim,dtype=np.float32)
            self.std = np.ones(dim,dtype=np.float32)
    
    def normalize(self,data,center=True):
        """Normalizes input data.
        
        Args:
            data (np.ndarray): data to normalize
            center (bool): if True, center data using running mean

        Returns:
            Normalized data.
        """
        if center:
            stand = (data - self.mean) / np.maximum(self.std,1e-8)
        else:
            stand = data / np.maximum(self.std,1e-8)
        
        return stand

    def denormalize(self,data_norm,center=True):
        """Denormalizes input data.

        Args:
            data_norm (np.ndarray): normalized data
            center (bool): whether normalized data has been centered
        
        Returns:
            Denormalized data.
        """
        if center:
            data = data_norm * np.maximum(self.std,1e-8) + self.mean
        else:
            data = data_norm * np.maximum(self.std,1e-8)
        
        return data

    def update(self,data):
        """Updates statistics based on batch of data."""
        std_norm = np.maximum(self.std,1e-8)
        var_norm = np.square(std_norm)
        
        data_norm = data / std_norm
        t_batch = data_norm.shape[0]
        M_batch_norm = data_norm.mean(axis=0)
        S_batch_norm = np.sum(np.square(data_norm - M_batch_norm),axis=0)

        t = t_batch + self.t_last

        self.var =  ((var_norm * S_batch_norm 
            + self.var * np.maximum(1,self.t_last-1)  
            + (t_batch / t) * self.t_last * var_norm * np.square(
                M_batch_norm-self.mean/std_norm)
            ) / np.maximum(1,t-1))

        self.mean = (t_batch * M_batch_norm * std_norm 
            + self.t_last * self.mean) / t

        self.mean = self.mean.astype('float32')
        self.var = self.var.astype('float32')

        if t==1:
            self.std = np.ones_like(self.var)
        else:
            self.std = np.sqrt(self.var)

        self.t_last = t
    
    def reset(self):
        """Resets statistics."""
        self.t_last = 0

        if self.dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(self.dim,dtype=np.float32)
            self.var = np.zeros(self.dim,dtype=np.float32)
            self.std = np.ones(self.dim,dtype=np.float32)

    def instantiate(self,t,mean,var,ignore=None):
        """Instantiates normalizer based on saved statistics."""
        self.t_last = t
        self.mean = mean
        self.var = var
        if self.t_last==0:
            self.reset()
        elif self.t_last==1:
            self.std = np.abs(self.mean)
        else:
            self.std = np.sqrt(self.var)
    
    def get_stats(self):
        """Returns stats needed to recover normalizer."""
        rms_stats = {
            't':    self.t_last,
            'mean': self.mean,
            'var':  self.var
        }
        return rms_stats


class RunningNormalizers:
    """Class that tracks all running normalization stats during training."""

    def __init__(self,s_dim,a_dim,gamma,init_rms_stats=None):
        """Initializes RunningNormalizers class.

        Args:
            s_dim (int): state dimension
            a_dim (int): action dimension
            gamma (float): discount rate
            init_rms_stats (dict): initial normalization stats, if not None
        """       
        self.gamma = gamma

        # Running normalizers
        self.s_rms = RunningNormalizer(s_dim)       # states
        self.a_rms = RunningNormalizer(a_dim)       # actions
        self.r_rms = RunningNormalizer(1)           # rewards
        self.delta_rms = RunningNormalizer(s_dim)   # deltas
        self.ret_rms = RunningNormalizer(1)         # returns

        self.set_rms_stats(init_rms_stats)

    def update_rms(self,s_traj,a_traj,r_traj,sp_traj):
        """Updates running normalization statistics."""
        self.s_rms.update(s_traj)
        self.a_rms.update(a_traj)
        self.r_rms.update(r_traj)
        
        delta_traj = sp_traj - s_traj
        self.delta_rms.update(delta_traj)

        ret_traj = discounted_sum(r_traj,self.gamma)
        self.ret_rms.update(ret_traj)

    def get_rms(self):
        """Returns normalizers needed by other classes."""
        return self.s_rms, self.a_rms, self.r_rms, self.delta_rms, self.ret_rms

    def set_rms_stats(self,init_rms_stats):
        """Sets normalizer statistics."""
        if init_rms_stats is not None:
            self.s_rms.instantiate(**init_rms_stats['s_rms'])
            self.a_rms.instantiate(**init_rms_stats['a_rms'])
            self.r_rms.instantiate(**init_rms_stats['r_rms'])
            self.delta_rms.instantiate(**init_rms_stats['delta_rms'])
            self.ret_rms.instantiate(**init_rms_stats['ret_rms'])

    def get_rms_stats(self):
        """Returns normalizer statistics for logging."""
        s_rms_stats = self.s_rms.get_stats()
        a_rms_stats = self.a_rms.get_stats()
        r_rms_stats = self.r_rms.get_stats()
        delta_rms_stats = self.delta_rms.get_stats()
        ret_rms_stats = self.ret_rms.get_stats()

        rms_stats = {
            's_rms':        s_rms_stats,
            'a_rms':        a_rms_stats,
            'r_rms':        r_rms_stats,
            'delta_rms':    delta_rms_stats,
            'ret_rms':      ret_rms_stats,
        }

        return rms_stats