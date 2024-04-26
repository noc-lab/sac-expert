import numpy as np

class TrajectoryCorruptor:
    """Class for corrupting trajectory data."""

    def __init__(self,s_noise_std=0.0,s_noise_type='all'):
        """Initializes TrajectoryCorruptor class.

        Args:
            s_noise_std (float): std dev multiple for state noise
            s_noise_type (str): state noise on 'all' states or 'next' states
        """
        # Corruption noise
        self.s_noise_std = s_noise_std
        self.s_noise_rng = np.random.default_rng(0)
        self.s_noise_type = s_noise_type
    
    def set_rms(self,normalizer):
        """Updates normalizers."""
        all_rms = normalizer.get_rms()
        _, _, _, self.delta_rms, _ = all_rms

    def corrupt_samples(self,sp):
        """Corrupts samples by adding noise."""
        if self.s_noise_std > 0.0:
            u = self.s_noise_rng.normal(size=sp.shape).astype('float32')
            sp_noise = u * np.sqrt(self.delta_rms.var) * self.s_noise_std

            sp = sp + sp_noise

        return sp