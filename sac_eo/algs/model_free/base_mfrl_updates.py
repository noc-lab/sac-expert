
class BaseOnPolicyUpdate:
    """Base class for on-policy model-free updates."""

    def __init__(self,actor,update_kwargs):
        """Initializes on-policy model-free update class.
        
        Args:
            update_kwargs (dict): dictionary of hyperparameters
        """

        self._setup(update_kwargs)
        self.actor = actor

    def _setup(self,update_kwargs):
        """Sets up hyperparameters as class attributes.
        
        Args:
            update_kwargs (dict): dictionary of hyperparameters
        """
        raise NotImplementedError

    def update(self,rollout_data,expert_reg=None):
        """Updates actor.

        Args:
            rollout_data (tuple): tuple of data collected to use for updates
        """
        raise NotImplementedError
        return log_actor