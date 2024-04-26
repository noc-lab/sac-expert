"""Interface to actors."""
import gym

from sac_eo.actors.continuous_actors import GaussianActor
from sac_eo.actors.continuous_actors import SquashedGaussianActor
from sac_eo.actors.discrete_actors import SoftMaxActor

def init_actor(env,actor_layers,actor_activations,actor_gain,actor_std_mult,
               actor_init_type,actor_layer_norm,actor_weights,
               actor_per_state_std=False, actor_squash=False, actor_output_norm=False,):
    """Initializes actor."""
    if isinstance(env.action_space,gym.spaces.Box):
        
        
        if actor_squash:
            actor = SquashedGaussianActor(env,actor_layers,actor_activations,actor_gain,actor_init_type,actor_layer_norm,
                  actor_std_mult,actor_per_state_std,actor_output_norm)
        
        else:

            actor = GaussianActor(env,actor_layers,actor_activations,actor_gain,actor_init_type,actor_layer_norm,
                                  actor_std_mult,actor_per_state_std,actor_output_norm)
    elif isinstance(env.action_space,gym.spaces.Discrete):
        actor = SoftMaxActor(env,actor_layers,actor_activations,actor_gain)
    else:
        raise TypeError('Only Gym Box and Discrete action spaces supported')

    if actor_weights is not None:
        actor.set_weights(actor_weights)
    
    return actor