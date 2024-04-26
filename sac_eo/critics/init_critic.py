"""Interface to critics."""
from sac_eo.critics.critics import VCritic
from sac_eo.critics.critics import QCritic

def init_critics(env,critic_layers,critic_activations,critic_gain,
    critic_weights,num_models,critic_ensemble,critic_init_type, critic_layer_norm):
    """Initializes critic."""

    if critic_ensemble:
        num_critics = num_models
    else:
        num_critics = 1
    
    critics = []
    for idx in range(num_critics):
        critic_active = VCritic(env,critic_layers,critic_activations,
            critic_gain)

        if critic_weights is not None:
            critic_active.set_weights(critic_weights[idx])
        
        critics.append(critic_active)
        
    
    
    q_critics=[]
    q_critics.append( QCritic(env,critic_layers,critic_activations,critic_gain))
    q_critics.append( QCritic(env,critic_layers,critic_activations,critic_gain))
    
    q_targets= []
    q_targets.append( QCritic(env,critic_layers,critic_activations,critic_gain))
    q_targets.append( QCritic(env,critic_layers,critic_activations,critic_gain))
    
    for i in range(len(q_targets)):
        q_targets[i].set_weights(q_critics[i].get_weights())


    return critics,q_targets,q_critics