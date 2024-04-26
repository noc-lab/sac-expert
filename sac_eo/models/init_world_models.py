"""Interface to learned dynamics models."""
from sac_eo.models.continuous_models import GaussianModel
from sac_eo.models.continuous_models import MSEModel

def init_world_models(env,
    model_layers,model_activations,model_gain,model_std_mult,model_weights,
    reward_layers,reward_activations,reward_gain,reward_weights,
    num_models,gaussian_model,model_setup_kwargs):
    """Initializes learned dynamics models."""
    
    models=[]
    for idx in range(num_models):
        if gaussian_model:
            model = GaussianModel(env,model_layers,model_activations,model_gain,
                reward_layers,reward_activations,reward_gain,
                model_setup_kwargs,model_std_mult)
        else:
            model = MSEModel(env,model_layers,model_activations,model_gain,
                reward_layers,reward_activations,reward_gain,
                model_setup_kwargs)
        
        if model_weights is not None:
            model.set_weights(model_weights[idx])
        if reward_weights is not None:
            model.set_reward_weights(reward_weights[idx])
        
        models.append(model)
    
    return models
        