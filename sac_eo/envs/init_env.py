"""Interface to environments."""

def init_env(env_type,env_name,task_name):
    """Creates environment.
    
    Args:
        env_type (string): OpenAI Gym (gym) or DeepMind Control (dmc)
        env_name (string): environment / domain name
        task_name (string): task name (dmc only)
    
    Returns:
        Environment.
    """

    if env_type == 'gym':
        from sac_eo.envs.wrappers.gym_wrapper import make_gym_env
        env = make_gym_env(env_name)
    elif env_type == 'dmc':
        from sac_eo.envs.wrappers.dmc_wrapper import make_dmc_env
        env = make_dmc_env(env_name,task_name)
    else:
        raise ValueError('Only gym and dmc env_type supported')
    
    return env