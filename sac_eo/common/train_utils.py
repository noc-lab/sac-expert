import os
import pickle
from sac_eo.common.train_parser import all_kwargs


def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict

def import_inputs(inputs_dict):
    """Imports inputs from provided log file."""

    setup_dict = inputs_dict['setup_kwargs']
    
    import_path = setup_dict['import_path']
    import_file = setup_dict['import_file']
    import_idx = setup_dict['import_idx']
    import_all = setup_dict['import_all']

    train_idx = setup_dict['idx'] - setup_dict['runs_start']

    if import_path and import_file:
        import_filefull = os.path.join(import_path,import_file)
        with open(import_filefull,'rb') as f:
            import_log = pickle.load(f)
        
        if import_idx is None:
            if len(import_log) > train_idx:
                import_idx = train_idx
            else:
                import_idx = 0
        else:
            assert import_idx < len(import_log), 'import_idx too large'
        
        import_log_param = import_log[import_idx]['param']

        env_kwargs = import_log_param['env_kwargs']
        actor_kwargs = import_log_param['actor_kwargs']
        critic_kwargs = import_log_param['critic_kwargs']
        model_kwargs = import_log_param['model_kwargs']
        model_setup_kwargs = import_log_param['model_setup_kwargs']

        if import_all:
            inputs_dict = import_log_param
            inputs_dict['setup_kwargs'] = setup_dict
        else:
            inputs_dict['env_kwargs'] = env_kwargs
            inputs_dict['actor_kwargs'] = actor_kwargs
            inputs_dict['critic_kwargs'] = critic_kwargs

        
        import_log_final = import_log[import_idx]['final']
        
        actor_weights = import_log_final['actor_weights']
        critic_weights = import_log_final['critic_weights']
        rms_stats = import_log_final['rms_stats']

        try:
            model_weights = import_log_final['model_weights']
            reward_weights = import_log_final['reward_weights']

            # Only use model inputs if import contains model weights
            inputs_dict['model_kwargs'] = model_kwargs
            inputs_dict['model_setup_kwargs'] = model_setup_kwargs
        except:
            model_weights = None
            reward_weights = None

    else:
        actor_weights = None
        critic_weights = None
        model_weights = None
        reward_weights = None
        rms_stats = None

    inputs_dict['actor_kwargs']['actor_weights'] = actor_weights
    inputs_dict['critic_kwargs']['critic_weights'] = critic_weights
    inputs_dict['model_kwargs']['model_weights'] = model_weights
    inputs_dict['model_kwargs']['reward_weights'] = reward_weights
    inputs_dict['alg_kwargs']['init_rms_stats'] = rms_stats
    
    return inputs_dict

def organize_rms_inputs(logs_rms):
    
        if 'rms_stats' in logs_rms.keys():
            return logs_rms['rms_stats']
        else:
            init_expert_rms_stats=dict()
            
            init_expert_rms_stats['s_rms']=dict()
            init_expert_rms_stats['a_rms']=dict()
            init_expert_rms_stats['r_rms']=dict()
            init_expert_rms_stats['delta_rms']=dict()
            init_expert_rms_stats['ret_rms']=dict()
            
            
            #t, mean, var
            init_expert_rms_stats['s_rms']['t']=logs_rms['s_t']
            init_expert_rms_stats['s_rms']['mean']=logs_rms['s_mean']
            init_expert_rms_stats['s_rms']['var']=logs_rms['s_var']
            
            init_expert_rms_stats['a_rms']['t']=logs_rms['a_t']
            init_expert_rms_stats['a_rms']['mean']=logs_rms['a_mean']
            init_expert_rms_stats['a_rms']['var']=logs_rms['a_var']
            
            init_expert_rms_stats['r_rms']['t']=logs_rms['r_t']
            init_expert_rms_stats['r_rms']['mean']=logs_rms['r_mean']
            init_expert_rms_stats['r_rms']['var']=logs_rms['r_var']
            
            init_expert_rms_stats['delta_rms']['t']=logs_rms['delta_t']
            init_expert_rms_stats['delta_rms']['mean']=logs_rms['delta_mean']
            init_expert_rms_stats['delta_rms']['var']=logs_rms['delta_var']
            
            init_expert_rms_stats['ret_rms']['t']=logs_rms['ret_t']
            init_expert_rms_stats['ret_rms']['mean']=logs_rms['ret_mean']
            init_expert_rms_stats['ret_rms']['var']=logs_rms['ret_var']
            
            return init_expert_rms_stats

