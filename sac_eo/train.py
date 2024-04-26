"""Entry point for RL training."""
import os
#import sys
#sys.path.append("/Users/can/Documents/GitHub/mbrl_with_expert")
#sys.path.append("/home/erhan/mbrl_cdc")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

from datetime import datetime
import pickle
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import copy

from sac_eo.envs import init_env
from sac_eo.actors import init_actor
from sac_eo.critics import init_critics
from sac_eo.models import init_world_models
from sac_eo.algs import init_alg
from sac_eo.common.seeding import init_seeds
from sac_eo.common.train_parser import create_train_parser
from sac_eo.common.train_utils import gather_inputs, import_inputs,organize_rms_inputs

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def train(inputs_dict):
    """Training on given seed."""

    idx = inputs_dict['setup_kwargs']['idx']
    setup_seed = inputs_dict['setup_kwargs']['setup_seed']
    sim_seed = inputs_dict['setup_kwargs']['sim_seed']
    eval_seed = inputs_dict['setup_kwargs']['eval_seed']
    exp_seed = inputs_dict['setup_kwargs']['expert_seed']
    alg_seed = inputs_dict['setup_kwargs']['algorithm_seed']
    inputs_dict['alg_kwargs']['alg_seed']=alg_seed
    
    env_kwargs = inputs_dict['env_kwargs']
    actor_kwargs = inputs_dict['actor_kwargs']
    critic_kwargs = inputs_dict['critic_kwargs']
    model_kwargs = inputs_dict['model_kwargs']
    model_setup_kwargs = inputs_dict['model_setup_kwargs']
    alg_kwargs = inputs_dict['alg_kwargs']
    mf_update_kwargs = inputs_dict['mf_update_kwargs']

    total_timesteps = inputs_dict['alg_kwargs']['total_timesteps']
    
    

        
    

    init_seeds(setup_seed)
    env = init_env(**env_kwargs)
    env_eval = init_env(**env_kwargs)
    env_expert = init_env(**env_kwargs)
    actor = init_actor(env,**actor_kwargs)
    
    if inputs_dict['setup_kwargs']['expert_file'] is not None:
        import_filefull = os.path.join(inputs_dict['setup_kwargs']['expert_path'],inputs_dict['setup_kwargs']['expert_file'])
        with open(import_filefull,'rb') as f:
            import_logs = pickle.load(f)
            
        import_log=import_logs[0]
        expert_kwargs=import_log['param']['actor_kwargs']
        #import_log=import_log['final']
        
        expert_kwargs['actor_weights']=import_log['final']['actor_weights']
        
        if expert_kwargs['actor_squash'] is not None:
            expert_kwargs.pop('actor_squash')
        if expert_kwargs['actor_adversary_prob'] is not None:
            expert_kwargs.pop('actor_adversary_prob')
        
        
        logs_rms=import_logs[0]['final']
        init_expert_rms_stats=organize_rms_inputs(logs_rms)
        

    else:
        expert_kwargs=actor_kwargs
        expert_kwargs['actor_weights']=None
        init_expert_rms_stats=None
    
    expert = init_actor(env,**expert_kwargs)
    
    critics,q_targets,q_critics = init_critics(env,**critic_kwargs)
    models = init_world_models(env,**model_kwargs,
        model_setup_kwargs=model_setup_kwargs)
    #perturbed_models = init_world_models(env,**model_kwargs,
    #    model_setup_kwargs=model_setup_kwargs)
    
    init_seeds(eval_seed,env_eval)
    init_seeds(sim_seed,env)
    init_seeds(exp_seed,env_expert)
    alg = init_alg(idx,env,env_eval,env_expert,actor,critics,q_targets,q_critics,models,
        alg_kwargs,mf_update_kwargs,expert,init_expert_rms_stats)
    
    log_name = alg.train(total_timesteps,inputs_dict)

    return log_name

def main():
    """Parses inputs, runs simulations, saves data."""
    start_time = datetime.now()
    
    parser = create_train_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(5)
    setup_seeds = np.random.SeedSequence(seeds[0]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    sim_seeds = np.random.SeedSequence(seeds[1]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    eval_seeds = np.random.SeedSequence(seeds[2]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    expert_seeds = np.random.SeedSequence(seeds[3]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]
    algorithm_seeds = np.random.SeedSequence(seeds[4]).generate_state(
        args.runs+args.runs_start)[args.runs_start:]

    inputs_list = []
    for run in range(args.runs):
        inputs_dict['setup_kwargs']['idx'] = run + args.runs_start
        if args.setup_seed is None:
            inputs_dict['setup_kwargs']['setup_seed'] = int(setup_seeds[run])
        if args.sim_seed is None:
            inputs_dict['setup_kwargs']['sim_seed'] = int(sim_seeds[run])
        if args.eval_seed is None:
            inputs_dict['setup_kwargs']['eval_seed'] = int(eval_seeds[run])
        if args.expert_seed is None:
            inputs_dict['setup_kwargs']['expert_seed'] = int(expert_seeds[run])
        if args.alg_seed is None:
            inputs_dict['setup_kwargs']['algorithm_seed'] = int(algorithm_seeds[run])
        
        inputs_dict = import_inputs(inputs_dict)

        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        args.cores = args.runs

    with mp.get_context('spawn').Pool(args.cores) as pool:
        log_names = pool.map(train,inputs_list)

    #log_names = []
    #for run in range(args.runs):
    #    log_name = train(inputs_list[run])
    #    log_names.append(log_name)
    
    # Aggregate results
    outputs = []
    for log_name in log_names:
        os.makedirs(args.save_path,exist_ok=True)
        filename = os.path.join(args.save_path,log_name)
        
        with open(filename,'rb') as f:
            log_data = pickle.load(f)
        
        outputs.append(log_data)

    # Save data
    save_env_type = args.env_type.lower()
    save_env = args.env_name.split('-')[0].lower()
    if args.task_name is not None:
        save_env = '%s_%s'%(save_env,args.task_name.lower())
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    if args.save_file is None:
        save_file = '%s_%s_%s_%s_%s'%(save_env_type,save_env,
            args.alg_type,args.mf_algo,save_date)
    else:
        save_file = '%s_%s_%s_%s_%s_%s'%(save_env_type,save_env,
            args.alg_type,args.mf_algo,args.save_file,save_date)

    os.makedirs(args.save_path,exist_ok=True)
    save_filefull = os.path.join(args.save_path,save_file)

    with open(save_filefull,'wb') as f:
        pickle.dump(outputs,f)
    
    for log_name in log_names:
        filename = os.path.join(args.save_path,log_name)
        os.remove(filename)
    
    end_time = datetime.now()
    print('Time Elapsed: %s'%(end_time-start_time))

if __name__=='__main__':
    main()