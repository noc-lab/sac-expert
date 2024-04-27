"""Creates command line parser for train.py."""
import argparse

parser = argparse.ArgumentParser()

# Setup
#########################################
setup_kwargs = [
    'runs','runs_start','cores','seed','setup_seed','sim_seed','eval_seed', 'expert_seed',
    'save_path','save_file','import_path','import_file','import_idx',
    'import_all','expert_file','expert_path'
]

parser.add_argument('--runs',help='number of trials',type=int,default=1)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)

parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--setup_seed',help='setup seed',type=int)
parser.add_argument('--sim_seed',help='simulation seed',type=int)
parser.add_argument('--eval_seed',help='simulation seed',type=int)
parser.add_argument('--expert_seed',help='simulation seed',type=int)
parser.add_argument('--alg_seed',help='algorithm seed',type=int)

parser.add_argument('--save_path',help='save path',type=str,default='./logs')
parser.add_argument('--save_file',help='save file name',type=str)

parser.add_argument('--import_path',help='import path',type=str,
    default='./logs')
parser.add_argument('--import_file',help='import file name',type=str)
parser.add_argument('--import_idx',help='import index',type=int)
parser.add_argument('--import_all',help='import all params',action='store_true')
#parser.add_argument('--expert_file',help='import file name',type=str,default='gym_pendulum_mfrl_mpo_051023_144258')
parser.add_argument('--expert_file',help='import file name',type=str)
parser.add_argument('--expert_path',help='import file name',type=str,default='./experts')

# Environment initialization
#########################################
env_kwargs = [
    'env_type','env_name','task_name'
]

parser.add_argument('--env_type',help='environment type',type=str,default='gym')
parser.add_argument('--env_name',help='environment name',type=str,
    default='Pendulum-v1')
parser.add_argument('--task_name',help='task name',type=str)

# Actor initialization
#########################################
actor_kwargs = [
    'actor_layers','actor_activations','actor_gain','actor_std_mult',
    'actor_init_type','actor_layer_norm','actor_per_state_std','actor_squash'
]

parser.add_argument('--actor_layers',nargs='+',
    help='list of hidden layer sizes for actor',type=int,default=[64,64])
parser.add_argument('--actor_activations',nargs='+',
    help='list of activations for actor',type=str,default=['tanh'])
parser.add_argument('--actor_gain',
    help='mult factor for final layer of actor',type=float,default=0.01)
parser.add_argument('--actor_std_mult',
    help='initial policy std deviation multiplier',type=float,default=1.0)
parser.add_argument('--actor_init_type',help='actor initialization type',
    type=str,default='orthogonal')
parser.add_argument('--actor_layer_norm',
    help='use layer norm on first layer',action='store_true')
parser.add_argument('--actor_per_state_std',
    help='use state dependent std deviation',action='store_true')#store_true
parser.add_argument('--actor_squash',
    help='squash actions with tanh',action='store_true')

# Critic initialization
#########################################
critic_kwargs = [
    'critic_layers','critic_activations','critic_gain','critic_ensemble',
    'num_models','critic_init_type','critic_layer_norm'
]

parser.add_argument('--critic_layers',nargs='+',
    help='list of hidden layer sizes for value function',
    type=int,default=[64,64])
parser.add_argument('--critic_activations',nargs='+',
    help='list of activations for value function',type=str,default=['tanh'])
parser.add_argument('--critic_gain',
    help='mult factor for final layer of value function',type=float,default=1.0)
parser.add_argument('--critic_ensemble',help='use one critic per model',
    action='store_true')
parser.add_argument('--critic_init_type',help='critic initialization type',
    type=str,default='orthogonal')
parser.add_argument('--critic_layer_norm',
    help='use layer norm on first layer',action='store_true')

# Model initialization
#########################################

model_kwargs = [
    'gaussian_model','num_models',
    'model_layers','model_activations','model_gain','model_std_mult',
    'reward_layers','reward_activations','reward_gain',
]

parser.add_argument('--gaussian_model',help='use Gaussian model',
    action='store_true')
parser.add_argument('--num_models',help='number of models in ensemble',
    type=int,default=2)
parser.add_argument('--model_layers',nargs='+',
    help='list of hidden layer sizes for model',type=int,default=[512,512])
parser.add_argument('--model_activations',nargs='+',
    help='list of activations for model',type=str,default=['relu'])
parser.add_argument('--model_gain',
    help='mult factor for final layer of model',type=float,default=0.01)
parser.add_argument('--model_std_mult',
    help='initial model std deviation multiplier',type=float,default=1.0)

parser.add_argument('--reward_layers',nargs='+',
    help='list of hidden layer sizes for reward NN',type=int,default=[512,512])
parser.add_argument('--reward_activations',nargs='+',
    help='list of activations for reward NN',type=str,default=['relu'])
parser.add_argument('--reward_gain',
    help='mult factor for final layer of reward NN',type=float,default=0.01)

# Model setup parameters
#########################################

model_setup_kwargs = [
    'separate_reward_nn','reward_loss_coef','scale_model_loss',
    'delta_clip_loss','reward_clip_loss','delta_clip_pred','reward_clip_pred'
]

parser.add_argument('--separate_reward_nn',help='separate network for reward',
    action='store_true')
parser.add_argument('--reward_loss_coef',
    help='reward loss coefficient for model loss',type=float,default=1.0)
parser.add_argument('--scale_model_loss',
    help='scale model loss by average variance',action='store_true')

parser.add_argument('--delta_clip_loss',
    help='clipping parameter for normalized state deltas in model loss',
    type=float)
parser.add_argument('--reward_clip_loss',
    help='clipping parameter for normalized reward in model loss',
    type=float)

parser.add_argument('--delta_clip_pred',
    help='clipping parameter for normalized state delta predictions',
    type=float)
parser.add_argument('--reward_clip_pred',
    help='clipping parameter for normalized reward predictions',
    type=float)

# Algorithm parameters
#########################################

# Buffer parameters
# ---------------------------

buffer_kwargs = ['gamma','lam','env_buffer_size','sim_buffer_size','model_buffer_size','expert_buffer_size']

parser.add_argument('--gamma',help='discount rate',type=float,default=0.995)
parser.add_argument('--lam',help='GAE parameter',type=float,default=0.97)
parser.add_argument('--env_buffer_size',help='real data buffer size',
    type=float)
parser.add_argument('--sim_buffer_size',help='simulated data buffer size',
    type=float)
parser.add_argument('--model_buffer_size',help='model buffer size to store env data',
    type=float)
parser.add_argument('--expert_buffer_size',help='simulated data buffer size',
    type=float,default=20)

# Training parameters
# ---------------------------

train_kwargs = [
    'save_path','checkpoint_file','save_freq',
    'eval_freq','eval_num_traj',
    'alg_type','mf_algo','total_timesteps',
    'env_horizon','env_batch_type','env_batch_size_init','env_batch_size',
    's_noise_std','s_noise_type',
    'sim_horizon','sim_batch_type','sim_batch_size','exp_batch_type'
]

parser.add_argument('--checkpoint_file',help='checkpoint file name',type=str,
    default='TEMPLOG')
parser.add_argument('--save_freq',help='how often to store temp files',
    type=float)

parser.add_argument('--eval_freq',help='how often to evaluate policy',
    type=float)
parser.add_argument('--eval_num_traj',help='number of trajectories for eval',
    type=int,default=5)

parser.add_argument('--alg_type',help='type of algorithm',type=str,
    default='sac_imit')
parser.add_argument('--mf_algo',help='model-free algorithm',type=str,
    default='trpo')
parser.add_argument('--total_timesteps',
    help='total number of real timesteps for training',type=float,default=5e5)

parser.add_argument('--env_horizon',
    help='rollout horizon for real trajectories',type=int,default=1000)
parser.add_argument('--env_batch_type',help='real data batch type',
    type=str,default='steps',choices=['steps','traj'])
parser.add_argument('--env_batch_size_init',
    help='real data batch size (initial batch)',type=int,default=5000)
parser.add_argument('--env_batch_size',
    help='real data batch size',type=int,default=3000)

parser.add_argument('--s_noise_std',help='std dev multiple for state noise',
    type=float,default=0.0)
parser.add_argument('--s_noise_type',help='state noise type',
    type=str,default='all',choices=['all','next'])

parser.add_argument('--sim_horizon',
    help='rollout horizon for simulated trajectories',type=int,default=5)
parser.add_argument('--sim_batch_type',help='simulated data batch type',
    type=str,default='steps',choices=['steps','traj'])
parser.add_argument('--sim_batch_size',
    help='total simulated data batch size',type=int,default=10000)
parser.add_argument('--exp_batch_type',help='expert real data batch type',
    type=str,default='steps',choices=['steps','traj'])

# Model update parameters
# ---------------------------
model_update_kwargs = [
    'model_lr','model_num_epochs','model_batch_size','model_batch_shuffle',
    'model_max_updates','model_max_grad_norm',
    'model_holdout_ratio','model_holdout_epochs','reset_model_optimizer'
]

parser.add_argument('--model_lr',help='learning rate for model fitting',
    type=float,default=1e-3)
parser.add_argument('--model_num_epochs',
    help='number of epochs for model update',type=int,default=10)
parser.add_argument('--model_batch_size',help='model update batch size',
    type=int,default=200)
parser.add_argument('--no_model_batch_shuffle',
    help='do not use different minibatches per model for model learning',
    dest='model_batch_shuffle',default=True,action='store_false')

parser.add_argument('--model_max_updates',
    help='max number of gradient steps for model update',type=float,default=1e5)
parser.add_argument('--model_max_grad_norm',help='max model gradient norm',
    type=float)
parser.add_argument('--model_holdout_ratio',
    help='ratio of holdout data for validation',type=float,default=0.0)
parser.add_argument('--model_holdout_epochs',
    help='max epochs without validation set improvement',type=int,default=5)
parser.add_argument('--reset_model_optimizer',help='resets optimizer stats at the end of model training'
                    ,action='store_true')

# Actor-critic update parameters
# ---------------------------

ac_update_kwargs = [
    'critic_lr','critic_update_it','critic_nminibatch','num_mf_updates'
]

parser.add_argument('--critic_lr',help='critic optimizer learning rate',
    type=float,default=3e-4)
parser.add_argument('--critic_update_it',
    help='number of epochs per critic update',type=int,default=10)
parser.add_argument('--critic_nminibatch',
    help='number of minibatches per epoch in critic update',type=int,default=32)

parser.add_argument('--num_mf_updates',
    help='number of policy updates before updating model',type=int,default=25)

# mbrl_imit_kwargs parameters
# ---------------------------

mbrl_imit_kwargs = [
    'epsilon','scale_epsilon_by_true_MSE','scale_max_disc','scale_median_disc','scale_total_disc',
    'use_expert_actions','min_mult','exp_mult','mult_coeff',
    'init_from_expert','max_exp_state_ratio'
]

parser.add_argument('--epsilon',help='expert regularization coefficient of the actor update',
    type=float,default=1e-3)
parser.add_argument('--scale_epsilon_by_true_MSE',help='USE expert true MSE to for adaptive epsilon',
    action='store_true')
parser.add_argument('--scale_max_disc',help='USE maximum discrepancy of models on expert data tfor adaptive epsilon',
    action='store_true')
parser.add_argument('--scale_median_disc',help='USE median discrepancy of models on expert data tfor adaptive epsilon',
    action='store_true')
parser.add_argument('--scale_total_disc',help='USE total discrepancy of models on expert data tfor adaptive epsilon',
    action='store_true')
parser.add_argument('--use_expert_actions',help='Use expert actions in epsilon calculation',
    action='store_true')
parser.add_argument('--min_mult',help='scale epsilon by min multiplicative',
    action='store_true')
parser.add_argument('--exp_mult',help='scale epsilon by exp multiplicative',
    action='store_true')
parser.add_argument('--mult_coeff',help='pos. integers for exp_mult and fractions between 0 and 1 for min_mult',
    type=float, default=1.0)
parser.add_argument('--init_from_expert',help='counterfactual trajectories are intiated from expert states',
    action='store_true')
parser.add_argument('--max_exp_state_ratio',help='At most this percentage of counterfactual trajectories will be started from expert states',
    type=float, default=0.25)


mbpo_kwargs = ['init_temperature',
    'q_crit_lr','mbpo_actor_lr','mbpo_alpha_lr','mbpo_E','mbpo_G','mbpo_M','sac_batch_size','expert_batch_size','soft_tau',
    'target_update_int','real_step_mod','random_act', 'update_normalizers','only_model_normalizer', 'adaptive_model_horizon',
    'modelhorx','modelhory','modelhora','modelhorb'
]

parser.add_argument('--init_temperature',help='initial value of temperature',
    type=float,default=1e-1)
parser.add_argument('--q_crit_lr',help='learning rate for q critics',
    type=float,default=3e-4)
parser.add_argument('--mbpo_actor_lr',help='learning rate for actor',
    type=float,default=1e-4)
parser.add_argument('--mbpo_alpha_lr',help='learning rate for alpha',
    type=float,default=1e-4)
parser.add_argument('--mbpo_E',
    help='denotes realstep per model update',type=int,default=1000)#default value in MBPO paper
parser.add_argument('--mbpo_G',
    help='Policy updates per real step ',type=int,default=3)#default is either 20 or 40.
parser.add_argument('--mbpo_M',
    help='denotes the number of simulated trajectories',type=int,default=400)#default value in MBPO paper
parser.add_argument('--sac_batch_size',
    help='randomly selects samples from simulated data',type=int,default=256)#default value must be checked.
parser.add_argument('--expert_batch_size',
    help='randomly selects samples from expert data',type=int)#default value must be None.
parser.add_argument('--soft_tau',
    help='combines parameters of value and value bar functions',type=float,default=5e-3)
parser.add_argument('--target_update_int',
    help='Q targets are updated per target_update_int iterations',type=int,default=1)#Need to check code.
#https://arxiv.org/pdf/1801.01290.pdf see Appendix D.
parser.add_argument('--real_step_mod',
    help='repeat updates after this many real steps',type=int,default=3)
parser.add_argument('--random_act',help='samples random actions to collect data',
    action='store_true')
parser.add_argument('--update_normalizers',help='update normalizer stats by using collected real data',
    action='store_true')
parser.add_argument('--only_model_normalizer',help='only updates model normalizer by using collected real data',
    action='store_true')
parser.add_argument('--adaptive_model_horizon',help='adaptive model horizon',
    action='store_true')
parser.add_argument('--modelhorx',help='model horizon x',
    type=float,default=1)
parser.add_argument('--modelhory',help='model horizon x',
    type=float,default=15)
parser.add_argument('--modelhora',help='model horizon x',
    type=float,default=20)
parser.add_argument('--modelhorb',help='model horizon x',
    type=float,default=100)



# Combined
# ---------------------------
alg_kwargs = buffer_kwargs + train_kwargs + model_update_kwargs + ac_update_kwargs + mbrl_imit_kwargs + mbpo_kwargs



# Model-free update parameters
#########################################

# Shared
# ---------------------------
mf_shared_kwargs = [
    'adv_center','adv_scale','ent_reg','alpha_lr'
]

parser.add_argument('--no_adv_center',help='do not center advantages',
    dest='adv_center',default=True,action='store_false')
parser.add_argument('--no_adv_scale',help='do not scale advantages',
    dest='adv_scale',default=True,action='store_false')

parser.add_argument('--ent_reg',help='use entropy regularization',
    action='store_true')
parser.add_argument('--alpha_lr',help='entropy parameter learning rate',
    type=float,default=3e-4)

# TRPO
# ---------------------------
mf_trpo_kwargs = ['delta_trpo','cg_it','trust_sub','trust_damp','kl_maxfactor']

parser.add_argument('--delta_trpo',help='TRPO trust region parameter',
    type=float,default=0.02)
parser.add_argument('--cg_it',help='conjugate gradient iterations',
    type=int,default=20)
parser.add_argument('--trust_sub',help='trust region subsampling factor',
    type=int,default=1)
parser.add_argument('--trust_damp',help='trust region damping coefficient',
    type=float,default=0.01)
parser.add_argument('--kl_maxfactor',
    help='mult factor for TRPO backtracking line search',type=float,default=1.5)

# PPO
# ---------------------------
mf_ppo_kwargs = [
    'actor_update_it','actor_nminibatch','actor_lr','eps_ppo','max_grad_norm',
    'adaptlr','adapt_factor','adapt_minthresh','adapt_maxthresh'
]

parser.add_argument('--actor_update_it',
    help='number of epochs per actor update',type=int,default=10)
parser.add_argument('--actor_nminibatch',
    help='number of minibatches per epoch in actor update',type=int,default=32)

parser.add_argument('--actor_lr',help='actor learning rate',
    type=float,default=3e-4)
parser.add_argument('--eps_ppo',help='PPO clipping parameter',
    type=float,default=0.2)
parser.add_argument('--max_grad_norm',help='max policy gradient norm',
    type=float,default=0.5)

parser.add_argument('--no_adaptlr',help='do not adapt LR based on TV',
    dest='adaptlr',default=True,action='store_false')
parser.add_argument('--adapt_factor',help='factor used to adapt LR',
    type=float,default=0.03)
parser.add_argument('--adapt_minthresh',
    help='min multiple of TV for adapting LR',type=float,default=0.0)
parser.add_argument('--adapt_maxthresh',
    help='max multiple of TV for adapting LR',type=float,default=1.0)

# Combined
# ---------------------------
mf_update_kwargs = mf_shared_kwargs + mf_trpo_kwargs + mf_ppo_kwargs

# For export to train.py
#########################################
def create_train_parser():
    return parser

all_kwargs = {
    'setup_kwargs':         setup_kwargs,
    'env_kwargs':           env_kwargs,
    'actor_kwargs':         actor_kwargs,
    'critic_kwargs':        critic_kwargs,
    'model_kwargs':         model_kwargs,
    'model_setup_kwargs':   model_setup_kwargs,
    'alg_kwargs':           alg_kwargs,
    'mf_update_kwargs':     mf_update_kwargs,
    #'ami_update_kwargs':    ami_update_kwargs
}