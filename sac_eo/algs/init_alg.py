from sac_eo.algs.mfrl_onpolicy_alg import MFRLOnPolicyAlg
from sac_eo.algs.mbrl_onpolicy_alg import MBRLOnPolicyAlg
from sac_eo.algs.SAC import SAC
from sac_eo.algs.SAC_expert import SAC_exp
from sac_eo.algs.BC import BC
"""Interface to algorithms."""


def init_alg(idx,env,env_eval,env_expert,actor,critics,q_targets,q_critics,models,alg_kwargs,
    mf_update_kwargs,expert,init_expert_rms_stats):
    """Initializes algorithm."""

    alg_type = alg_kwargs['alg_type']

    if alg_type == 'mfrl':
        alg = MFRLOnPolicyAlg(idx,env,env_eval,actor,critics,alg_kwargs,
            mf_update_kwargs)
    elif alg_type == 'mbrl':
        alg = MBRLOnPolicyAlg(idx,env,env_eval,actor,critics,models,alg_kwargs,
            mf_update_kwargs)
    elif alg_type == 'sac':
        alg = SAC(idx,env,env_eval,actor,critics,q_targets,q_critics,models,alg_kwargs,
            mf_update_kwargs)
    elif alg_type == 'sac_imit':
        alg = SAC_exp(idx,env,env_eval,env_expert,actor,expert,init_expert_rms_stats,
                      critics,q_targets,q_critics,models,alg_kwargs,mf_update_kwargs)
    elif alg_type == 'bc':
        alg = BC(idx,env,env_eval,env_expert,actor,expert,init_expert_rms_stats,
                      critics,q_targets,q_critics,models,alg_kwargs,mf_update_kwargs)
    
    else:
        raise ValueError('invalid alg_type')
        
    return alg