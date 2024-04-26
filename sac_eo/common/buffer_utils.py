"""Helper functions used by buffers."""
import numpy as np
import scipy.signal as sp_sig

def aggregate_data(data):
    return np.concatenate(data,0)

def discounted_sum(x,rate):
    return sp_sig.lfilter([1], [1, float(-rate)], x[::-1], axis=0)[::-1]

def gae(critic,gamma,lam,s_traj,r_traj,sp_traj,d_traj):
    """Calculates Generalized Advantage Estimation for a trajectory.

    Args:
        critic (object): current critic
        gamma (float): discount rate
        lam (float): Generalized Advantage Estimation parameter lambda
        s_traj (np.ndarray): states
        r_traj (np.ndarray): rewards
        sp_traj (np.ndarray): next states
        d_traj (np.ndarray): done flags

    Returns:
        adv (np.ndarray): advantages
        rtg (np.ndarray): rewards to go
        rtg_sp (np.ndarray): next state rewards to go
    """

    V_s = critic.value(s_traj).numpy()
    V_sp = critic.value(sp_traj).numpy()

    delta = r_traj + gamma * (1-d_traj) * V_sp - V_s

    adv = discounted_sum(delta,gamma*lam)
    rtg = adv + V_s
    rtg_sp = (rtg - r_traj) / gamma

    adv = adv.astype('float32')
    rtg = rtg.astype('float32')
    rtg_sp = rtg_sp.astype('float32')

    return adv, rtg, rtg_sp

def gae_batch(critic,gamma,lam,s_batch,r_batch,sp_batch,d_batch,idx_batch):
    """Calculates advantage estimates for a batch.
    
    Args:
        critic (object): current critic
        gamma (float): discount rate
        lam (float): Generalized Advantage Estimation parameter lambda
        s_batch (np.ndarray): states
        r_batch (np.ndarray): rewards
        sp_batch (np.ndarray): next states
        d_batch (np.ndarray): done flags
        idx_batch (np.ndarray): trajectory indices

    Returns:
        adv_batch (np.ndarray): advantages
        rtg_batch (np.ndarray): rewards to go
        rtg_sp_batch (np.ndarray): next state rewards to go
    """

    sections = np.flatnonzero(idx_batch[1:] - idx_batch[:-1]) + 1    
    idx_all = np.arange(len(idx_batch))
    batches = np.array_split(idx_all,sections)

    adv_batch = np.empty((0,),dtype=np.float32)
    rtg_batch = np.empty((0,),dtype=np.float32)
    rtg_sp_batch = np.empty((0,),dtype=np.float32)
    for batch_idx in batches:
        s_traj = s_batch[batch_idx]
        r_traj = r_batch[batch_idx]
        sp_traj = sp_batch[batch_idx]
        d_traj = d_batch[batch_idx]

        adv_traj, rtg_traj, rtg_sp_traj = gae(
            critic,gamma,lam,s_traj,r_traj,sp_traj,d_traj)
    
        adv_batch = aggregate_data((adv_batch,adv_traj))
        rtg_batch = aggregate_data((rtg_batch,rtg_traj))
        rtg_sp_batch = aggregate_data((rtg_sp_batch,rtg_sp_traj))
    
    return adv_batch, rtg_batch, rtg_sp_batch