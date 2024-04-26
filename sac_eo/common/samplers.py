import numpy as np

def trajectory_sampler(env,actor,horizon,s_init=None,eval=False,
    deterministic=False,corruptor=None):
    """Generates single trajectory.
    
    Args:
        env (object): dynamics model (true or learned)
        actor (object): current actor
        horizon (int): length of rollout
        s_init (np.ndarray): initial state
        eval (bool): if True, return total rewards
        deterministic (bool): if True, use deterministic actor
        corruptor (object): trajectory corruptor
    """
    s_traj = []
    a_traj = []
    r_traj = []
    sp_traj = []
    d_traj = []
    J_tot = 0.0

    if s_init is not None:
        s = env.reset(s_init)
    else:
        s = env.reset()

    for t in range(horizon):
        s_old = s

        a = actor.sample(s_old,deterministic=deterministic).numpy()

        s_true, r, d, _ = env.step(actor.clip(a))

        if corruptor is not None:
            s_store = corruptor.corrupt_samples(s_true)
            if corruptor.s_noise_type == 'all':
                s = s_store
            else:
                s = s_true
        else:
            s_store = s_true
            s = s_true

        if eval:
            J_tot += r

        if t == (horizon-1):
            d = False

        # Store
        s_traj.append(s_old)
        a_traj.append(a)
        r_traj.append(r)
        sp_traj.append(s_store)
        d_traj.append(d)

        if d:
            break

    s_traj = np.array(s_traj,dtype=np.float32)
    a_traj = np.array(a_traj,dtype=np.float32)
    r_traj = np.array(r_traj,dtype=np.float32)
    sp_traj = np.array(sp_traj,dtype=np.float32)
    d_traj = np.array(d_traj)

    if eval:
        return s_traj, a_traj, r_traj, sp_traj, d_traj, J_tot
    else:
        return s_traj, a_traj, r_traj, sp_traj, d_traj
    

def batch_simtrajectory_sampler(env,actor,horizon,s_init,deterministic=False):
    """Generates batch of simulated trajectories.
    
    Args:
        env (object): dynamics model (learned only)
        actor (object): current actor
        horizon (int): length of rollout
        s_init (np.ndarray): initial states
        deterministic (bool): if True, use deterministic actor
    """
    if len(s_init.shape) == 1:
        return trajectory_sampler(env,actor,horizon,s_init,eval=False,
            deterministic=deterministic)
    else:    
        s = env.reset(s_init)
        terminated = np.zeros(len(s),dtype='bool')

        for t in range(horizon):
            s_old = s

            a = actor.sample(s_old,deterministic=deterministic).numpy()

            s, r, d, _ = env.step(actor.clip(a))

            if t == (horizon-1):
                d = terminated
            else:
                terminated = np.logical_or(d,terminated)

            # Store
            s_active = np.expand_dims(s_old,axis=1)
            a_active = np.expand_dims(a,axis=1)
            r_active = np.expand_dims(r,axis=1)
            sp_active = np.expand_dims(s,axis=1)
            d_active = np.expand_dims(d,axis=1)

            if t == 0:
                s_batch = s_active
                a_batch = a_active
                r_batch = r_active
                sp_batch = sp_active
                d_batch = d_active
            else:
                s_batch = np.concatenate((s_batch,s_active),axis=1)
                a_batch = np.concatenate((a_batch,a_active),axis=1)
                r_batch = np.concatenate((r_batch,r_active),axis=1)
                sp_batch = np.concatenate((sp_batch,sp_active),axis=1)
                d_batch = np.concatenate((d_batch,d_active),axis=1)

        return s_batch, a_batch, r_batch, sp_batch, d_batch