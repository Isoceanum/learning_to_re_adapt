import numpy as np

def sample_meta_batch(buffer, meta_batch_size, support_length, query_length, split):
    # Compute size of window
    window_size = support_length + query_length
    # Use train or eval
    trajectories = buffer.train_trajectories if split == "train" else buffer.eval_trajectories 
    
    # List over all trajectories of lenght equal or greater then requried window
    eligible_trajectories = [] 

    # Loop over all trajectories of split in buffer 
    for trajectory_index, (obs, act, next_obs) in enumerate(trajectories):
        # Find lenght of each trajectory
        trajectory_length = obs.shape[0] 
        if trajectory_length >= window_size:
            # If lenght is equal or greater then window we add it to list of eligible trajectories
            eligible_trajectories.append((trajectory_index, trajectory_length)) 

    if len(eligible_trajectories) == 0:
        raise ValueError(f"No trajectories long enough for window_size={window_size}")
    
    meta_batch_picks = [] # List of sampled windows

    for _ in range(meta_batch_size):
        # Sample 1 trajectory at random
        chosen = buffer.rng.integers(0, len(eligible_trajectories))
        trajectory_index, trajectory_length = eligible_trajectories[chosen]
        # Sample index in trajectory at random
        center_timestep = buffer.rng.integers(support_length, trajectory_length - query_length)
        start_timestep = center_timestep - support_length
        # Add window
        meta_batch_picks.append((trajectory_index, start_timestep))
        
        
    # Build full windows for obs/act/next_obs
    obs_windows = []
    act_windows = []
    next_obs_windows = []

    for trajectory_index, start_timestep in meta_batch_picks:
        obs_traj, act_traj, next_obs_traj = trajectories[trajectory_index]
        end_timestep = start_timestep + window_size

        # Slice the contiguous window [start_timestep : start_timestep + window_size]
        obs_windows.append(obs_traj[start_timestep:end_timestep])
        act_windows.append(act_traj[start_timestep:end_timestep])
        next_obs_windows.append(next_obs_traj[start_timestep:end_timestep])
    
    # Stack into arrays of shape
    obs_windows = np.stack(obs_windows, axis=0)
    act_windows = np.stack(act_windows, axis=0)
    next_obs_windows = np.stack(next_obs_windows, axis=0)

    # Split each window into support and query
    support_obs = obs_windows[:,:support_length]
    query_obs = obs_windows[:,support_length:]

    support_act = act_windows[:,:support_length]
    query_act = act_windows[:,support_length:]

    support_next_obs = next_obs_windows[:,:support_length]
    query_next_obs = next_obs_windows[:,support_length:]
    
    return {
        "support_obs": support_obs,
        "support_act": support_act,
        "support_next_obs": support_next_obs,
        "query_obs": query_obs,
        "query_act": query_act,
        "query_next_obs": query_next_obs,
    }


