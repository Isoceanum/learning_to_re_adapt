import numpy as np
import torch

def sample_meta_batch(buffer, split, meta_batch_size, support_window_size, query_window_size, device):
    # sample batch from buffer
    observations, actions, next_observations = buffer.get_trajectories(split)

    if len(observations) == 0:
        raise RuntimeError(f"No episodes available for split='{split}'")

    window_size = support_window_size + query_window_size

    episode_lengths = np.asarray([len(ep) for ep in observations], dtype=np.int64)
    valid_episode_indices = np.flatnonzero(episode_lengths >= window_size)
    if valid_episode_indices.size == 0:
        raise RuntimeError(f"No episodes long enough for window_size={window_size} in split='{split}'")

    # Sample episodes and start indices in a vectorized way.
    sampled = buffer.rng.integers(0, valid_episode_indices.size, size=meta_batch_size)
    episode_indices = valid_episode_indices[sampled]
    max_start = episode_lengths[episode_indices] - window_size
    start_indices = (buffer.rng.random(meta_batch_size) * (max_start + 1)).astype(np.int64)

    obs_dim = observations[episode_indices[0]].shape[1]
    act_dim = actions[episode_indices[0]].shape[1]

    support_observations = np.empty((meta_batch_size, support_window_size, obs_dim), dtype=np.float32)
    support_actions = np.empty((meta_batch_size, support_window_size, act_dim), dtype=np.float32)
    support_next_observations = np.empty((meta_batch_size, support_window_size, obs_dim), dtype=np.float32)

    query_len = query_window_size
    query_observations = np.empty((meta_batch_size, query_len, obs_dim), dtype=np.float32)
    query_actions = np.empty((meta_batch_size, query_len, act_dim), dtype=np.float32)
    query_next_observations = np.empty((meta_batch_size, query_len, obs_dim), dtype=np.float32)

    for i in range(meta_batch_size):
        episode_index = int(episode_indices[i])
        start_index = int(start_indices[i])
        end_index = start_index + window_size

        window_obs = observations[episode_index][start_index:end_index]
        window_act = actions[episode_index][start_index:end_index]
        window_next_obs = next_observations[episode_index][start_index:end_index]

        support_observations[i] = window_obs[:support_window_size]
        support_actions[i] = window_act[:support_window_size]
        support_next_observations[i] = window_next_obs[:support_window_size]

        query_observations[i] = window_obs[support_window_size:]
        query_actions[i] = window_act[support_window_size:]
        query_next_observations[i] = window_next_obs[support_window_size:]

    support_observations = torch.as_tensor(support_observations, dtype=torch.float32)
    support_actions = torch.as_tensor(support_actions, dtype=torch.float32)
    support_next_observations = torch.as_tensor(support_next_observations, dtype=torch.float32)

    query_observations = torch.as_tensor(query_observations, dtype=torch.float32)
    query_actions = torch.as_tensor(query_actions, dtype=torch.float32)
    query_next_observations = torch.as_tensor(query_next_observations, dtype=torch.float32)
    # move returned tensors to device 
    support_observations = support_observations.to(device)
    support_actions = support_actions.to(device)
    support_next_observations = support_next_observations.to(device)
    query_observations = query_observations.to(device)
    query_actions = query_actions.to(device)
    query_next_observations = query_next_observations.to(device)
    
    # return the batch
    return support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations
    
