import numpy as np
import torch


def sample_transitions(buffer, batch_size, split):
    observations, actions, next_observations = buffer.get_trajectories(split)

    if len(observations) == 0:
        raise RuntimeError(f"No episodes available for split='{split}'")

    observations_batch = []
    actions_batch = []
    next_observations_batch = []

    for _ in range(batch_size):
        episode_index = buffer.rng.integers(0, len(observations))
        episode_length = len(observations[episode_index])
        step_index = buffer.rng.integers(0, episode_length)

        observations_batch.append(observations[episode_index][step_index])
        actions_batch.append(actions[episode_index][step_index])
        next_observations_batch.append(next_observations[episode_index][step_index])

    observations_batch = torch.as_tensor(np.asarray(observations_batch), dtype=torch.float32)
    actions_batch = torch.as_tensor(np.asarray(actions_batch), dtype=torch.float32)
    next_observations_batch = torch.as_tensor(np.asarray(next_observations_batch), dtype=torch.float32)

    return observations_batch, actions_batch, next_observations_batch


def sample_k_step_batch(buffer, k_step, batch_size, split):
    observations, actions, next_observations = buffer.get_trajectories(split)

    if len(observations) == 0:
        raise RuntimeError(f"No episodes available for split='{split}'")

    horizon = k_step
    valid_indices = [idx for idx, ep in enumerate(observations) if len(ep) >= horizon]
    if not valid_indices:
        raise RuntimeError(f"No episodes long enough for horizon {horizon} in split='{split}'")

    obs_batch = []
    action_batch = []
    target_batch = []

    for _ in range(batch_size):
        episode_index = valid_indices[buffer.rng.integers(0, len(valid_indices))]
        episode_obs = observations[episode_index]
        episode_act = actions[episode_index]
        episode_next_obs = next_observations[episode_index]

        max_start = len(episode_obs) - horizon
        start_index = buffer.rng.integers(0, max_start + 1) if max_start > 0 else 0

        obs_batch.append(episode_obs[start_index])
        action_batch.append(episode_act[start_index:start_index + horizon])
        target_batch.append(episode_next_obs[start_index:start_index + horizon])

    obs_batch = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32)
    action_batch = torch.as_tensor(np.asarray(action_batch), dtype=torch.float32)
    target_batch = torch.as_tensor(np.asarray(target_batch), dtype=torch.float32)

    return obs_batch, action_batch, target_batch
