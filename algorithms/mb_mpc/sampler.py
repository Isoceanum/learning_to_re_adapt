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