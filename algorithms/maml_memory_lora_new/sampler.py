import numpy as np
import torch


def sample_meta_batch(buffer, split, meta_batch_size, support_window_size, query_window_size, device):
    observations, actions, next_observations = buffer.get_trajectories(split)

    if len(observations) == 0:
        raise RuntimeError(f"No episodes available for split='{split}'")

    window_size = support_window_size + query_window_size

    support_observations = []
    support_actions = []
    support_next_observations = []

    query_observations = []
    query_actions = []
    query_next_observations = []

    tries = 0
    max_tries = meta_batch_size * 100

    while len(support_observations) < meta_batch_size:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"Could not sample {meta_batch_size} episodes with length >= {window_size} after {max_tries} attempts.")

        episode_index = int(buffer.rng.integers(0, len(observations)))
        episode_length = len(observations[episode_index])
        if episode_length < window_size:
            continue

        max_start = episode_length - window_size
        start_index = int(buffer.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end_index = start_index + window_size

        # contiguous split from a random window location
        window_obs = observations[episode_index][start_index:end_index]
        window_act = actions[episode_index][start_index:end_index]
        window_next_obs = next_observations[episode_index][start_index:end_index]

        support_observations.append(window_obs[:support_window_size])
        support_actions.append(window_act[:support_window_size])
        support_next_observations.append(window_next_obs[:support_window_size])

        query_observations.append(window_obs[support_window_size:])
        query_actions.append(window_act[support_window_size:])
        query_next_observations.append(window_next_obs[support_window_size:])

    support_observations = torch.as_tensor(np.asarray(support_observations), dtype=torch.float32, device=device)
    support_actions = torch.as_tensor(np.asarray(support_actions), dtype=torch.float32, device=device)
    support_next_observations = torch.as_tensor(np.asarray(support_next_observations), dtype=torch.float32, device=device)

    query_observations = torch.as_tensor(np.asarray(query_observations), dtype=torch.float32, device=device)
    query_actions = torch.as_tensor(np.asarray(query_actions), dtype=torch.float32, device=device)
    query_next_observations = torch.as_tensor(np.asarray(query_next_observations), dtype=torch.float32, device=device)

    return support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations
