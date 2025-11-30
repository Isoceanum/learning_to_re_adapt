
import numpy as np

def sample_batch(replay, batch_size, past_length, future_length):
    episode_starts = replay.episode_start_indices
    episode_ends = episode_starts[1:] + [replay.current_size]
    warmup_end_index = replay.warmup_end_index
    
    window_len = past_length + future_length
    eligible_ranges = []
    
    for i in range(len(episode_starts)):
        start_index = episode_starts[i]
        end_index = episode_ends[i]
        if (end_index - start_index) >= window_len:
            eligible_ranges.append((start_index, end_index))
        
    if not eligible_ranges:
        raise RuntimeError(f"No episode has length ≥ window_len={window_len} (past={past_length}, future={future_length})")
    
    valid_starts = []
    
    for start_index, end_index in eligible_ranges:
        last_start = end_index - window_len
        lower_bound = max(start_index, warmup_end_index)
        if lower_bound > last_start:
            continue
        for t in range(lower_bound, last_start + 1):
            valid_starts.append(t)
            
    num_starts = len(valid_starts)
    replace = batch_size > num_starts
    
    chosen = np.random.choice(valid_starts, size=batch_size, replace=replace)
    
    past_observations = []
    past_actions = []
    past_next_observations = []

    future_observations = []
    future_actions = []
    future_next_observations = []
    
    for t in chosen:
        past_idx = slice(t, t + past_length)
        future_idx = slice(t + past_length, t + past_length + future_length)
        
        past_observations.append(replay.observations[past_idx])
        past_actions.append(replay.actions[past_idx])
        past_next_observations.append(replay.next_observations[past_idx])

        future_observations.append(replay.observations[future_idx])
        future_actions.append(replay.actions[future_idx])
        future_next_observations.append(replay.next_observations[future_idx])
        
        
    past_observations = np.stack(past_observations, axis=0)
    past_actions = np.stack(past_actions, axis=0)
    past_next_observations = np.stack(past_next_observations, axis=0)

    future_observations = np.stack(future_observations, axis=0)
    future_actions = np.stack(future_actions, axis=0)
    future_next_observations = np.stack(future_next_observations, axis=0)
    
    return {
        "past": {
            "obs": past_observations,
            "act": past_actions,
            "next_obs": past_next_observations,
        },
        "future": {
            "obs": future_observations,
            "act": future_actions,
            "next_obs": future_next_observations,
        },
    }

