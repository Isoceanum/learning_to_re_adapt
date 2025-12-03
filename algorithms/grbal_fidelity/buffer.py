import torch

class ReplayBuffer:
    
    def __init__(self, max_size, observation_dim, action_dim):
        self.max_size = max_size # buffer capacity
        self.observation_dim = observation_dim # Dimension of the observation vector.
        self.action_dim = action_dim # Dimension of the action vector
        self.current_size = 0 # number of valid entries
        self.episode_start_indices = [0] # Keeps track of episode start flags 
        self.warmup_end_index = None
                
        # Preallocate storage
        self.observations = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.next_observations = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        
    def add(self, observation, action, next_observation):
        """Add a single transition (s, a, s′) to the buffer."""
        
        if self.current_size >= self.max_size: 
            raise RuntimeError(f"ReplayBuffer full: {self.current_size}/{self.max_size}")
        
        self.observations[self.current_size] = torch.as_tensor(observation, dtype=torch.float32) # save observations
        self.actions[self.current_size] = torch.as_tensor(action, dtype=torch.float32) # save action
        self.next_observations[self.current_size] = torch.as_tensor(next_observation, dtype=torch.float32) # save next_observation
                
        self.current_size += 1
    
    def compute_normalization_stats(self):
        """Compute mean and std for states, actions, and deltas (s' - s)."""
        
        if self.current_size == 0:
            raise ValueError("Cannot compute normalization stats on an empty buffer.")
        
        
        observations = self.observations[:self.current_size]
        actions = self.actions[:self.current_size]
        next_observations = self.next_observations[:self.current_size]
        
        delta = next_observations - observations
        
        self.observations_mean = observations.mean(0)
        self.observations_std = observations.std(0) + 1e-8

        self.actions_mean = actions.mean(0)
        self.actions_std = actions.std(0) + 1e-8

        self.delta_mean = delta.mean(0)
        self.delta_std = delta.std(0) + 1e-8
        
        return {
            "observations_mean": self.observations_mean,
            "observations_std": self.observations_std,
            "actions_mean": self.actions_mean,
            "actions_std": self.actions_std,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
        }

    def normalize_batch(self, observations, actions, next_observations):
        """Normalize a batch of (s, a, s′) using stored mean/std statistics."""
        
        if not hasattr(self, "observations_mean"):
            raise RuntimeError("Normalization stats not yet computed. Call compute_normalization_stats() first.")
        
        norm_observations = (observations - self.observations_mean) / self.observations_std
        norm_actions = (actions - self.actions_mean) / self.actions_std
        
        delta = next_observations - observations
        norm_deltas = (delta - self.delta_mean) / self.delta_std
        
        return norm_observations, norm_actions, norm_deltas

    def unnormalize_delta(self, normalized_delta):
        """Convert a normalized delta back to real-world scale using stored stats."""
        
        if not hasattr(self, "delta_mean"):
            raise RuntimeError("Normalization stats not yet computed. Call compute_normalization_stats() first.")
        
        delta = normalized_delta * self.delta_std + self.delta_mean
        
        return delta

    def retrieve_batch(self, indices):
        """Return (s, a, s′) for the given indices."""
        return self.observations[indices], self.actions[indices],  self.next_observations[indices]
    
    def set_episode_start(self):
        """Mark the current index as the start of a new episode in list of episode_start_indices."""
        self.episode_start_indices.append(self.current_size)
        
    def episode_size(self):
        """Return the number of transitions in the current episode."""
        if not self.episode_start_indices:
            return 0
        last_start = self.episode_start_indices[-1]
        return self.current_size - last_start
        
    def retrieve_recent_transitions_in_episode(self, n):
        """Return up to the last n transitions from the current episode."""   
        last_start = self.episode_start_indices[-1]
        start = max(last_start, self.current_size - n)
        end = self.current_size
        return self.observations[start:end], self.actions[start:end], self.next_observations[start:end]
        
    def clear(self):
        """Reset the buffer contents (size and episode markers)."""
        self.current_size = 0
        self.episode_start_indices = [0]

    def set_warmup_end_index(self):
        """Record the index where the warmup phase ends."""
        self.warmup_end_index = self.current_size
