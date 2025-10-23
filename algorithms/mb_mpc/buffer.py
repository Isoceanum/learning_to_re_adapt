import torch

class ReplayBuffer:
    
    def __init__(self, max_size, observation_dim, action_dim):
        self.max_size = max_size # buffer capacity
        self.observation_dim = observation_dim # Dimension of the observation vector.
        self.action_dim = action_dim # Dimension of the action vector
        self.write_index = 0 # index where next sample will be written
        self.current_size = 0 # number of valid entries
        
        # Preallocate storage
        self.observations = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.next_observations = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        
    def add(self, observation, action, next_observation):
        """Add a single transition (s, a, s′) to the buffer."""
        self.observations[self.write_index] = torch.as_tensor(observation, dtype=torch.float32) # save observations
        self.actions[self.write_index] = torch.as_tensor(action, dtype=torch.float32) # save action
        self.next_observations[self.write_index] = torch.as_tensor(next_observation, dtype=torch.float32) # save next_observation
        
        self.write_index = (self.write_index + 1) % self.max_size # update index pointer
        self.current_size = min(self.current_size + 1, self.max_size) # update current size
    
    def sample_batch(self, batch_size):
        """Randomly sample a batch of transitions (s, a, s′)."""
        indices = torch.randint(0, self.current_size, (batch_size,)) # sample batch_size random indices
        observations_batch = self.observations[indices]
        actions_batch = self.actions[indices]
        next_observations_batch = self.next_observations[indices]    
        return observations_batch, actions_batch, next_observations_batch
