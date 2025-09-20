import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer for MB-MPC style training.
    Stores (s, a, s') transitions, supports dataset aggregation,
    sampling, and normalization stats.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)

    def add(self, state, action, next_state):
        """
        Add a single transition (s, a, s').
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, next_states):
        """
        Add a batch of transitions.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)

        batch_size = states.shape[0]
        if batch_size == 0:
            return

        if batch_size >= self.max_size:
            states = states[-self.max_size :]
            actions = actions[-self.max_size :]
            next_states = next_states[-self.max_size :]
            batch_size = self.max_size
            start_idx = 0
        else:
            start_idx = self.ptr

        end_idx = start_idx + batch_size
        if end_idx <= self.max_size:
            self.states[start_idx:end_idx] = states
            self.actions[start_idx:end_idx] = actions
            self.next_states[start_idx:end_idx] = next_states
        else:
            first_chunk = self.max_size - start_idx
            self.states[start_idx:] = states[:first_chunk]
            self.actions[start_idx:] = actions[:first_chunk]
            self.next_states[start_idx:] = next_states[:first_chunk]

            remaining = batch_size - first_chunk
            self.states[:remaining] = states[first_chunk:]
            self.actions[:remaining] = actions[first_chunk:]
            self.next_states[:remaining] = next_states[first_chunk:]

        self.ptr = (start_idx + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        """
        Sample a random minibatch as PyTorch tensors.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch_states = torch.tensor(self.states[idxs], dtype=torch.float32)
        batch_actions = torch.tensor(self.actions[idxs], dtype=torch.float32)
        batch_next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32)
        return batch_states, batch_actions, batch_next_states

    def get_all(self):
        """
        Return all data as numpy arrays (for normalization).
        """
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.next_states[:self.size],
        )

    def train_val_split(self, val_ratio=0.1):
        """
        Split stored data into training and validation sets.
        Returns numpy arrays (s_train, a_train, ns_train), (s_val, a_val, ns_val).
        """
        N = self.size
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        split = int(N * (1 - val_ratio))
        train_idx, val_idx = idxs[:split], idxs[split:]

        train = (
            self.states[train_idx],
            self.actions[train_idx],
            self.next_states[train_idx],
        )
        val = (
            self.states[val_idx],
            self.actions[val_idx],
            self.next_states[val_idx],
        )
        return train, val

    def __len__(self):
        return self.size
