import numpy as np
import torch


class TransitionBuffer:
    def __init__(self, valid_split_ratio, seed):
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.train_trajectories = []
        self.eval_trajectories = []
        
    def add_trajectories(self, trajectories):
        num_trajectories = len(trajectories)
        num_eval = int(np.ceil(self.valid_split_ratio * num_trajectories))
        num_eval = max(1, min(num_eval, num_trajectories - 1))
        
        eval_indices = self.rng.choice(num_trajectories, size=num_eval, replace=False)
        eval_index_set = set(eval_indices.tolist())
        
        for i, traj in enumerate(trajectories):
            if i in eval_index_set:
                self.eval_trajectories.append(traj)
            else:
                self.train_trajectories.append(traj)
                