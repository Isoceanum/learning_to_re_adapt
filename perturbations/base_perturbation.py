"""Base perturbation class."""

import random

class Perturbation:
    """Base class for all perturbations."""
    
    def __init__(self, name=None, probability=1.0, **kwargs):
        self.name = name
        self.probability = probability
        self.active = False

    def reset(self, env):
        """Called at the beginning of each episode."""
        self.active = random.random() < self.probability

    def apply_action(self, action):
        """Modify the action before it reaches the env."""
        return action

    def apply_observation(self, obs):
        """Modify the observation before returning to the agent."""
        return obs
    
    def __str__(self):
        raise NotImplementedError("__str__ must be implemented in subclass")