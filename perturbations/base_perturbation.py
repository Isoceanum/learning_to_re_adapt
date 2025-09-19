"""Base perturbation class."""

import random
import numpy as np

class Perturbation:
    """Base class for all perturbations."""
    
    def __init__(self, probability=1.0, name=None, **kwargs):
        self.probability = probability
        self.name = name or self.__class__.__name__
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