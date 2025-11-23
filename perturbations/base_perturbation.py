import random
import gymnasium as gym

class BasePerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        # Config values
        self.perturbation_config = perturbation_config

    def reset(self, **kwargs):
        raise NotImplementedError("reset() must be implemented in subclass")
    
    def step(self, action):
        raise NotImplementedError("step() must be implemented in subclass")
     
    def _sample(self):
        raise NotImplementedError("_sample() must be implemented in subclass")
  
    def is_active(self):
        raise NotImplementedError("is_active() must be implemented in subclass")
        
    def __str__(self):
        raise NotImplementedError("__str__() must be implemented in subclass")