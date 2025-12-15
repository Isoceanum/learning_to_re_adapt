import random
import gymnasium as gym
import numpy as np


class ActionScalingPerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        # Config values
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.candidate_action_indices = self.perturbation_config["candidate_action_indices"]
        self.range = self.perturbation_config["range"]
        
        
        # Resolved episode specific values 
        self.sampled_index = None
        self.sampled_scale = None
        self.active = False
    
    def reset(self, **kwargs):
        self._sample()
        return self.env.reset(**kwargs)
        
    def step(self, action):
        if not self.active:
            return self.env.step(action) 
        
        scaled_action = action.copy()
        scaled_action[self.sampled_index] *= self.sampled_scale
        return self.env.step(scaled_action)
        

    def _sample(self):
        self.active = self._rng.random() < self.probability
        
        if not self.active:
            self.sampled_index = None
            self.sampled_scale = None
            return 
    
        self.sampled_index = self._rng.choice(self.candidate_action_indices)
        low, high = self.range
        self.sampled_scale = self._rng.uniform(low, high)

    def is_active(self):
        return self.active
        
    def __str__(self):
        cls = self.__class__.__name__
            
        if not self.active:
            return (f"{cls}: inactive ")
        
        return (f"{cls}: active seed={self.seed} index={self.sampled_index} scale={self.sampled_scale:.3f}")
    
    
    
    
    # Action clipping/asymmetry: tighten or skew bounds on selected joints (e.g., only half-range on one leg).    
    # Friction changes: scale friction for specific geoms/contacts (feet) or globally to simulate slippery/rough terrain.
    # Wind/drag: apply a constant force/torque to a body (e.g., torso) via xfrc_applied at reset.