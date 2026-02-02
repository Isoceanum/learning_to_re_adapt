import random
import gymnasium as gym
import numpy as np

""" 
perturbation:
    type: action_inversion
    probability: 1
    candidate_action_indices: [0,1]
 """
 
class ActionInversionPerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.candidate_action_indices = self.perturbation_config["candidate_action_indices"]
        
        self.sampled_index = None
        self.active = False
        
    def reset(self, **kwargs):
        self._sample()
        return self.env.reset(**kwargs)
    
    def _sample(self):
        self.active = self._rng.random() < self.probability
        
        if not self.active:
            self.sampled_index = None
            return 
    
        self.sampled_index = self._rng.choice(self.candidate_action_indices)
    
    def step(self, action):
        if not self.active:
            return self.env.step(action)
        
        inverted_action = np.array(action, copy=True)
        inverted_action[self.sampled_index] *= -1
        return self.env.step(inverted_action)
            
    def is_active(self):
        return self.active
        
