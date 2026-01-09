import random
import gymnasium as gym
import numpy as np

""" 
perturbation:
    type: action_inversion
    probability: 1
    indices: [0,1]
 """
 
class ActionInversionPerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        # Config values
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.indices = self.perturbation_config["indices"]
        
        self.active = False
    
    def reset(self, **kwargs):
        self._roll_activation()
        return self.env.reset(**kwargs)
        
    def step(self, action):
        if not self.active:
            return self.env.step(action)

        inverted_action = np.array(action, copy=True)
        for idx in self.indices:
            inverted_action[idx] *= -1

        return self.env.step(inverted_action)
            
    def _roll_activation(self):
        self.active = self._rng.random() < self.probability

    def is_active(self):
        return self.active
        
