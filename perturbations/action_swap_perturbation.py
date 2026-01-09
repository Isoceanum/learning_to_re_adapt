import random
import gymnasium as gym
import numpy as np

""" 
perturbation:
    type: action_swap
    probability: 1
    swap_pairs: [0, 1]
 """
 
class ActionSwapPerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        # Config values
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.swap_pairs = self.perturbation_config["swap_pairs"]
        
        self.active = False
    
    def reset(self, **kwargs):
        self._roll_activation()
        return self.env.reset(**kwargs)
        
    def step(self, action):
        if not self.active:
            return self.env.step(action) 
        
        swapped_action = np.array(action, copy=True)
        i, j = self.swap_pairs
        swapped_action[i], swapped_action[j] = swapped_action[j], swapped_action[i]
        return self.env.step(swapped_action)
        
    def _roll_activation(self):
        self.active = self._rng.random() < self.probability

    def is_active(self):
        return self.active
        
