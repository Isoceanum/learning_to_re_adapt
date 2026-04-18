import random
import gymnasium as gym
import numpy as np

""" 
perturbation:
    type: cripple
    probability: 1
    candidate_action_indices: [[0,1], [2,3]]
 """
 
class CripplePerturbation(gym.Wrapper):
    def __init__(self, env, perturbation_config, seed):
        super().__init__(env)
        
        self.seed = seed
        self._rng = random.Random(self.seed)
        
        self.perturbation_config = perturbation_config
        self.probability = float(self.perturbation_config["probability"])
        self.candidate_action_indices = self.perturbation_config["candidate_action_indices"]
        
        self.sampled_indices = None
        self.active = False
        # Map action index pairs to logical leg names (matching envs/ant.py)
        self.action_pair_to_leg = {
            (0, 1): "back_right",
            (2, 3): "front_left",
            (4, 5): "front_right",
            (6, 7): "back_left",
        }
        
    def reset(self, **kwargs):
        self._sample()
        obs, info = self.env.reset(**kwargs)
        self._apply_geom_cripple()
        return obs, info
    
    def _sample(self):
        self.active = self._rng.random() < self.probability
        
        if not self.active:
            self.sampled_indices = None
            return 
    
        self.sampled_indices = self._rng.choice(self.candidate_action_indices)

    def _apply_geom_cripple(self):
        """If supported by the env, physically disable the selected leg."""
        base_env = getattr(self.env, "unwrapped", self.env)
        restore = getattr(base_env, "restore_disabled_legs", None)
        disable = getattr(base_env, "disable_leg", None)
        if restore:
            restore()
        if not (self.active and self.sampled_indices and disable):
            return
        leg = self.action_pair_to_leg.get(tuple(self.sampled_indices))
        if leg is None:
            return
        disable(leg)
    
    def step(self, action):
        if not self.active:
            return self.env.step(action)
        
        masked_action = np.array(action, copy=True)
        masked_action[self.sampled_indices] = 0
        return self.env.step(masked_action)
            
    def is_active(self):
        return self.active

    def get_task(self):
        if not (self.active and self.sampled_indices):
            return "nominal"

        leg = self.action_pair_to_leg.get(tuple(self.sampled_indices))
        if leg is None:
            return "cripple"

        return f"cripple_{leg}"
        
