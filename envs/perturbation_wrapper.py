"""Environment wrapper for applying a single perturbation."""

import copy
import gymnasium as gym


class PerturbationWrapper(gym.Wrapper):
    """Wrapper that applies a single perturbation to an environment."""

    def __init__(self, env, perturbation=None, perturbation_config=None):
        super().__init__(env)
        if perturbation is not None and perturbation_config is not None:
            raise ValueError("Provide either a perturbation instance or a configuration, not both.")

        if perturbation_config is not None:
            from perturbations.factory import build_perturbation_from_config

            cfg = copy.deepcopy(perturbation_config)
            perturbation = build_perturbation_from_config(cfg)

        self.perturbation = perturbation
        self.episode_idx = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_idx += 1

        if self.perturbation is not None:
            self.perturbation.reset(self.env)
            obs = self.perturbation.apply_observation(obs)

        return obs, info

    def step(self, action):
        if self.perturbation is not None:
            action = self.perturbation.apply_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.perturbation is not None:
            obs = self.perturbation.apply_observation(obs)

        return obs, reward, terminated, truncated, info
