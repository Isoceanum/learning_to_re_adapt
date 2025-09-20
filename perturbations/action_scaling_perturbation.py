import numpy as np

from perturbations.base_perturbation import Perturbation


class ActionScalingPerturbation(Perturbation):
    """Scales specific action dimensions by a factor within a range."""

    def __init__(self, effected_action_indices, scale_range, probability=1.0, name=None):
        super().__init__(name=name, probability=probability)

        self.effected_action_indices = effected_action_indices
        self.scale_range = scale_range

        self.current_index = None
        self.current_scale = 1.0

    def reset(self, env):
        super().reset(env)
        if not self.active:
            self.current_index = None
            self.current_scale = 1.0
            return

        self.current_index = int(self.rng.choice(self.effected_action_indices))
        low, high = self.scale_range
        self.current_scale = float(self.rng.uniform(low, high))

    def apply_action(self, action):
        if not self.active or self.current_index is None:
            return action

        action_array = np.array(action, copy=True)
        action_array[self.current_index] *= self.current_scale
        return action_array

    def __str__(self):
        if not self.active:
            return f"{self.name}(inactive)"
        return f"{self.name}(index={self.current_index}, scale={self.current_scale:.2f})"
