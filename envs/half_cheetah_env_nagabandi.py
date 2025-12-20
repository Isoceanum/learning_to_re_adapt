__credits__ = ["Rushiv Arora"]

import numpy as np
from pathlib import Path

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils.ezpickle import EzPickle
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

XML_PATH = str(Path(__file__).with_name("assets") / "half_cheetah.xml")

class HalfCheetahEnv(MujocoEnv, EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.05,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        frame_skip=1,
        **kwargs
    ):
        EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            frame_skip,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Append torso COM (x,y,z) to observation to match Nagabandi env
        # Always exclude root x from observations: obs = qpos[1:], qvel, torso COM (3)
        # => 17 (qpos excluding root x) + 17 (qvel) + 3 (COM) = 37? but classic HalfCheetah has 17 total without COM
        # Gymnasium HalfCheetah uses 17 dims when excluding root x (qpos[1:] + qvel) -> 17
        # We add 3 COM dims -> 20 total
        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)

        MujocoEnv.__init__(
            self, XML_PATH, frame_skip, observation_space=observation_space, **kwargs
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        dt_action = float(self.dt)
        com_x_before = float(self._get_torso_com()[0])

        self.do_simulation(action, self.frame_skip)

        obs_after = self._get_obs()
        com_x_after = float(obs_after[-3])

        com_x_velocity = (com_x_after - com_x_before) / dt_action
        forward_reward = float(self._forward_reward_weight) * com_x_velocity
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        terminated = False
        truncated = False
        info = {
            "x_position": self.data.qpos[0],   # root body position (for compatibility)
            "x_velocity": self.data.qvel[0],   # root body velocity (approx)
            "com_x_position": com_x_after,     # torso COM position (x)
            "com_x_velocity": com_x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return obs_after, reward, terminated, truncated, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # Always exclude root x to match Nagabandi
        position = position[1:]

        # Append torso COM (x, y, z) as last 3 entries to match original code
        com = self._get_torso_com().copy()

        observation = np.concatenate((position, velocity, com)).ravel()
        return observation

    def _get_torso_com(self):
        """Return torso COM (x,y,z); fail fast if unavailable."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        return self.data.subtree_com[idx]

    def reset_model(self):
        # Match Nagabandi initialization: small Gaussian noise on qpos (std=0.01) and
        # larger Gaussian noise on qvel (std=0.1) scaled by reset_noise_scale.
        qpos = self.init_qpos + self.np_random.normal(
            scale=0.1 * self._reset_noise_scale, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.normal(
            scale=1.0 * self._reset_noise_scale, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    # Expose a model-based reward function for planning (torch tensors expected)
    def get_model_reward_fn(self):
        dt = float(self.dt)
        forward_reward_weight = float(getattr(self, "_forward_reward_weight", 1.0))
        ctrl_cost_weight = float(getattr(self, "_ctrl_cost_weight", 0.05))

        def reward_fn(state, action, next_state):
            import torch

            if isinstance(state, np.ndarray):
                state = torch.as_tensor(state, dtype=torch.float32)
            if isinstance(action, np.ndarray):
                action = torch.as_tensor(action, dtype=torch.float32)
            if isinstance(next_state, np.ndarray):
                next_state = torch.as_tensor(next_state, dtype=torch.float32)

            state = torch.as_tensor(state, dtype=torch.float32)
            action = torch.as_tensor(action, dtype=torch.float32)
            next_state = torch.as_tensor(next_state, dtype=torch.float32)

            com_x_before = state[..., -3]
            com_x_after = next_state[..., -3]
            com_x_velocity = (com_x_after - com_x_before) / dt
            forward_reward = forward_reward_weight * com_x_velocity
            ctrl_cost = ctrl_cost_weight * torch.sum(action ** 2, dim=-1)
            return forward_reward - ctrl_cost

        return reward_fn
