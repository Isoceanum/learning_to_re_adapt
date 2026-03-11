import os
from typing import Any, Union

import numpy as np
import torch
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class FaithfulAntEnv(MujocoEnv, utils.EzPickle):
    """
    Faithful reimplementation of Nagabandi et al. ant_env.py reward + observation semantics,
    while remaining compatible with our perturbation wrappers and action clipping.

    Observation = [qpos, qvel, torso_xmat (9), torso_com (3)]
    Reward = forward_velocity + survive_reward (no health gating, no ctrl/contact cost)
    """

    LEG_GEOM_NAMES = {
        "front_left": ["aux_1_geom", "left_leg_geom", "left_ankle_geom"],
        "front_right": ["aux_2_geom", "right_leg_geom", "right_ankle_geom"],
        "back_left": ["aux_3_geom", "back_leg_geom", "third_ankle_geom"],
        "back_right": ["aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom"],
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        reset_noise_scale: float = 0.1,
        main_body: Union[str, int] = "torso",
        exclude_current_positions_from_observation: bool = False,
        **kwargs: Any,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            reset_noise_scale,
            main_body,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._reset_noise_scale = float(reset_noise_scale)
        self._main_body = main_body
        self._exclude_current_positions_from_observation = bool(
            exclude_current_positions_from_observation
        )

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        if not os.path.isabs(xml_file):
            xml_file = os.path.join(assets_dir, xml_file)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # set after model init
            **kwargs,
        )

        # Observation space matches the original ant_env.py layout:
        # qpos + qvel + torso_xmat(9) + torso_com(3)
        obs_size = self.data.qpos.size + self.data.qvel.size + 9 + 3
        if self._exclude_current_positions_from_observation:
            obs_size -= 2
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Survive reward from the original ant_env.py
        self._survive_reward = 0.05

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        body = self.data.body(self._main_body)
        torso_xmat = body.xmat.reshape(-1)
        torso_com = body.xpos.reshape(-1)

        return np.concatenate((qpos, qvel, torso_xmat, torso_com))

    def reset_model(self) -> np.ndarray:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        # Clip to action bounds for stability and wrapper compatibility.
        if isinstance(self.action_space, Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        com_before = self.data.body(self._main_body).xpos.copy()
        self.do_simulation(action, self.frame_skip)
        com_after = self.data.body(self._main_body).xpos.copy()

        com_vel = (com_after - com_before) / self.dt
        forward_reward = float(com_vel[0])
        reward = forward_reward + self._survive_reward

        observation = self._get_obs()
        info = {
            "x_position": float(com_after[0]),
            "y_position": float(com_after[1]),
            "x_velocity": float(com_vel[0]),
        }

        terminated = False
        truncated = False
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    # --- Model-based reward (planner) -----------------------------------
    def reward(self, obs, action, next_obs):
        """
        Numpy reward for compatibility with the original ant_env.py MPC interface.
        obs/next_obs shape: (N, obs_dim)
        """
        assert obs.ndim == 2 and next_obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]

        # COM x is stored in the last 3 entries of the observation
        vel = (next_obs[:, -3] - obs[:, -3]) / self.dt
        reward = vel + self._survive_reward
        return reward

    def get_model_reward_fn(self):
        """
        Torch reward used by planners. Must match env reward exactly.
        """
        dt = float(self.dt)
        survive = float(self._survive_reward)

        def reward_fn(state, action, next_state):
            assert torch.is_tensor(state) and torch.is_tensor(next_state)
            # COM x is stored in the last 3 entries
            vel = (next_state[..., -3] - state[..., -3]) / dt
            return vel + survive

        return reward_fn

    # --- Crippling helpers (for perturbation wrapper) --------------------
    def _ensure_geom_cache(self):
        if hasattr(self, "_geom_cache"):
            return
        if hasattr(self.model, "geom_names"):
            self._geom_name_to_id = {
                (name.decode() if isinstance(name, bytes) else name): i
                for i, name in enumerate(self.model.geom_names)
            }
        else:
            import mujoco

            self._geom_name_to_id = {}
            for i in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name is not None:
                    self._geom_name_to_id[name] = i
        self._geom_cache = {
            "size": self.model.geom_size.copy(),
            "pos": self.model.geom_pos.copy(),
            "rgba": self.model.geom_rgba.copy(),
        }
        self._disabled_leg = None

    def _geom_index(self, name: str):
        if not hasattr(self, "_geom_name_to_id"):
            self._ensure_geom_cache()
        return self._geom_name_to_id.get(name)

    def restore_disabled_legs(self):
        if not hasattr(self, "_geom_cache"):
            return
        self.model.geom_size[:] = self._geom_cache["size"]
        self.model.geom_pos[:] = self._geom_cache["pos"]
        self.model.geom_rgba[:] = self._geom_cache["rgba"]
        self._disabled_leg = None

    def disable_leg(self, leg_name: str):
        if leg_name not in self.LEG_GEOM_NAMES:
            return
        self._ensure_geom_cache()
        for geom_name in self.LEG_GEOM_NAMES[leg_name]:
            idx = self._geom_index(geom_name)
            if idx is None:
                continue
            self.model.geom_size[idx, :2] = self._geom_cache["size"][idx, :2] / 2.0
            self.model.geom_pos[idx] = self._geom_cache["pos"][idx]
            self.model.geom_rgba[idx, :3] = np.array([1.0, 0.0, 0.0])
        self._disabled_leg = leg_name
