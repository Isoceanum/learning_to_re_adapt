import unittest
from unittest.mock import patch
import numpy as np
import torch

from algorithms.mb_mpc.trainer import MBMPCTrainer


class _DummyObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class _DummyActionSpace:
    def __init__(self, lows, highs):
        self.shape = (len(lows),)
        self.low = np.array(lows, dtype=np.float32)
        self.high = np.array(highs, dtype=np.float32)

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class _DummyEnv:
    def __init__(self):
        self.observation_space = _DummyObservationSpace((2,))
        self.action_space = _DummyActionSpace([-1.0, -1.0], [1.0, 1.0])
        self.reward_call_count = 0
        self.unwrapped = self

    def get_model_reward_fn(self):
        def reward_fn(state, action, next_state):
            self.reward_call_count += 1
            return torch.ones(state.shape[0], dtype=torch.float32)

        return reward_fn

    def reset(self, seed=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class MBMPCTrainerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "env": "DummyEnv-v0",
            "train": {
                "seed": 0,
                "buffer_size": 100,
                "learning_rate": 1e-3,
                "hidden_sizes": [8, 8],
                "horizon": 3,
                "n_candidates": 16,
                "discount": 0.99,
                "iterations": 1,
                "epochs": 1,
                "rollout_steps": 1,
                "init_random_steps": 1,
                "batch_size": 1,
            },
            "eval": {"episodes": 1, "seeds": [0]},
        }

    @patch("algorithms.base_trainer.BaseTrainer._make_train_env")
    def test_components_initialized_from_environment(self, make_env_mock):
        dummy_env = _DummyEnv()
        make_env_mock.return_value = dummy_env

        trainer = MBMPCTrainer(self.config, output_dir="/tmp")

        self.assertIs(trainer.env, dummy_env)
        self.assertEqual(trainer.buffer.max_size, self.config["train"]["buffer_size"])
        self.assertEqual(trainer.dynamics_model.observation_dim, dummy_env.observation_space.shape[0])
        self.assertEqual(trainer.planner.horizon, self.config["train"]["horizon"])

        state = torch.zeros(dummy_env.observation_space.shape, dtype=torch.float32)
        action = trainer.planner.plan(state)

        self.assertTrue(torch.all(action >= trainer.planner.act_low))
        self.assertTrue(torch.all(action <= trainer.planner.act_high))
        self.assertGreater(dummy_env.reward_call_count, 0)


if __name__ == "__main__":
    unittest.main()
