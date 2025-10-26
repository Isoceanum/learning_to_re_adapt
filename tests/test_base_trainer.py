import unittest
from unittest.mock import patch
import numpy as np

from algorithms.base_trainer import BaseTrainer


class _DummyEnv:
    def __init__(self):
        self.reset_calls = []

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        info = {"dummy": True}
        return obs, info

    def close(self):
        pass


class _EvalEnv:
    def __init__(self):
        self.reset_seed = None
        self.steps = 0

    def reset(self, seed=None):
        self.reset_seed = seed
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        info = {"dummy": True}
        return obs, info

    def step(self, action):
        self.steps += 1
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        reward = 1.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class _TrainerStub(BaseTrainer):
    def __init__(self, config):
        super().__init__(config, output_dir="/tmp")

    def train(self):
        pass

    def predict(self, obs, deterministic=True):
        return np.zeros(1, dtype=np.float32)

    def load(self, path):
        pass

    def save(self):
        pass


class _EvalTrainer(_TrainerStub):
    def __init__(self, config):
        super().__init__(config)
        self.created_eval_envs = []

    def _make_eval_env(self, seed):
        env = _EvalEnv()
        self.created_eval_envs.append(env)
        return env


class BaseTrainerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "env": "DummyEnv-v0",
            "train": {"seed": 123},
            "eval": {"episodes": 1, "seeds": [1]},
        }

    @patch("gymnasium.make")
    def test_make_train_env_uses_env_id_and_seed(self, make_mock):
        dummy_env = _DummyEnv()
        make_mock.return_value = dummy_env
        trainer = _TrainerStub(self.config)

        env = trainer._make_train_env()

        make_mock.assert_called_once_with("DummyEnv-v0", exclude_current_positions_from_observation=False)
        self.assertEqual(dummy_env.reset_calls, [123])
        self.assertIs(env, dummy_env)

    def test_evaluate_uses_eval_seed(self):
        trainer = _EvalTrainer(self.config)

        trainer.evaluate()

        expected_seed = self.config["eval"]["seeds"][0] * 1000 + 0
        recorded_seeds = [env.reset_seed for env in trainer.created_eval_envs]
        self.assertEqual(recorded_seeds, [expected_seed])


if __name__ == "__main__":
    unittest.main()
