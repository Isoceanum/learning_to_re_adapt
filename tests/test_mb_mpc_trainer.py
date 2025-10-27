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

        # Sanity: components correctly initialized
        self.assertIs(trainer.env, dummy_env)
        self.assertEqual(trainer.buffer.max_size, self.config["train"]["buffer_size"])
        self.assertEqual(trainer.dynamics_model.observation_dim, dummy_env.observation_space.shape[0])
        self.assertEqual(trainer.planner.horizon, self.config["train"]["horizon"])

        # --- Patch dummy normalization stats so predict_next_state works ---
        trainer.dynamics_model.set_normalization_stats({
            "observations_mean": torch.zeros(2),
            "observations_std": torch.ones(2),
            "actions_mean": torch.zeros(2),
            "actions_std": torch.ones(2),
            "delta_mean": torch.zeros(2),
            "delta_std": torch.ones(2),
        })

        # --- Test planner call ---
        state = torch.zeros(dummy_env.observation_space.shape, dtype=torch.float32)
        action = trainer.planner.plan(state)

        # --- Validate outputs ---
        self.assertEqual(action.shape, (2,))
        self.assertTrue(torch.all(action >= trainer.planner.act_low))
        self.assertTrue(torch.all(action <= trainer.planner.act_high))
        self.assertGreater(dummy_env.reward_call_count, 0)

        
        
    @patch("algorithms.base_trainer.BaseTrainer._make_train_env")
    def test_train_executes_and_updates_buffer(self, make_env_mock):
        dummy_env = _DummyEnv()
        make_env_mock.return_value = dummy_env

        trainer = MBMPCTrainer(self.config, output_dir="/tmp")

        # Run a very short training loop
        trainer.train()

        # Buffer should have grown beyond the initial random steps
        self.assertGreater(trainer.buffer.current_size, self.config["train"]["init_random_steps"])

        # Dynamics model should have performed gradient updates
        sample_obs, sample_act, sample_next = trainer.buffer.sample_batch(1)
        pred_next = trainer.dynamics_model.predict_next_state(sample_obs, sample_act)
        self.assertTrue(torch.is_tensor(pred_next))


    @patch("algorithms.base_trainer.BaseTrainer._make_train_env")
    def test_rollout_and_reward_accumulation(self, make_env_mock):
        dummy_env = _DummyEnv()
        make_env_mock.return_value = dummy_env
        trainer = MBMPCTrainer(self.config, output_dir="/tmp")

        # Run one iteration manually
        trainer.buffer.add(np.zeros(2), np.zeros(2), np.zeros(2))
        trainer.dynamics_model.set_normalization_stats({
            "observations_mean": torch.zeros(2),
            "observations_std": torch.ones(2),
            "actions_mean": torch.zeros(2),
            "actions_std": torch.ones(2),
            "delta_mean": torch.zeros(2),
            "delta_std": torch.ones(2),
        })
        start = torch.zeros(2)
        a = trainer.planner.plan(start)
        self.assertTrue(torch.all(a <= 1.0))
        self.assertTrue(dummy_env.reward_call_count > 0)


import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import torch

from algorithms.mb_mpc.trainer import MBMPCTrainer


class _EpisodeLengthOneEnv(_DummyEnv):
    """Env that TERMINATES every single step to stress the collection loop.
    We count resets to verify the trainer keeps resetting and continues collecting
    until the full rollout_steps budget is met (the correct behavior)."""
    def __init__(self):
        super().__init__()
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return super().reset(seed=seed)

    def step(self, action):
        self.step_calls += 1
        # Always terminate immediately to force the trainer to reset-and-continue
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class MBMPCTrainerRolloutBudgetTests(unittest.TestCase):
    @patch("algorithms.base_trainer.BaseTrainer._make_train_env")
    def test_collects_full_rollout_budget_per_iteration(self, make_env_mock):
        """This test encodes the CORRECT contract:
        after warm-up, each iteration must add EXACTLY rollout_steps
        transitions to the buffer, even if episodes end early.

        With the current bug (break on first done), this test should FAIL,
        because only ~1 step is collected instead of rollout_steps.
        """
        env = _EpisodeLengthOneEnv()
        make_env_mock.return_value = env

        cfg = {
            "env": "DummyEnv-v0",
            "train": {
                "seed": 0,
                "buffer_size": 10_000,
                "learning_rate": 1e-3,
                "hidden_sizes": [8, 8],
                "horizon": 3,
                "n_candidates": 16,
                "discount": 1.0,
                "iterations": 1,          # single iteration to isolate behavior
                "epochs": 1,              # skip SGD to keep the test fast
                "rollout_steps": 10,      # budget we EXPECT to be collected
                "init_random_steps": 5,   # small warm-up to make counts easy
                "batch_size": 1,
            },
            "eval": {"episodes": 1, "seeds": [0]},
        }

        trainer = MBMPCTrainer(cfg, output_dir="/tmp")

        # Run training: warm-up + ONE iteration of collection
        trainer.train()

        expected_total = cfg["train"]["init_random_steps"] + cfg["train"]["rollout_steps"]
        actual_total = trainer.buffer.current_size

        # This is the **contract**. It will FAIL with the current buggy loop,
        # because the loop breaks after the first episode, collecting ~1 step
        # instead of the full rollout_steps budget.
        self.assertEqual(
            actual_total,
            expected_total,
            msg=(
                f"Trainer did not collect full rollout budget. "
                f"expected buffer size={expected_total}, got {actual_total}. "
                f"(Likely broke on first done instead of reset-and-continue.)"
            ),
        )

        # Extra sanity signals (not strictly required, but helpful to read logs)
        # - We should see multiple resets when episodes end every step.
        self.assertGreaterEqual(
            env.reset_calls, 2,
            "Expected multiple resets during collection; got only one. "
            "Trainer likely stopped collecting after first episode."
        )
        # - We should see at least rollout_steps steps during the iteration.
        #   (warm-up also steps; we can't cleanly separate without internal counters,
        #   but step_calls should be >= warm-up + rollout_steps)
        self.assertGreaterEqual(
            env.step_calls, cfg["train"]["init_random_steps"] + cfg["train"]["rollout_steps"],
            "Env.step was not called enough times to satisfy the rollout budget."
        )



if __name__ == "__main__":
    unittest.main()
