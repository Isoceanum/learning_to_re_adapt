import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import algorithms.base_trainer as base_trainer
from algorithms.grbal_lite.trainer import GrBALLiteTrainer


class _DummyObservationSpace:
    def __init__(self, shape):
        self.shape = shape


class _DummyActionSpace:
    def __init__(self, act_dim):
        self.shape = (act_dim,)
        self.low = -np.ones(self.shape, dtype=np.float32)
        self.high = np.ones(self.shape, dtype=np.float32)

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class _DummyEnv:
    def __init__(self, obs_dim, act_dim):
        self.observation_space = _DummyObservationSpace((obs_dim,))
        self.action_space = _DummyActionSpace(act_dim)
        self.unwrapped = self

    def reset(self, seed=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_obs, 0.0, True, False, {"x_position": 0.0}

    def get_model_reward_fn(self):
        def reward_fn(state, action, next_state):
            batch = state.shape[0] if state.ndim > 1 else 1
            return torch.zeros(batch, dtype=torch.float32)

        return reward_fn


class _DummyPlanner:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.calls = 0

    def plan(self, obs_tensor):
        self.calls += 1
        return torch.zeros(self.action_dim, dtype=torch.float32)


def _clone_state(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}


def _state_dicts_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a.keys())


def _make_buffer_transition(obs_dim, act_dim, scale):
    obs = np.full((obs_dim,), scale, dtype=np.float32)
    act = np.full((act_dim,), scale, dtype=np.float32)
    next_obs = obs + 0.5
    return obs, act, next_obs


class GrbalLiteInnerUpdateTests(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 3
        self.act_dim = 2
        self.env = _DummyEnv(self.obs_dim, self.act_dim)

        def fake_make_train_env(_self):
            return self.env

        self.env_patch = patch.object(base_trainer.BaseTrainer, "_make_train_env", fake_make_train_env)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _make_trainer(self, recent_window_size=5, inner_steps=2, inner_lr=1e-2):
        torch.manual_seed(1234)
        config = {
            "env": "DummyEnv-v0",
            "train": {
                "seed": 0,
                "total_env_steps": 32,
                "hidden_sizes": [8, 8],
                "learning_rate": 1e-3,
                "horizon": 3,
                "n_candidates": 8,
                "discount": 0.99,
                "recent_window_size": recent_window_size,
                "inner_steps": inner_steps,
                "inner_lr": inner_lr,
            },
            "eval": {"episodes": 1, "seeds": [0]},
        }
        trainer = GrBALLiteTrainer(config, output_dir=self.tmpdir.name)
        trainer.planner = _DummyPlanner(self.act_dim)

        stats = {
            "observations_mean": torch.zeros(self.obs_dim),
            "observations_std": torch.ones(self.obs_dim),
            "actions_mean": torch.zeros(self.act_dim),
            "actions_std": torch.ones(self.act_dim),
            "delta_mean": torch.zeros(self.obs_dim),
            "delta_std": torch.ones(self.obs_dim),
        }
        trainer.dynamics_model.set_normalization_stats(stats)
        return trainer

    def test_plan_after_inner_update_skips_when_insufficient_data(self):
        trainer = self._make_trainer()
        planner = trainer.planner

        for i in range(3):
            trainer.buffer.add(*_make_buffer_transition(self.obs_dim, self.act_dim, float(i)))

        pre_state = _clone_state(trainer.dynamics_model.state_dict())

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        action = trainer._plan_after_inner_update(obs)

        self.assertEqual(planner.calls, 1)
        self.assertTrue(_state_dicts_equal(trainer.dynamics_model.state_dict(), pre_state))
        self.assertEqual(action.detach().cpu().numpy().shape, (self.act_dim,))

    def test_plan_after_inner_update_adapts_then_restores(self):
        trainer = self._make_trainer()
        planner = trainer.planner
        recent_window = int(trainer.train_config["recent_window_size"])

        obs_list = []
        act_list = []
        next_list = []
        for i in range(recent_window):
            scale = float(i + 1)
            obs = np.full((self.obs_dim,), scale, dtype=np.float32)
            act = np.full((self.act_dim,), scale, dtype=np.float32)
            offset = 0.25 * (i + 1)
            next_obs = obs + 0.5 + offset
            trainer.buffer.add(obs, act, next_obs)
            obs_list.append(torch.tensor(obs))
            act_list.append(torch.tensor(act))
            next_list.append(torch.tensor(next_obs))

        stats = trainer.buffer.compute_normalization_stats()
        trainer.dynamics_model.set_normalization_stats(stats)

        obs_batch = torch.stack(obs_list)
        act_batch = torch.stack(act_list)
        next_batch = torch.stack(next_list)

        with torch.no_grad():
            preds = trainer.dynamics_model.predict_next_state(obs_batch, act_batch)
            loss_before = torch.nn.functional.mse_loss(preds, next_batch)
        self.assertGreater(loss_before.item(), 0.0)

        pre_state = _clone_state(trainer.dynamics_model.state_dict())

        captured = {}
        original_load_state_dict = trainer.dynamics_model.load_state_dict

        def capture_and_restore(state_dict, strict=True):
            captured["mutated_state"] = _clone_state(trainer.dynamics_model.state_dict())
            return original_load_state_dict(state_dict, strict=strict)

        with patch.object(trainer.dynamics_model, "load_state_dict", capture_and_restore):
            planner.calls = 0
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            action = trainer._plan_after_inner_update(obs)

        self.assertEqual(planner.calls, 1)
        self.assertEqual(action.detach().cpu().numpy().shape, (self.act_dim,))
        self.assertTrue(_state_dicts_equal(trainer.dynamics_model.state_dict(), pre_state))
        # Mutated state is captured before restoration; numerical equality with the snapshot
        # is acceptable because the inner loop may produce extremely small deltas when the
        # buffer transitions are nearly consistent with the current model.
        self.assertIn("mutated_state", captured)

        mutated_state = captured["mutated_state"]
        max_param_diff = 0.0
        for key in pre_state:
            diff = (mutated_state[key] - pre_state[key]).abs().max().item()
            max_param_diff = max(max_param_diff, diff)
        self.assertGreater(
            max_param_diff,
            1e-6,
            f"Expected inner update to change parameters, but max diff={max_param_diff}",
        )

        with torch.no_grad():
            trainer.dynamics_model.load_state_dict(mutated_state, strict=True)
            preds_inner = trainer.dynamics_model.predict_next_state(obs_batch, act_batch)
            loss_inner = torch.nn.functional.mse_loss(preds_inner, next_batch)
        self.assertLess(loss_inner.item(), loss_before.item())
        trainer.dynamics_model.load_state_dict(pre_state, strict=True)

        with torch.no_grad():
            preds_after = trainer.dynamics_model.predict_next_state(obs_batch, act_batch)
            loss_after = torch.nn.functional.mse_loss(preds_after, next_batch)
        self.assertTrue(torch.allclose(loss_before, loss_after))

    def test_plan_after_inner_update_respects_inner_steps(self):
        trainer = self._make_trainer()
        recent_window = int(trainer.train_config["recent_window_size"])

        trainer.planner = _DummyPlanner(self.act_dim)
        for i in range(recent_window):
            trainer.buffer.add(*_make_buffer_transition(self.obs_dim, self.act_dim, float(i + 2)))

        stats = trainer.buffer.compute_normalization_stats()
        trainer.dynamics_model.set_normalization_stats(stats)

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        trainer.train_config["inner_steps"] = 1
        trainer.planner = _DummyPlanner(self.act_dim)
        with patch.object(
            trainer.dynamics_model, "predict_next_state", wraps=trainer.dynamics_model.predict_next_state
        ) as mock_predict:
            trainer._plan_after_inner_update(obs)
            self.assertEqual(mock_predict.call_count, 1)
            self.assertEqual(trainer.planner.calls, 1)

        trainer.train_config["inner_steps"] = 3
        trainer.planner = _DummyPlanner(self.act_dim)
        with patch.object(
            trainer.dynamics_model, "predict_next_state", wraps=trainer.dynamics_model.predict_next_state
        ) as mock_predict:
            trainer._plan_after_inner_update(obs)
            self.assertEqual(mock_predict.call_count, 3)
            self.assertEqual(trainer.planner.calls, 1)


if __name__ == "__main__":
    unittest.main()
