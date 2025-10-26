import unittest
import torch
from algorithms.mb_mpc.dynamics_model import DynamicsModel


class DynamicsModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _set_dummy_norm_stats(self, model):
        """Attach identity normalization stats so tests don't depend on ReplayBuffer."""
        model.set_normalization_stats({
            "observations_mean": torch.zeros(model.observation_dim),
            "observations_std": torch.ones(model.observation_dim),
            "actions_mean": torch.zeros(model.action_dim),
            "actions_std": torch.ones(model.action_dim),
            "delta_mean": torch.zeros(model.observation_dim),
            "delta_std": torch.ones(model.observation_dim),
        })

    def test_predict_next_state_shape(self):
        model = DynamicsModel(observation_dim=2, action_dim=1, hidden_sizes=[8])
        self._set_dummy_norm_stats(model)

        observation = torch.zeros(5, 2)
        action = torch.zeros(5, 1)
        next_state_pred = model.predict_next_state(observation, action)

        self.assertEqual(next_state_pred.shape, (5, 2))

    def test_update_returns_float_loss(self):
        model = DynamicsModel(observation_dim=1, action_dim=1, hidden_sizes=[16], learning_rate=1e-2)
        self._set_dummy_norm_stats(model)

        observations = torch.zeros(32, 1)
        actions = 0.5 * torch.ones(32, 1)
        target_next = observations + actions

        loss = model.update(observations, actions, target_next)
        self.assertIsInstance(loss, float)

    def test_model_learns_simple_dynamics(self):
        model = DynamicsModel(observation_dim=1, action_dim=1, hidden_sizes=[32], learning_rate=5e-2)
        self._set_dummy_norm_stats(model)

        observations = torch.zeros(64, 1)
        actions = torch.ones(64, 1)
        target_next = observations + 0.5 * actions  # delta = 0.5

        final_loss = None
        for _ in range(400):
            final_loss = model.update(observations, actions, target_next)

        self.assertLess(final_loss, 1e-2)

        test_obs = torch.zeros(4, 1)
        test_act = torch.ones(4, 1)
        preds = model.predict_next_state(test_obs, test_act)
        expected = test_obs + 0.5 * test_act

        self.assertTrue(torch.allclose(preds, expected, atol=1e-1))
        
        
    def test_double_normalization_effect(self):
        """Check if predict_next_state misbehaves when inputs are already normalized."""
        model = DynamicsModel(observation_dim=3, action_dim=1, hidden_sizes=[8])
        self._set_dummy_norm_stats(model)

        # Make simple deterministic input
        obs_raw = torch.tensor([[1.0, 2.0, 3.0]])
        act_raw = torch.tensor([[0.5]])

        # Case 1: Raw input (correct usage)
        out_raw = model.predict_next_state(obs_raw, act_raw)

        # Case 2: Pretend planner already normalized them (possible bug)
        obs_normed = (obs_raw - model.observations_mean) / model.observations_std
        act_normed = (act_raw - model.actions_mean) / model.actions_std
        out_normed = model.predict_next_state(obs_normed, act_normed)

        # Compare outputs — if model is stable, these should match closely
        diff = torch.mean(torch.abs(out_raw - out_normed)).item()
        print(f"Average difference between raw and pre-normalized predictions: {diff:.6f}")

        # Allow only tiny deviation
        self.assertLess(
            diff,
            1e-3,
            msg="predict_next_state changes behavior when inputs are already normalized (possible double normalization).",
        )



if __name__ == "__main__":
    unittest.main()
