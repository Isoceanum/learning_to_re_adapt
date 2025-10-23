import unittest
import torch

from algorithms.mb_mpc.dynamics_model import DynamicsModel


class DynamicsModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_predict_next_state_shape(self):
        model = DynamicsModel(observation_dim=2, action_dim=1, hidden_sizes=[8])
        observation = torch.zeros(5, 2)
        action = torch.zeros(5, 1)

        next_state_pred = model.predict_next_state(observation, action)

        self.assertEqual(next_state_pred.shape, (5, 2))

    def test_update_returns_float_loss(self):
        model = DynamicsModel(observation_dim=1, action_dim=1, hidden_sizes=[16], learning_rate=1e-2)
        observations = torch.zeros(32, 1)
        actions = 0.5 * torch.ones(32, 1)
        target_next = observations + actions

        loss = model.update(observations, actions, target_next)

        self.assertIsInstance(loss, float)

    def test_model_learns_simple_dynamics(self):
        model = DynamicsModel(observation_dim=1, action_dim=1, hidden_sizes=[32], learning_rate=5e-2)
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


if __name__ == "__main__":
    unittest.main()
