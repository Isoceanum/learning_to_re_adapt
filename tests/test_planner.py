import unittest
import torch

from contribution.mb_mpc.planner import RandomShootingPlanner


# Fake dynamics: next_state = state + action
def fake_dynamics(states, actions):
    return states + actions


# Fake reward: sum of next_state components
def fake_reward(states, actions, next_states):
    return next_states.sum(dim=1)


def make_planner(seed=42):
    return RandomShootingPlanner(
        dynamics_fn=fake_dynamics,
        reward_fn=fake_reward,
        horizon=3,
        n_candidates=50,
        act_low=[-1.0, -1.0],
        act_high=[1.0, 1.0],
        seed=seed,
    )


class RandomShootingPlannerTests(unittest.TestCase):
    def test_plan_returns_action_in_bounds(self):
        planner = make_planner()
        state = torch.tensor([0.0, 0.0], dtype=torch.float32)
        action = planner.plan(state)

        # Action should be a 1D tensor of correct dimension
        self.assertEqual(action.shape, (2,))
        self.assertTrue(torch.all(action >= planner.act_low))
        self.assertTrue(torch.all(action <= planner.act_high))

    def test_same_seed_produces_same_action(self):
        state = torch.tensor([0.0, 0.0], dtype=torch.float32)
        action_1 = make_planner(seed=7).plan(state)
        action_2 = make_planner(seed=7).plan(state)
        torch.testing.assert_close(action_1, action_2)


if __name__ == "__main__":
    unittest.main()
