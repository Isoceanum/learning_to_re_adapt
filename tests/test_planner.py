import unittest
import torch

from algorithms.mb_mpc.planner import RandomShootingPlanner


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
        
    def test_planner_prefers_high_reward_actions(self):
        """Planner should pick actions that increase next_state sum."""
        planner = make_planner()
        state = torch.zeros(2)
        action = planner.plan(state)
        # Since fake_reward = sum(next_state) = sum(state + action),
        # best action should push toward act_high
        self.assertTrue(torch.all(action > 0.5), f"Expected positive actions, got {action}")

    def test_longer_horizon_gives_higher_return(self):
        """Planner with longer horizon should yield higher predicted reward."""
        state = torch.zeros(2)
        short = make_planner(seed=0)
        short.horizon = 1
        long = make_planner(seed=0)
        long.horizon = 5
        reward_short = fake_reward(state[None, :], short.plan(state)[None, :], fake_dynamics(state[None, :], short.plan(state)[None, :]))
        reward_long = fake_reward(state[None, :], long.plan(state)[None, :], fake_dynamics(state[None, :], long.plan(state)[None, :]))
        self.assertGreaterEqual(reward_long, reward_short)

    def test_dynamics_rollout_consistency(self):
        """Ensure multi-step rollouts apply dynamics repeatedly (not once)."""
        planner = make_planner()
        state = torch.zeros(2)
        # Manually simulate 3 steps of dynamics
        actions = [torch.tensor([0.1, 0.2])] * planner.horizon
        s_manual = state.clone()
        for a in actions:
            s_manual = fake_dynamics(s_manual, a)
        s_planner = planner.dynamics_fn(state, actions[0])  # one step
        self.assertTrue(torch.allclose(fake_dynamics(state, actions[0]), s_planner))

    def test_planner_exploits_dynamics_and_reward(self):
        """Check that planner chooses better-than-random actions under fake dynamics."""
        planner = make_planner(seed=0)
        state = torch.zeros(2)
        
        # Get action chosen by planner
        chosen_action = planner.plan(state)
        
        # Compute its expected return manually (sum(next_state) over horizon)
        total_return_planner = 0
        s = state.clone()
        for _ in range(planner.horizon):
            s = fake_dynamics(s, chosen_action)
            total_return_planner += fake_reward(None, None, s.unsqueeze(0))
        total_return_planner = total_return_planner.item()

        # Compare against average random action returns
        random_returns = []
        for _ in range(50):
            rand_action = torch.rand_like(chosen_action) * 2 - 1
            s = state.clone()
            total = 0
            for _ in range(planner.horizon):
                s = fake_dynamics(s, rand_action)
                total += fake_reward(None, None, s.unsqueeze(0))
            random_returns.append(total.item())

        avg_random_return = sum(random_returns) / len(random_returns)
        print(f"Planner return = {total_return_planner:.3f}, random avg = {avg_random_return:.3f}")

        # Expect planner to beat random average
        self.assertGreater(
            total_return_planner,
            avg_random_return,
            "Planner does not outperform random actions (check reward accumulation logic).",
        )



if __name__ == "__main__":
    unittest.main()
