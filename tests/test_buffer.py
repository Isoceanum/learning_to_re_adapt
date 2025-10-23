
import unittest
import numpy as np

from algorithms.mb_mpc.buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_add_increases_size_and_stores_values(self):
        buf = ReplayBuffer(max_size=3, observation_dim=2, action_dim=1)
        obs = np.array([1.0, 2.0], dtype=np.float32)
        act = np.array([0.5], dtype=np.float32)
        next_obs = np.array([1.5, 2.5], dtype=np.float32)

        buf.add(obs, act, next_obs)

        self.assertEqual(buf.current_size, 1)
        np.testing.assert_array_equal(buf.observations[0], obs)
        np.testing.assert_array_equal(buf.actions[0], act)
        np.testing.assert_array_equal(buf.next_observations[0], next_obs)

    def test_add_wraps_when_full(self):
        buf = ReplayBuffer(max_size=2, observation_dim=1, action_dim=1)

        for i in range(3):
            obs = np.array([float(i)], dtype=np.float32)
            act = np.array([float(i + 10)], dtype=np.float32)
            next_obs = np.array([float(i + 0.5)], dtype=np.float32)
            buf.add(obs, act, next_obs)

        self.assertEqual(buf.current_size, 2)
        np.testing.assert_array_equal(buf.observations[0], np.array([2.0], dtype=np.float32))
        np.testing.assert_array_equal(buf.actions[0], np.array([12.0], dtype=np.float32))
        np.testing.assert_array_equal(buf.next_observations[0], np.array([2.5], dtype=np.float32))

    def test_sample_returns_entries_from_buffer(self):
        buf = ReplayBuffer(max_size=5, observation_dim=2, action_dim=1)
        transitions = []
        for i in range(4):
            obs = np.array([i, i + 1], dtype=np.float32)
            act = np.array([i + 10], dtype=np.float32)
            next_obs = obs + 0.5
            buf.add(obs, act, next_obs)
            transitions.append((tuple(obs.tolist()), tuple(act.tolist()), tuple(next_obs.tolist())))

        np.random.seed(0)
        obs_batch, act_batch, next_obs_batch = buf.sample_batch(batch_size=3)

        self.assertEqual(obs_batch.shape, (3, 2))
        self.assertEqual(act_batch.shape, (3, 1))
        self.assertEqual(next_obs_batch.shape, (3, 2))

        valid_transitions = set(transitions)
        for o, a, n in zip(obs_batch, act_batch, next_obs_batch):
            triple = (tuple(o.tolist()), tuple(a.tolist()), tuple(n.tolist()))
            self.assertIn(triple, valid_transitions)



if __name__ == "__main__":
    unittest.main()
