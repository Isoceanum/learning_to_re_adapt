import unittest

import numpy as np

from algorithms.grbal_lite.buffer import ReplayBuffer


def _make_transition(i):
    obs = np.array([float(i), 0.0, 0.0], dtype=np.float32)
    act = np.array([float(i), 0.0], dtype=np.float32)
    next_obs = np.array([float(i + 1), 0.0, 0.0], dtype=np.float32)
    return obs, act, next_obs


class GrbalLiteReplayBufferTests(unittest.TestCase):
    def test_retrieve_n_recent_transitions_returns_last_n_in_order(self):
        buf = ReplayBuffer(max_size=10, observation_dim=3, action_dim=2)
        for i in range(6):
            buf.add(*_make_transition(i))

        obs_batch, act_batch, next_batch = buf.retrieve_n_recent_transitions(4)

        self.assertEqual(obs_batch.shape, (4, 3))
        self.assertEqual(act_batch.shape, (4, 2))
        self.assertEqual(next_batch.shape, (4, 3))

        expected_indices = [2, 3, 4, 5]
        np.testing.assert_array_equal(
            obs_batch.cpu().numpy(),
            np.stack([_make_transition(i)[0] for i in expected_indices]),
        )
        np.testing.assert_array_equal(
            act_batch.cpu().numpy(),
            np.stack([_make_transition(i)[1] for i in expected_indices]),
        )
        np.testing.assert_array_equal(
            next_batch.cpu().numpy(),
            np.stack([_make_transition(i)[2] for i in expected_indices]),
        )

    def test_retrieve_n_recent_transitions_strict_raises_if_insufficient(self):
        buf = ReplayBuffer(max_size=10, observation_dim=3, action_dim=2)
        for i in range(3):
            buf.add(*_make_transition(i))

        with self.assertRaises(RuntimeError):
            buf.retrieve_n_recent_transitions(4)

    def test_add_hard_cap_raises_when_full(self):
        buf = ReplayBuffer(max_size=4, observation_dim=3, action_dim=2)
        for i in range(4):
            buf.add(*_make_transition(i))

        with self.assertRaises(RuntimeError):
            buf.add(*_make_transition(4))


if __name__ == "__main__":
    unittest.main()
