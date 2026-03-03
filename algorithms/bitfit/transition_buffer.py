import numpy as np


class TransitionBuffer:
    def __init__(self, valid_split_ratio=0.1, seed=42):
        self.valid_split_ratio = float(valid_split_ratio)
        self.rng = np.random.default_rng(int(seed))

        self.train_observations = []
        self.train_actions = []
        self.train_next_observations = []

        self.eval_observations = []
        self.eval_actions = []
        self.eval_next_observations = []

    def _assert_episode_lengths(self, observations, actions, next_observations):
        if not (len(observations) == len(actions) == len(next_observations)):
            raise ValueError("Length mismatch between observations, actions and next_observations")

    def _choose_split(self):
        if len(self.train_observations) == 0:
            return "train"
        if len(self.eval_observations) == 0:
            return "eval"
        return "eval" if self.rng.random() < self.valid_split_ratio else "train"

    def add_trajectory(self, observations, actions, next_observations):
        self._assert_episode_lengths(observations, actions, next_observations)

        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)

        split = self._choose_split()
        if split == "eval":
            self.eval_observations.append(observations)
            self.eval_actions.append(actions)
            self.eval_next_observations.append(next_observations)
        else:
            self.train_observations.append(observations)
            self.train_actions.append(actions)
            self.train_next_observations.append(next_observations)

    def _concat_or_raise(self, items, name):
        if len(items) == 0:
            raise RuntimeError(f"No episodes available for {name}")
        return np.concatenate(items, axis=0)

    def _concat_or_empty(self, items, fallback_shape):
        if len(items) == 0:
            return np.zeros((0, *fallback_shape), dtype=np.float32)
        return np.concatenate(items, axis=0)

    def get_split_arrays(self):
        train_obs = self._concat_or_raise(self.train_observations, "train observations")
        train_act = self._concat_or_raise(self.train_actions, "train actions")
        train_next_obs = self._concat_or_raise(self.train_next_observations, "train next observations")

        eval_obs = self._concat_or_empty(self.eval_observations, train_obs.shape[1:])
        eval_act = self._concat_or_empty(self.eval_actions, train_act.shape[1:])
        eval_next_obs = self._concat_or_empty(self.eval_next_observations, train_next_obs.shape[1:])

        return train_obs, train_act, train_next_obs, eval_obs, eval_act, eval_next_obs

    def num_train_transitions(self):
        return int(sum(len(ep) for ep in self.train_observations))

    def num_eval_transitions(self):
        return int(sum(len(ep) for ep in self.eval_observations))
