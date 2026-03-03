import torch
import numpy as np


"""
planner:
    type: "cem" # Planner algorithm; rs, cem
    horizon: 15 # length of each candidate action sequence evaluated by the planner.
    n_candidates: 250 # Number of candidate action sequences sampled per planning step
    discount: 0.99 # Reward discount factor inside planning rollouts
    num_cem_iters: 4      # CEM iterations per MPC step. Total model rollouts per MPC step ≈ n_candidates * num_cem_iters.
    percent_elites: 0.15    # Fraction of top-return sequences used to refit mean/std each CEM iter (e.g., 0.1 => 10% elites).
    alpha: 0.20            # Smoothing for mean/std updates (with your code: 0.1 = move 10% toward elites, keep 90% previous).
"""
class CrossEntropyMethodPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha, seed):
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)

        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.dtype = self.act_low.dtype

        self.discount = float(discount)
        self.num_cem_iters = int(num_cem_iters)
        self.percent_elites = float(percent_elites)
        self.alpha = float(alpha)

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def plan(self, state):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=self.device, dtype=self.dtype)

        n = self.n_candidates
        h = self.horizon
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = self.dtype

        num_elites = max(int(n * self.percent_elites), 1)

        horizon_act_dim = h * act_dim
        clip_low = self.act_low.repeat(h).view(1, horizon_act_dim)
        clip_high = self.act_high.repeat(h).view(1, horizon_act_dim)

        mean = 0.5 * (clip_low + clip_high)
        mean = mean.repeat(1, 1).to(dtype=dtype, device=device)

        std = (0.5 * (clip_high - clip_low)).clamp_min(1e-3)

        for _ in range(self.num_cem_iters):
            noise = torch.randn((n, horizon_act_dim), generator=self.gen, device=device, dtype=dtype)
            samples = mean + noise * std
            samples = torch.clamp(samples, clip_low, clip_high)

            action_sequences = samples.view(n, h, act_dim)

            states = state.unsqueeze(0).repeat(n, 1)
            returns = torch.zeros(n, device=device, dtype=dtype)

            for t in range(h):
                a_t = action_sequences[:, t, :]
                next_states = self.dynamics_fn(states, a_t)
                rewards = self.reward_fn(states, a_t, next_states).squeeze(-1)
                returns += (self.discount ** t) * rewards
                states = next_states

            elite_idx = returns.topk(k=num_elites, largest=True, sorted=False).indices
            elites = samples[elite_idx]

            elite_mean = elites.mean(dim=0, keepdim=True)
            elite_std = elites.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

            mean = (1.0 - self.alpha) * mean + self.alpha * elite_mean
            std = (1.0 - self.alpha) * std + self.alpha * elite_std

        first_action = mean.view(h, act_dim)[0]
        return first_action.detach()


class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0):
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.device = torch.device(device)
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device)

        self.dtype = self.act_low.dtype

        self.discount = discount

    @torch.no_grad()
    def plan(self, state):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=self.device, dtype=self.dtype)
        action_dimension = self.act_low.shape[0]
        action_min = self.act_low.view(1, 1, action_dimension)
        action_max = self.act_high.view(1, 1, action_dimension)
        candidate_action_sequences = torch.rand(
            self.n_candidates, self.horizon, action_dimension, device=self.device, dtype=self.dtype
        )
        candidate_action_sequences = action_min + candidate_action_sequences * (action_max - action_min)
        states = state.unsqueeze(0).repeat(self.n_candidates, 1)
        candidate_returns = torch.zeros(self.n_candidates, device=self.device, dtype=self.dtype)

        for step in range(self.horizon):
            actions_t = candidate_action_sequences[:, step, :]
            next_states = self.dynamics_fn(states, actions_t)
            rewards_t = self.reward_fn(states, actions_t, next_states).squeeze(-1)
            candidate_returns += (self.discount ** step) * rewards_t
            states = next_states

        best = torch.argmax(candidate_returns)
        first_action = candidate_action_sequences[best, 0, :]
        return first_action.detach()


class MPPIPlanner:
    def __init__(
        self,
        dynamics_fn,
        reward_fn,
        horizon,
        n_candidates,
        act_low,
        act_high,
        device,
        discount=1.0,
        noise_sigma=0.3,
        lambda_=1.0,
        seed=0,
    ):
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)

        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.dtype = self.act_low.dtype

        self.discount = float(discount)
        self.lambda_ = float(lambda_)

        sigma = torch.as_tensor(noise_sigma, dtype=self.dtype, device=self.device)
        self.noise_sigma = sigma

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

        self._u = None

    def _init_u(self):
        mid = 0.5 * (self.act_low + self.act_high)
        self._u = mid.unsqueeze(0).repeat(self.horizon, 1)

    @torch.no_grad()
    def plan(self, state):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=self.device, dtype=self.dtype)

        h = self.horizon
        n = self.n_candidates
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = self.dtype

        if self._u is None or self._u.shape != (h, act_dim):
            self._init_u()

        noise = torch.randn((n, h, act_dim), generator=self.gen, device=device, dtype=dtype)
        noise = noise * self.noise_sigma

        u = self._u.unsqueeze(0).repeat(n, 1, 1)
        actions = torch.clamp(u + noise, self.act_low, self.act_high)

        states = state.unsqueeze(0).repeat(n, 1)
        returns = torch.zeros(n, device=device, dtype=dtype)

        for t in range(h):
            a_t = actions[:, t, :]
            next_states = self.dynamics_fn(states, a_t)
            rewards = self.reward_fn(states, a_t, next_states).squeeze(-1)
            returns += (self.discount ** t) * rewards
            states = next_states

        beta = returns.max()
        weights = torch.exp((returns - beta) / self.lambda_)
        weights = weights / (weights.sum() + 1e-8)

        weighted_noise = (weights.view(n, 1, 1) * noise).sum(dim=0)
        self._u = torch.clamp(self._u + weighted_noise, self.act_low, self.act_high)

        first_action = self._u[0]
        return first_action.detach()
