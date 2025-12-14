import torch
import numpy as np

class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0, seed=0):
        torch.manual_seed(seed) # seed:  random seed for reproducibility
        self.dynamics_fn = dynamics_fn # dynamics_fn: predicts next state from (state, action)
        self.reward_fn = reward_fn # reward_fn: computes reward for (state, action, next_state)
        self.horizon = horizon # horizon: number of steps to plan ahead
        self.n_candidates = n_candidates # n_candidates: number of random action sequences to test
        self.device = torch.device(device)
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device) # act_low: minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device) # act_high: maximum action values
        self.discount = discount # discount: weighting of future rewards
    
    @torch.no_grad()
    def plan(self, state, parameters=None):   
        dtype = self.act_low.dtype
        device = self.device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=device, dtype=dtype)
        N = self.n_candidates
        H = self.horizon
        d = self.act_low.shape[0]

        low = self.act_low.view(1, 1, d).to(device=device, dtype=dtype)
        high = self.act_high.view(1, 1, d).to(device=device, dtype=dtype)

        # sample random action sequences [N, H, d]
        A = torch.rand(N, H, d, device=device, dtype=dtype)
        A = low + A * (high - low)
    
        # initialize states and total returns
        S = state.unsqueeze(0).repeat(N, 1)       # [N, obs_dim]
        total_return = torch.zeros(N, device=device)
        
        for t in range(H):
            a_t = A[:, t, :]
            S_next = self.dynamics_fn(S, a_t, parameters)
            r_t = self.reward_fn(S, a_t, S_next).squeeze() # order is state, action, next state
            total_return += (self.discount ** t) * r_t
            S = S_next
        
        # pick the best sequence      
        best = torch.argmax(total_return)
        first_action = A[best, 0, :]
        # return its first action
        return first_action.detach()
        
class CrossEntropyMethodPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0, num_cem_iters=8, percent_elites=0.1, alpha=0.1, seed=0):
        torch.manual_seed(seed)
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.device = torch.device(device)
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device)
        self.discount = discount
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.alpha = alpha

    @torch.no_grad()
    def plan(self, state, parameters=None):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device=self.device, dtype=self.act_low.dtype)

        m = state.shape[0]
        n = self.n_candidates
        h = self.horizon
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = state.dtype

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        horizon_act_dim = h * act_dim
        clip_low = self.act_low.repeat(h).view(1, 1, horizon_act_dim)
        clip_high = self.act_high.repeat(h).view(1, 1, horizon_act_dim)
        mean = torch.zeros((m, horizon_act_dim), device=device, dtype=dtype)
        std = torch.ones_like(mean)

        last_returns = None
        last_cand_a = None

        for _ in range(self.num_cem_iters):
            noise = torch.randn((n, m, horizon_act_dim), device=device, dtype=dtype)
            samples = mean.unsqueeze(0) + noise * std.unsqueeze(0)
            samples = torch.max(torch.min(samples, clip_high), clip_low)
            samples = samples.contiguous()

            actions_flat = samples.permute(1, 0, 2).contiguous()  # (m, n, h*act_dim)
            action_sequences = samples.view(n, m, h, act_dim).permute(1, 0, 2, 3).contiguous()  # (m, n, h, act_dim)
            rollout_actions = action_sequences.view(m * n, h, act_dim)
            cand_a = action_sequences[:, :, 0, :]  # (m, n, act_dim)

            observation = state.unsqueeze(1).repeat(1, n, 1).reshape(m * n, -1)
            returns = torch.zeros((m, n), device=device, dtype=dtype)

            for t in range(h):
                a_t = rollout_actions[:, t, :]
                next_observation = self.dynamics_fn(observation, a_t, parameters)
                rewards = self.reward_fn(observation, a_t, next_observation).reshape(m, n)
                returns += (self.discount ** t) * rewards
                observation = next_observation

            last_returns = returns
            last_cand_a = cand_a

            _, elite_indices = returns.topk(k=num_elites, dim=1, largest=True, sorted=True)
            gather_idx = elite_indices.unsqueeze(-1).expand(-1, -1, horizon_act_dim)
            elites = torch.gather(actions_flat, 1, gather_idx)
            elite_mean = elites.mean(dim=1)
            elite_std = elites.std(dim=1, unbiased=False) + 1e-6

            mean = self.alpha * mean + (1 - self.alpha) * elite_mean
            std = self.alpha * std + (1 - self.alpha) * elite_std

        best = last_returns.argmax(dim=1)
        batch_indices = torch.arange(m, device=device)
        first_action = last_cand_a[batch_indices, best]
        return first_action.squeeze(0).detach()

class MPPIPlanner:
    """
    Simple MPPI (Williams et al., 2015) core update:

      - Sample K rollouts with control perturbations δu
      - Compute rollout costs S_k
      - Weights: w_k ∝ exp( - S_k / λ )
      - Update:  u_i <- u_i + sum_k w_k * δu_{i,k} / sum_k w_k
      - Execute u_0, then warm-start by shifting the sequence

    Notes:
      * We treat your reward_fn as r(x,u,x') and define cost = -r.
      * If actions are clamped, we use the *actual* delta after clamping:
            δu_actual = u_sampled - u_nominal
        so the update stays consistent with saturation.
    """

    def __init__(
        self,
        dynamics_fn,
        reward_fn,
        horizon: int,
        n_candidates: int,
        act_low,
        act_high,
        device,
        discount: float = 1.0,
        lambda_: float = 1.0,
        sigma: float = 0.5,
        warm_start: bool = True,
        u_init: float | np.ndarray | None = None,
        seed: int = 0,
        dtype=torch.float32,
    ):
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)
        self.discount = float(discount)
        self.lambda_ = float(lambda_)
        self.sigma = float(sigma)
        self.warm_start = bool(warm_start)

        self.act_low = torch.as_tensor(act_low, dtype=dtype, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=dtype, device=self.device)
        self.act_dim = int(self.act_low.shape[0])

        # Nominal open-loop control sequence u_{0:H-1}
        self.u = torch.zeros(self.horizon, self.act_dim, dtype=dtype, device=self.device)

        # Value used to fill the last control after shifting
        if u_init is None:
            self.u_init = torch.zeros(self.act_dim, dtype=dtype, device=self.device)
        else:
            self.u_init = torch.as_tensor(u_init, dtype=dtype, device=self.device).view(self.act_dim)

        # Use a private RNG (doesn't globally affect torch randomness)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

        # Small constant for stability
        self.eps = 1e-8

    @torch.no_grad()
    def plan(self, state, parameters=None):
        # State -> (1, obs_dim) torch
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(device=self.device, dtype=self.act_low.dtype).view(1, -1)

        K, H, A = self.n_candidates, self.horizon, self.act_dim

        # Sample control noise: δu ~ N(0, σ^2 I)
        du = torch.randn((K, H, A), generator=self.rng, device=self.device, dtype=self.act_low.dtype) * self.sigma

        # Candidate controls: u + δu, then clamp to action bounds
        u_nom = self.u.unsqueeze(0).expand(K, H, A)
        u_samp = torch.clamp(u_nom + du, self.act_low.view(1, 1, -1), self.act_high.view(1, 1, -1))

        # IMPORTANT: if clamped, the effective perturbation is (u_samp - u_nom), not raw du
        du_eff = u_samp - u_nom

        # Rollout cost for each candidate
        obs = state.expand(K, state.shape[1]).contiguous()
        costs = torch.zeros((K,), device=self.device, dtype=self.act_low.dtype)

        discount = self.discount
        disc = 1.0

        for t in range(H):
            a_t = u_samp[:, t, :]
            obs_next = self.dynamics_fn(obs, a_t, parameters)

            # reward_fn expected to return shape (K,) or (K,1)
            r_t = self.reward_fn(obs, a_t, obs_next)
            if r_t.ndim > 1:
                r_t = r_t.squeeze(-1)

            # MPPI is cost-minimization; treat cost = -reward
            c_t = -r_t

            costs += disc * c_t
            disc *= discount
            obs = obs_next

        # Weights w_k ∝ exp( -cost / lambda )
        # Stabilize with min-cost shift (log-sum-exp trick)
        lam = max(self.lambda_, self.eps)
        c_min = torch.min(costs)
        w_unnorm = torch.exp(-(costs - c_min) / lam)
        w_sum = torch.sum(w_unnorm) + self.eps
        w = w_unnorm / w_sum  # (K,)

        # Update sequence: u <- u + Σ w_k δu_eff_k
        self.u = self.u + torch.sum(w.view(K, 1, 1) * du_eff, dim=0)

        # First action to execute (clamp again to be safe)
        a0 = torch.clamp(self.u[0], self.act_low, self.act_high)

        # Warm-start by shifting u_{1:H-1} forward
        if self.warm_start:
            self.u = torch.cat([self.u[1:], self.u_init.view(1, A)], dim=0)
        else:
            self.u.zero_()

        return a0
