import torch
import numpy as np

class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, discount=1.0, seed=0):
        torch.manual_seed(seed) # seed:  random seed for reproducibility
        self.dynamics_fn = dynamics_fn # dynamics_fn: predicts next state from (state, action)
        self.reward_fn = reward_fn # reward_fn: computes reward for (state, action, next_state)
        self.horizon = horizon # horizon: number of steps to plan ahead
        self.n_candidates = n_candidates # n_candidates: number of random action sequences to test
        self.act_low = torch.tensor(act_low, dtype=torch.float32) # act_low: minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32) # act_high: maximum action values
        self.discount = discount # discount: weighting of future rewards
    
    @torch.no_grad()
    def plan(self, state):        
        N = self.n_candidates
        H = self.horizon
        d = self.act_low.shape[0]

        low = self.act_low.view(1, 1, d)
        high = self.act_high.view(1, 1, d)

        # sample random action sequences [N, H, d]
        A = torch.rand(N, H, d)
        A = low + A * (high - low)
    
        # initialize states and total returns
        S = state.unsqueeze(0).repeat(N, 1)       # [N, obs_dim]
        total_return = torch.zeros(N)
        
        for t in range(H):
            a_t = A[:, t, :]
            S_next = self.dynamics_fn(S, a_t)
            r_t = self.reward_fn(S, a_t, S_next).squeeze() # order is state, action, next state
            total_return += (self.discount ** t) * r_t
            S = S_next
              
        # pick the best sequence      
        best = torch.argmax(total_return)
        first_action = A[best, 0, :]
        # return its first action
        return first_action.detach()
        
        
class CrossEntropyMethodPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, discount=1.0, num_cem_iters=8, percent_elites=0.1, alpha=0.1, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)
        self.act_low_np = np.asarray(act_low, dtype=np.float32).reshape(-1)
        self.act_high_np = np.asarray(act_high, dtype=np.float32).reshape(-1)
        self.discount = discount
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.alpha = alpha

    @torch.no_grad()
    def plan(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device=self.act_low.device, dtype=self.act_low.dtype)

        m = state.shape[0]
        n = self.n_candidates
        h = self.horizon
        act_dim = self.act_low.shape[0]
        device = state.device
        dtype = state.dtype

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.zeros((m, h * act_dim), dtype=np.float32)
        std = np.ones_like(mean, dtype=np.float32)
        clip_low = np.tile(self.act_low_np, h)
        clip_high = np.tile(self.act_high_np, h)

        last_returns = None
        last_cand_a = None

        for _ in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m, h * act_dim)).astype(np.float32)
            a = mean + z * std
            a_stacked = np.clip(a, clip_low, clip_high)
            a = a_stacked.reshape(n * m, h, act_dim).transpose(1, 0, 2)
            returns = np.zeros((n * m,), dtype=np.float32)

            for t in range(h):
                if t == 0:
                    cand_a = a[t].reshape(m, n, act_dim)
                    observation = state.repeat_interleave(n, dim=0)
                a_t = torch.from_numpy(a[t]).to(device=device, dtype=dtype)
                next_observation = self.dynamics_fn(observation, a_t)
                rewards = self.reward_fn(observation, a_t, next_observation).view(-1)
                returns += (self.discount ** t) * rewards.detach().cpu().numpy()
                observation = next_observation

            returns = returns.reshape(m, n)
            last_returns = returns
            last_cand_a = cand_a

            elite_idx = np.argsort(returns, axis=1)[:, -num_elites:]
            elites = np.take_along_axis(
                a_stacked.transpose(1, 0, 2), elite_idx[..., None], axis=1
            )
            elite_mean = elites.mean(axis=1)
            elite_std = elites.std(axis=1) + 1e-6  # avoid zeros for next iter
            mean = self.alpha * mean + (1 - self.alpha) * elite_mean
            std = self.alpha * std + (1 - self.alpha) * elite_std

        best = np.argmax(last_returns, axis=1)
        first_action = last_cand_a[np.arange(m), best]
        first_action = torch.from_numpy(first_action).to(device=device, dtype=dtype)
        return first_action.squeeze(0).detach()
