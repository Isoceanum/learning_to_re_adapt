import torch

class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0, seed=0 ):
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
    def plan(self, state):   
        dtype = self.act_low.dtype
        device = self.device
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
    def plan(self, state):
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
                next_observation = self.dynamics_fn(observation, a_t)
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
