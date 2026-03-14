import torch
import numpy as np

class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0):
        self.dynamics_fn = torch.compile(dynamics_fn) # function used to predict next state from current state and action
        self.reward_fn = reward_fn # model based reward function that computes reward using single transition (state, action, next_state)
        self.horizon = horizon # number of steps to plan ahead
        self.n_candidates = n_candidates # number of random action sequences to test
        self.device = torch.device(device) # Store compute device for later computation
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device) # minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device) # maximum action values
        
        self.dtype = self.act_low.dtype # store and use dtype consistently
        self.discount = discount # discount factor (gamma) for future rewards.
    
    @torch.inference_mode()
    def plan(self, state, parameters=None):
        # convert numpy state to a torch tensor if not already a tensor
        if isinstance(state, np.ndarray): state = torch.as_tensor(state)  
        # move state to device and dtype for planning
        state = state.to(device=self.device, dtype=self.dtype) 
        action_dimension = self.act_low.shape[0]
        # reshape for batch sampling
        action_min = self.act_low.view(1, 1, action_dimension)
        action_max = self.act_high.view(1, 1, action_dimension)
        # sample random action sequences (n_candidates, horizon, action_dimension)
        candidate_action_sequences = torch.rand(self.n_candidates, self.horizon, action_dimension, device=self.device, dtype=self.dtype)
        # scale random actions from [0, 1] to [action_min, action_max]
        candidate_action_sequences = action_min + candidate_action_sequences * (action_max - action_min)
        # set same initial state for all candidate rollouts
        states = state.unsqueeze(0).repeat(self.n_candidates, 1)
        # track each candidate total return
        candidate_returns = torch.zeros(self.n_candidates, device=self.device, dtype=self.dtype)
        
        # predict next state and compute rewards for horizon times
        for step in range(self.horizon):
            actions_t = candidate_action_sequences[:, step, :]  # actions at this timestep for all candidates
            next_states = self.dynamics_fn(states, actions_t, parameters) # predict next states for all candidates using learned dynamics model and provided parameters
            rewards_t = self.reward_fn(states, actions_t, next_states).squeeze(-1) # rewards for each candidate given reward function
            candidate_returns += (self.discount ** step) * rewards_t # discount future rewards
            states = next_states    
        
        # pick sequence with highest total reward     
        best = torch.argmax(candidate_returns)
        # pick first action of best sequence to return 
        first_action = candidate_action_sequences[best, 0, :]
        # return its first action
        return first_action.detach()
        
class CrossEntropyMethodPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha):
        self.dynamics_fn = torch.compile(dynamics_fn)
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

        # local RNG (do not touch global randomness)
        self.gen = torch.Generator(device=self.device)

    @torch.inference_mode()
    def plan(self, state, parameters=None):
        # Match RS: state is a single observation vector (obs_dim,)
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=self.device, dtype=self.dtype)  # (obs_dim,)

        n = self.n_candidates
        h = self.horizon
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = self.dtype

        num_elites = max(int(n * self.percent_elites), 1)

        # Sample in flat space: (n, h*act_dim)
        horizon_act_dim = h * act_dim
        clip_low = self.act_low.repeat(h).view(1, horizon_act_dim)   # (1, h*act_dim)
        clip_high = self.act_high.repeat(h).view(1, horizon_act_dim) # (1, h*act_dim)

        # Initialize mean at center of bounds (robust even if bounds aren't symmetric)
        mean = 0.5 * (clip_low + clip_high)                          # (1, h*act_dim)
        mean = mean.repeat(1, 1).to(dtype=dtype, device=device)      # (1, h*act_dim)

        # Initialize std from action range (half-range)
        std = (0.5 * (clip_high - clip_low)).clamp_min(1e-3)         # (1, h*act_dim)

        for _ in range(self.num_cem_iters):
            noise = torch.randn((n, horizon_act_dim), generator=self.gen, device=device, dtype=dtype)
            samples = mean + noise * std                              # (n, h*act_dim)
            samples = torch.clamp(samples, clip_low, clip_high)       # (n, h*act_dim)

            # Reshape to sequences: (n, h, act_dim)
            action_sequences = samples.view(n, h, act_dim)

            # Rollout all candidates in parallel (like RS)
            states = state.unsqueeze(0).repeat(n, 1)                  # (n, obs_dim)
            returns = torch.zeros(n, device=device, dtype=dtype)

            for t in range(h):
                a_t = action_sequences[:, t, :]                       # (n, act_dim)
                next_states = self.dynamics_fn(states, a_t, parameters)           # (n, obs_dim)
                rewards = self.reward_fn(states, a_t, next_states).squeeze(-1)  # (n,)
                returns += (self.discount ** t) * rewards
                states = next_states

            # Select elites globally (single-state planning)
            elite_idx = returns.topk(k=num_elites, largest=True, sorted=False).indices  # (k,)
            elites = samples[elite_idx]                                   # (k, h*act_dim)

            elite_mean = elites.mean(dim=0, keepdim=True)                 # (1, h*act_dim)
            elite_std = elites.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

            # Smoothing toward previous distribution
            mean = (1.0 - self.alpha) * mean + self.alpha * elite_mean
            std  = (1.0 - self.alpha) * std  + self.alpha * elite_std

        # Deterministic first action from final mean
        first_action = mean.view(h, act_dim)[0]  # (act_dim,)
        return first_action.detach()

class FaithfulCrossEntropyMethodPlanner:
    """
    Faithful CEM matching Nagabandi et al. implementation:
      - mean initialized to 0, std to 1
      - mean update: mean = alpha*old + (1-alpha)*elite_mean
      - std update: std = std(elites) (no smoothing)
      - return best sampled action (not mean action)
    """
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha, seed=0):
        self.dynamics_fn = torch.compile(dynamics_fn)
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

    @torch.inference_mode()
    def plan(self, state, parameters=None):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state)
        state = state.to(device=self.device, dtype=self.dtype)  # (obs_dim,)

        n = self.n_candidates
        h = self.horizon
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = self.dtype

        num_elites = max(int(n * self.percent_elites), 1)

        horizon_act_dim = h * act_dim
        clip_low = self.act_low.repeat(h).view(1, horizon_act_dim)
        clip_high = self.act_high.repeat(h).view(1, horizon_act_dim)

        mean = torch.zeros((1, horizon_act_dim), device=device, dtype=dtype)
        std = torch.ones_like(mean)

        last_returns = None
        last_action_sequences = None

        for _ in range(self.num_cem_iters):
            noise = torch.randn((n, horizon_act_dim), generator=self.gen, device=device, dtype=dtype)
            samples = mean + noise * std
            samples = torch.clamp(samples, clip_low, clip_high)

            action_sequences = samples.view(n, h, act_dim)

            states = state.unsqueeze(0).repeat(n, 1)
            returns = torch.zeros(n, device=device, dtype=dtype)

            for t in range(h):
                a_t = action_sequences[:, t, :]
                next_states = self.dynamics_fn(states, a_t, parameters)
                rewards = self.reward_fn(states, a_t, next_states).squeeze(-1)
                returns += (self.discount ** t) * rewards
                states = next_states

            last_returns = returns
            last_action_sequences = action_sequences

            elite_idx = returns.topk(k=num_elites, largest=True, sorted=False).indices
            elites = samples[elite_idx]
            elite_mean = elites.mean(dim=0, keepdim=True)
            elite_std = elites.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

            mean = self.alpha * mean + (1.0 - self.alpha) * elite_mean
            std = elite_std

        best = last_returns.argmax()
        first_action = last_action_sequences[best, 0, :]
        return first_action.detach()

class MPPIPlanner:
    """
    MPPI (Williams et al., 2015) core update:
      - Sample K rollouts with control perturbations du
      - Compute rollout costs S_k
      - Weights: w_k ∝ exp(-S_k / lambda)
      - Update: u_i <- u_i + sum_k w_k * du_{i,k}
      - Execute u_0, then warm-start by shifting the sequence

    Notes:
      * reward_fn is treated as r(x,u,x'); cost = -reward.
      * If actions are clamped, we use the *actual* delta after clamping:
            du_eff = u_sampled - u_nominal
        so the update stays consistent with saturation.
    """

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
        lambda_=1.0,
        noise_sigma=0.5,
        warm_start=True,
        u_init=None,
        seed=0,
    ):
        self.dynamics_fn = torch.compile(dynamics_fn)
        self.reward_fn = reward_fn
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)
        self.discount = float(discount)
        self.lambda_ = float(lambda_)
        self.noise_sigma = float(noise_sigma)
        self.warm_start = bool(warm_start)

        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.dtype = self.act_low.dtype
        self.act_dim = int(self.act_low.shape[0])

        # Nominal open-loop control sequence u_{0:H-1}
        self.u = torch.zeros(self.horizon, self.act_dim, dtype=self.dtype, device=self.device)

        # Value used to fill the last control after shifting
        if u_init is None:
            self.u_init = torch.zeros(self.act_dim, dtype=self.dtype, device=self.device)
        else:
            self.u_init = torch.as_tensor(u_init, dtype=self.dtype, device=self.device).view(self.act_dim)

        # Use a private RNG (doesn't globally affect torch randomness)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

        # Small constant for stability
        self.eps = 1e-8

    @torch.inference_mode()
    def plan(self, state, parameters=None):
        # State -> (1, obs_dim) torch
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(device=self.device, dtype=self.dtype).view(1, -1)

        K, H, A = self.n_candidates, self.horizon, self.act_dim

        # Sample control noise: du ~ N(0, sigma^2 I)
        du = torch.randn((K, H, A), generator=self.rng, device=self.device, dtype=self.dtype) * self.noise_sigma

        # Candidate controls: u + du, then clamp to action bounds
        u_nom = self.u.unsqueeze(0).expand(K, H, A)
        u_samp = torch.clamp(u_nom + du, self.act_low.view(1, 1, -1), self.act_high.view(1, 1, -1))

        # IMPORTANT: if clamped, the effective perturbation is (u_samp - u_nom), not raw du
        du_eff = u_samp - u_nom

        # Rollout cost for each candidate
        obs = state.expand(K, state.shape[1]).contiguous()
        costs = torch.zeros((K,), device=self.device, dtype=self.dtype)

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
            disc *= self.discount
            obs = obs_next

        # Weights w_k ∝ exp(-cost / lambda)
        lam = max(self.lambda_, self.eps)
        c_min = torch.min(costs)
        w_unnorm = torch.exp(-(costs - c_min) / lam)
        w_sum = torch.sum(w_unnorm) + self.eps
        w = w_unnorm / w_sum  # (K,)

        # Update sequence: u <- u + sum_k w_k * du_eff_k
        self.u = self.u + torch.sum(w.view(K, 1, 1) * du_eff, dim=0)

        # First action to execute (clamp again to be safe)
        a0 = torch.clamp(self.u[0], self.act_low, self.act_high)

        # Warm-start by shifting u_{1:H-1} forward
        if self.warm_start:
            self.u = torch.cat([self.u[1:], self.u_init.view(1, A)], dim=0)
        else:
            self.u.zero_()

        return a0.detach()
