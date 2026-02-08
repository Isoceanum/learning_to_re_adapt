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

        # local RNG (do not touch global randomness)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def plan(self, state):
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
                next_states = self.dynamics_fn(states, a_t)           # (n, obs_dim)
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


class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount=1.0):
        self.dynamics_fn = dynamics_fn # function used to predict next state from current state and action
        self.reward_fn = reward_fn # model based reward function that computes reward using single transition (state, action, next_state)
        self.horizon = horizon # number of steps to plan ahead
        self.n_candidates = n_candidates # number of random action sequences to test
        self.device = torch.device(device) # Store compute device for later computation
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device) # minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device) # maximum action values
        
        self.dtype = self.act_low.dtype
        
        self.discount = discount # discount factor (gamma) for future rewards.
    
    @torch.no_grad()
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
            next_states = self.dynamics_fn(states, actions_t, parameters) # predict next states for all candidates using learned dynamics model 
            rewards_t = self.reward_fn(states, actions_t, next_states).squeeze(-1) # rewards for each candidate given reward function
            candidate_returns += (self.discount ** step) * rewards_t # discount future rewards
            states = next_states    

        # pick sequence with highest total reward     
        best = torch.argmax(candidate_returns)
        # pick first action of best sequence to return 
        first_action = candidate_action_sequences[best, 0, :]
        return first_action.detach()
        
        
