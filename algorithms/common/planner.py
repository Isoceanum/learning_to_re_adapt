import torch
import numpy as np



def make_planner(planner_config, dynamics_fn, reward_fn, action_space, device, seed):
    if planner_config is None:
        raise AttributeError("Planner config not provided in make_planner")
    
    planner_type = planner_config.get("type")         
    horizon = int(planner_config.get("horizon"))
    n_candidates = int(planner_config.get("n_candidates"))
    discount = float(planner_config.get("discount"))
    act_low = action_space.low
    act_high = action_space.high
    
    if planner_type == "rs":
        return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed)

    elif planner_type == "cem":
        num_cem_iters = int(planner_config.get("num_cem_iters"))
        percent_elites = float(planner_config.get("percent_elites"))
        alpha = float(planner_config.get("alpha"))
        
        if num_cem_iters is None or percent_elites is None or alpha is None:
            raise AttributeError("cem planner missing required params")
            
        return CrossEntropyMethodPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed, num_cem_iters, percent_elites, alpha)

    elif planner_type == "mppi":
        noise_sigma = float(planner_config.get("noise_sigma"))
        lambda_ = float(planner_config.get("lambda_"))
        
        if noise_sigma is None or lambda_ is None:
            raise AttributeError("mppi planner missing required params")
        
        return MPPIPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed, lambda_, noise_sigma)

    raise AttributeError(f"Planner type {planner_type} not supported")


class RandomShootingPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed):
        self.dynamics_fn = torch.compile(dynamics_fn) # function used to predict next state from current state and action
        self.reward_fn = reward_fn # model based reward function that computes reward using single transition (state, action, next_state)
        self.horizon = horizon # number of steps to plan ahead
        self.n_candidates = n_candidates # number of random action sequences to test
        self.device = torch.device(device) # Store compute device for later computation
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device) # minimum action values
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device) # maximum action values
        
        self.dtype = self.act_low.dtype # store and use dtype consistently
        self.discount = discount # discount factor (gamma) for future rewards.
        
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
    
    @torch.inference_mode()
    def plan(self, state):
        # convert numpy state to a torch tensor if not already a tensor
        if isinstance(state, np.ndarray): state = torch.as_tensor(state)  
        # move state to device and dtype for planning
        state = state.to(device=self.device, dtype=self.dtype) 
        action_dimension = self.act_low.shape[0]
        # reshape for batch sampling
        action_min = self.act_low.view(1, 1, action_dimension)
        action_max = self.act_high.view(1, 1, action_dimension)
        # sample random action sequences (n_candidates, horizon, action_dimension)
        candidate_action_sequences = torch.rand(self.n_candidates, self.horizon, action_dimension, generator=self.gen, device=self.device, dtype=self.dtype)
        # scale random actions from [0, 1] to [action_min, action_max]
        candidate_action_sequences = action_min + candidate_action_sequences * (action_max - action_min)
        # set same initial state for all candidate rollouts
        states = state.unsqueeze(0).repeat(self.n_candidates, 1)
        # track each candidate total return
        candidate_returns = torch.zeros(self.n_candidates, device=self.device, dtype=self.dtype)
        
        # predict next state and compute rewards for horizon times
        for step in range(self.horizon):
            actions_t = candidate_action_sequences[:, step, :]  # actions at this timestep for all candidates
            next_states = self.dynamics_fn(states, actions_t)
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
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed, num_cem_iters, percent_elites, alpha):
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
    def plan(self, state):
        actions = self.plan_batch(state)
        return actions[0].detach()

    @torch.inference_mode()
    def plan_batch(self, states, parameters_batch=None):
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        states = states.to(device=self.device, dtype=self.dtype)  # (m, obs_dim)

        n = self.n_candidates
        h = self.horizon
        m = states.shape[0]
        act_dim = self.act_low.shape[0]
        device = self.device
        dtype = self.dtype

        num_elites = max(int(n * self.percent_elites), 1)

        horizon_act_dim = h * act_dim
        clip_low = self.act_low.repeat(h).view(1, horizon_act_dim)
        clip_high = self.act_high.repeat(h).view(1, horizon_act_dim)

        mean = torch.zeros((m, horizon_act_dim), device=device, dtype=dtype)
        std = torch.ones_like(mean)

        last_returns = None
        last_action_sequences = None

        clip_low = self.act_low.repeat(h).view(1, 1, horizon_act_dim)
        clip_high = self.act_high.repeat(h).view(1, 1, horizon_act_dim)

        if parameters_batch is not None:
            sample_param = next(iter(parameters_batch.values()))
            if sample_param.shape[0] != m:
                raise ValueError("parameters_batch must have batch dimension matching states")

            def step_fn(obs, act):
                def _step_fn(params, obs_t, act_t):
                    return self.dynamics_fn(obs_t, act_t, params)
                return torch.func.vmap(_step_fn)(parameters_batch, obs, act)
        else:
            def step_fn(obs, act):
                obs_flat = obs.reshape(m * n, -1)
                act_flat = act.reshape(m * n, act_dim)
                next_flat = self.dynamics_fn(obs_flat, act_flat)
                return next_flat.reshape(m, n, -1)

        for _ in range(self.num_cem_iters):
            noise = torch.randn((m, n, horizon_act_dim), generator=self.gen, device=device, dtype=dtype)
            samples = mean.unsqueeze(1) + noise * std.unsqueeze(1)
            samples = torch.clamp(samples, clip_low, clip_high)

            action_sequences = samples.view(m, n, h, act_dim)
            states_t = states.unsqueeze(1).expand(m, n, -1)
            returns = torch.zeros((m, n), device=device, dtype=dtype)

            for t in range(h):
                actions_t = action_sequences[:, :, t, :]
                next_states = step_fn(states_t, actions_t)
                rewards = self.reward_fn(
                    states_t.reshape(m * n, -1),
                    actions_t.reshape(m * n, act_dim),
                    next_states.reshape(m * n, -1),
                ).squeeze(-1)
                returns += (self.discount ** t) * rewards.view(m, n)
                states_t = next_states

            last_returns = returns
            last_action_sequences = action_sequences

            elite_idx = returns.topk(k=num_elites, dim=1, largest=True, sorted=False).indices
            elite_idx = elite_idx.unsqueeze(-1).expand(m, num_elites, horizon_act_dim)
            elites = torch.gather(samples, 1, elite_idx)

            elite_mean = elites.mean(dim=1)
            elite_std = elites.std(dim=1, unbiased=False).clamp_min(1e-6)

            mean = self.alpha * mean + (1.0 - self.alpha) * elite_mean
            std = elite_std

        best_idx = last_returns.argmax(dim=1)
        batch_idx = torch.arange(m, device=device)
        first_action = last_action_sequences[batch_idx, best_idx, 0, :]
        return first_action.detach()

class MPPIPlanner:
    def __init__(self, dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, seed, lambda_=1.0, noise_sigma=0.5, warm_start=True, u_init=None):
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
        
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

        # Nominal open-loop control sequence u_{0:H-1}
        self.u = torch.zeros(self.horizon, self.act_dim, dtype=self.dtype, device=self.device)

        # Value used to fill the last control after shifting
        if u_init is None:
            self.u_init = torch.zeros(self.act_dim, dtype=self.dtype, device=self.device)
        else:
            self.u_init = torch.as_tensor(u_init, dtype=self.dtype, device=self.device).view(self.act_dim)

        # Small constant for stability
        self.eps = 1e-8

    @torch.inference_mode()
    def plan(self, state):
        # State -> (1, obs_dim) torch
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(device=self.device, dtype=self.dtype).view(1, -1)

        K, H, A = self.n_candidates, self.horizon, self.act_dim

        # Sample control noise: du ~ N(0, sigma^2 I)
        du = torch.randn((K, H, A), generator=self.gen, device=self.device, dtype=self.dtype) * self.noise_sigma

        u_nom = self.u.unsqueeze(0).expand(K, H, A)
        u_samp = torch.clamp(u_nom + du, self.act_low.view(1, 1, -1), self.act_high.view(1, 1, -1))
        du_eff = u_samp - u_nom

        # Rollout cost for each candidate
        obs = state.expand(K, state.shape[1]).contiguous()
        costs = torch.zeros((K,), device=self.device, dtype=self.dtype)

        disc = 1.0
        for t in range(H):
            a_t = u_samp[:, t, :]
            obs_next = self.dynamics_fn(obs, a_t)

            r_t = self.reward_fn(obs, a_t, obs_next)
            if r_t.ndim > 1:
                r_t = r_t.squeeze(-1)

            c_t = -r_t
            costs += disc * c_t
            disc *= self.discount
            obs = obs_next

        lam = max(self.lambda_, self.eps)
        c_min = torch.min(costs)
        w_unnorm = torch.exp(-(costs - c_min) / lam)
        w_sum = torch.sum(w_unnorm) + self.eps
        w = w_unnorm / w_sum  # (K,)

        self.u = self.u + torch.sum(w.view(K, 1, 1) * du_eff, dim=0)

        a0 = torch.clamp(self.u[0], self.act_low, self.act_high)

        if self.warm_start:
            self.u = torch.cat([self.u[1:], self.u_init.view(1, A)], dim=0)
        else:
            self.u.zero_()

        return a0.detach()
