import numpy as np
import torch
import contextlib


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) MPC Planner for MB-MPC.

    At each timestep, iteratively optimizes a Gaussian distribution over action
    sequences by sampling, evaluating with the learned dynamics, and refitting
    to top-performing elites.

    Interface: call plan(state) to get an action for the current step.
    """

    def __init__(
        self,
        dynamics_model,
        action_space,
        horizon: int = 20,
        num_candidates: int = 1000,
        device: str = "cpu",
        dt: float = 0.05,
        ctrl_cost_weight: float = 0.1,
        num_elites: int | None = None,
        max_iters: int = 5,
        alpha: float = 0.1,
        reward_fn=None,
        term_fn=None,
        particles: int = 1,
        aggregate: str = "mean",  # or "risk_averse"
        risk_coef: float = 0.0,
        mixed_precision: str | bool | None = None,
    ):
        """
        Args:
            dynamics_model: trained DynamicsModel (predicts Δs)
            action_space: environment action_space (Box)
            horizon: planning horizon (steps to simulate)
            num_candidates: number of candidate action sequences per CEM iteration
            device: torch device string
            dt: environment step size (HalfCheetah default = 0.05)
            ctrl_cost_weight: coefficient for action penalty
            num_elites: how many top sequences to use for refitting
            max_iters: number of CEM iterations
            alpha: smoothing factor for mean/std updates
        """
        # Accept either a single model with predict_next_state or an ensemble
        # wrapper exposing the same method with optional model_indices
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.dt = float(dt)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        # Track if num_elites was explicitly provided; otherwise use 10% of candidates
        self._num_elites_explicit = (num_elites is not None)
        self.num_elites = int(num_elites) if self._num_elites_explicit else None
        self.max_iters = int(max_iters)
        self.alpha = float(alpha)
        self.particles = max(1, int(particles))
        self.aggregate = str(aggregate)
        self.risk_coef = float(risk_coef)

        self.device = torch.device(device)
        # Optional injected env-specific reward function: expects torch tensors
        # of shapes (N, state_dim), (N, state_dim), (N, action_dim) -> (N,)
        self.reward_fn = reward_fn
        # Optional termination function: expects next_state (N, state_dim) -> (N,) bool
        self.term_fn = term_fn

        # Bounds as torch tensors
        self._low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

        # Configure autocast for planner inference if requested and on CUDA
        self._use_autocast = False
        self._autocast_dtype = None
        if self.device.type == "cuda" and mixed_precision is not None:
            if isinstance(mixed_precision, bool):
                self._use_autocast = bool(mixed_precision)
                self._autocast_dtype = torch.bfloat16 if self._use_autocast else None
            elif isinstance(mixed_precision, str):
                mp = mixed_precision.lower()
                if mp in ("bf16", "bfloat16"):
                    self._use_autocast = True
                    self._autocast_dtype = torch.bfloat16
                elif mp in ("fp16", "float16", "half"):
                    self._use_autocast = True
                    self._autocast_dtype = torch.float16
                elif mp in ("none", "off", "false"):
                    self._use_autocast = False

        # Warm-start cache for CEM: previous step's optimized mean sequence
        self._prev_mean = None

    def _compute_reward(self, state, next_state, action):
        """
        Compute reward for candidate rollouts.
        If a custom `reward_fn` was provided, use it. Otherwise, fall back to a
        simple forward-progress reward with control penalty, assuming x-position
        is at index 0 and per-step healthy reward of 1.0.
        Shapes:
          state/next_state: (N, state_dim), action: (N, action_dim)
        """
        if self.reward_fn is not None:
            return self.reward_fn(state, next_state, action)

        x_before = state[:, 0]
        x_after = next_state[:, 0]
        x_velocity = (x_after - x_before) / self.dt
        forward_reward = x_velocity
        healthy_reward = torch.ones_like(x_velocity)
        ctrl_cost = self.ctrl_cost_weight * torch.sum(action ** 2, dim=-1)
        return forward_reward + healthy_reward - ctrl_cost

    def plan(self, state):
        """
        Plan the next action using CEM.

        Args:
            state: np.ndarray (state_dim,) current state

        Returns:
            action: np.ndarray (action_dim,) first action of the optimized mean sequence
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_dim = int(self.action_space.shape[0])
        has_ensemble = hasattr(self.model, "num_models") and int(getattr(self.model, "num_models")) > 1

        # Initialize Gaussian over sequences with warm-start and scaled std
        # Warm-start: shift previous mean left by 1, fill last with previous last
        if (self._prev_mean is not None) and (tuple(self._prev_mean.shape) == (self.horizon, action_dim)):
            mean = torch.empty(self.horizon, action_dim, device=self.device)
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]
        else:
            mean = torch.zeros(self.horizon, action_dim, device=self.device)

        # Initial std as 0.5 * (action_high - action_low)
        std_init = 0.5 * (self._high - self._low)  # (action_dim,)
        std = std_init.expand(self.horizon, -1).clone()  # (horizon, action_dim)

        for itr in range(self.max_iters):
            # Sample candidates: (num_candidates, horizon, action_dim)
            eps = torch.randn(self.num_candidates, self.horizon, action_dim, device=self.device)
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * eps

            # Clamp to action bounds (in-place to reduce temporaries)
            candidates.clamp_(min=self._low, max=self._high)

            # Use autocast for faster matmuls on CUDA when enabled
            mp_ctx = (
                torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
                if (self._use_autocast and self.device.type == "cuda")
                else contextlib.nullcontext()
            )
            with torch.no_grad(), mp_ctx:
                # Rollout using dynamics model with particles (TS∞)
                N = self.num_candidates
                P = self.particles
                total_rewards = torch.zeros(N * P, device=self.device)
                # Repeat candidates per particle
                candidates_rep = candidates.repeat_interleave(P, dim=0)  # (N*P, H, A)
                current_states = state.repeat(N * P, 1)
                # TS1 per particle: one model index per (candidate, particle)
                if has_ensemble:
                    model_indices = torch.randint(low=0, high=int(self.model.num_models), size=(N * P,), device=self.device)
                else:
                    model_indices = None
                # Alive mask per particle
                alive = torch.ones(N * P, dtype=torch.bool, device=self.device)

                for t in range(self.horizon):
                    actions_t = candidates_rep[:, t, :]  # (N*P, action_dim)
                    # Sample next state if model supports stochastic predictions
                    sample = True
                    if has_ensemble:
                        next_states = self.model.predict_next_state(current_states, actions_t, model_indices=model_indices, sample=sample)
                    else:
                        try:
                            next_states = self.model.predict_next_state(current_states, actions_t, sample=sample)
                        except TypeError:
                            next_states = self.model.predict_next_state(current_states, actions_t)
                    reward = self._compute_reward(current_states, next_states, actions_t)
                    # Mask rewards for terminated candidates
                    total_rewards += reward * alive.float()

                    # Early termination handling
                    if self.term_fn is not None:
                        done_now = self.term_fn(next_states)
                        if not torch.is_tensor(done_now):
                            done_now = torch.as_tensor(done_now, dtype=torch.bool, device=self.device)
                        else:
                            done_now = done_now.to(device=self.device, dtype=torch.bool)
                        # Update alive mask
                        newly_done = alive & done_now
                        alive = alive & (~done_now)
                        # For done particles, keep state frozen
                        if torch.any(newly_done):
                            current_states = torch.where(alive.unsqueeze(-1), next_states, current_states)
                        else:
                            current_states = next_states
                        # If all candidates are done, break
                        if not torch.any(alive):
                            break
                    else:
                        current_states = next_states

            # Aggregate particle returns back to candidate scores
            returns = total_rewards.view(N, P)
            if self.aggregate == "risk_averse" and self.risk_coef > 0.0 and P > 1:
                scores = returns.mean(dim=1) - self.risk_coef * returns.std(dim=1, unbiased=False)
            else:
                scores = returns.mean(dim=1)

            # Select elites (10% of candidates unless explicitly set)
            k_elites = (self.num_elites if self._num_elites_explicit else max(1, int(0.1 * self.num_candidates)))
            k_elites = int(max(1, min(k_elites, self.num_candidates)))
            elite_vals, elite_idx = torch.topk(scores, k=k_elites, largest=True, sorted=False)
            elites = candidates[elite_idx]  # (num_elites, horizon, action_dim)

            # Refit Gaussian to elites
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0, unbiased=False).clamp_min(1e-6)

            # Smoothed update
            mean = self.alpha * new_mean + (1.0 - self.alpha) * mean
            std = self.alpha * new_std + (1.0 - self.alpha) * std

        # Cache optimized mean for warm-start on next step
        self._prev_mean = mean.detach()

        # Take first action from the final mean sequence and clamp
        first_action = mean[0]
        first_action = torch.max(torch.min(first_action, self._high), self._low)
        return first_action.detach().cpu().numpy()

 
