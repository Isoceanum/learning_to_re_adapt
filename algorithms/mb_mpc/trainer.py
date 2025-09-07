import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .dynamics import DynamicsModelProb, EnsembleDynamics
from .buffer import ReplayBuffer
from .planner import CEMPlanner
import os


class DynamicsTrainer:
    def __init__(self, state_dim, action_dim, action_space,
                 hidden_sizes=[256, 256],
                 lr=1e-3, batch_size=256, val_ratio=0.1,
                 horizon=20, num_candidates=1000,
                 device="cpu",
                 ctrl_cost_weight: float = 0.1,
                 reward_fn=None,
                 term_fn=None,
                 ensemble_size: int = 1,
                 log_dir: str | None = None):
        self.device = torch.device(device)
        # Build ensemble (K>=1)
        self.models = [
            DynamicsModelProb(state_dim, action_dim, hidden_sizes).to(self.device)
            for _ in range(int(max(1, ensemble_size)))
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        self.buffer = ReplayBuffer(state_dim, action_dim)

        # Planner (initialized with environment-specific control penalty)
        self.dynamics = EnsembleDynamics(self.models)
        # Planner knobs: allow optional overrides
        planner_kwargs = {}
        # Attach optional keys if present on self (set by caller)
        for k in ("num_elites", "max_iters", "alpha", "particles", "aggregate", "risk_coef", "mixed_precision"):
            if hasattr(self, k):
                planner_kwargs[k] = getattr(self, k)

        self.planner = CEMPlanner(
            self.dynamics, action_space,
            horizon=horizon, num_candidates=num_candidates,
            device=self.device,
            ctrl_cost_weight=ctrl_cost_weight,
            reward_fn=reward_fn,
            term_fn=term_fn,
            **planner_kwargs,
        )

    def collect_rollouts(self, env, num_steps=1000, use_planner=False):
        """
        Collect rollouts into the buffer.
        - If use_planner=False: random actions
        - If use_planner=True: use MPC planner
        """
        state, _ = env.reset()
        # Episode tracking for TensorBoard
        ep_reward = 0.0
        ep_len = 0
        ep_rewards = []
        ep_lengths = []
        log_prefix = "planner" if use_planner else "random"
        # Show a progress bar for planner-based rollouts so it's clear work is ongoing
        step_iter = tqdm(range(num_steps), desc="Collecting planner rollouts", leave=False) if use_planner else range(num_steps)
        for _ in step_iter:
            if use_planner:
                action = self.planner.plan(state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, next_state)
            state = next_state

            # accumulate episode stats
            ep_reward += float(reward)
            ep_len += 1

            if done:
                state, _ = env.reset()
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reward = 0.0
                ep_len = 0

        # Log rollout stats if any full episodes completed
        if ep_rewards:
            mean_rew = float(np.mean(ep_rewards))
            mean_len = float(np.mean(ep_lengths))
            print(f"Rollout {log_prefix}: ep_rew_mean={mean_rew:.2f} ep_len_mean={mean_len:.1f} episodes={len(ep_rewards)}")
            self.writer.add_scalar(f"{log_prefix}/ep_rew_mean", mean_rew, self.global_step)
            self.writer.add_scalar(f"{log_prefix}/ep_len_mean", mean_len, self.global_step)

    def train_dynamics(self, epochs=50):
        """Train dynamics model on buffer until val loss stabilizes."""
        states, actions, next_states = self.buffer.get_all()
        train, val = self.buffer.train_val_split(self.val_ratio)

        # Fit normalization per model (with bootstrap if desired in future)
        for m in self.models:
            m.fit_normalization(states, actions, next_states)

        train_states, train_actions, train_next_states = train
        val_states, val_actions, val_next_states = val

        train_states = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        train_actions = torch.tensor(train_actions, dtype=torch.float32, device=self.device)
        train_next_states = torch.tensor(train_next_states, dtype=torch.float32, device=self.device)

        val_states = torch.tensor(val_states, dtype=torch.float32, device=self.device)
        val_actions = torch.tensor(val_actions, dtype=torch.float32, device=self.device)
        val_next_states = torch.tensor(val_next_states, dtype=torch.float32, device=self.device)

        pbar = tqdm(range(epochs), desc="Train ensemble dynamics (epochs)")
        last_mean_train = 0.0
        last_mean_val = 0.0
        for epoch in pbar:
            # Accumulate average train loss across all models and minibatches
            total_train_loss = 0.0
            total_batches = 0
            # For each model, sample shuffled indices (could be bootstrap in future)
            for m, opt in zip(self.models, self.optimizers):
                m.train()
                n_train = len(train_states)
                # Bootstrap sampling with replacement per model per epoch
                boot_idxs = np.random.randint(0, n_train, size=n_train)
                for start in range(0, n_train, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = boot_idxs[start:end]

                    s = train_states[batch_idx]
                    a = train_actions[batch_idx]
                    ns = train_next_states[batch_idx]

                    loss = m.train_step(opt, s, a, ns)
                    total_train_loss += float(loss)
                    total_batches += 1

            # compute val loss per model
            with torch.no_grad():
                val_losses = [
                    m.loss_fn(val_states, val_actions, val_next_states).item()
                    for m in self.models
                ]

            mean_train = (total_train_loss / total_batches) if total_batches > 0 else 0.0
            mean_val = float(np.mean(val_losses)) if val_losses else 0.0
            last_mean_train, last_mean_val = mean_train, mean_val
            print(f"Epoch {epoch+1}/{epochs} | train_mean={mean_train:.4f} | val_mean={mean_val:.4f}")

            # TensorBoard logging
            self.writer.add_scalar("loss/train_mean", mean_train, self.global_step)
            self.writer.add_scalar("loss/val_mean", mean_val, self.global_step)
            self.global_step += 1
            pbar.set_postfix(train=f"{mean_train:.4f}", val=f"{mean_val:.4f}")

    def run_training_loop(self, env, n_iterations=5,
                          init_random_steps=1000, rollout_steps=1000, epochs=50, save_path="outputs/mb_mpc_dynamics.pt"):
        """
        Full MB-MPC training loop:
          - Iter 0: collect random rollouts
          - Train dynamics model
          - Iter >0: use planner for rollouts
          - Retrain dynamics model each iteration
        """
        for itr in range(n_iterations):
            print(f"\n=== Iteration {itr+1}/{n_iterations} ===")

            if itr == 0:
                print("Collecting initial random rollouts...")
                self.collect_rollouts(env, num_steps=init_random_steps, use_planner=False)
            else:
                print("Collecting planner-based rollouts...")
                self.collect_rollouts(env, num_steps=rollout_steps, use_planner=True)

            print("Training dynamics model...")
            self.train_dynamics(epochs=epochs)
            
        # Close TensorBoard writer
        self.writer.close()
