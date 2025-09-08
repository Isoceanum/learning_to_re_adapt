import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from algorithms.base_trainer import BaseTrainer
from algorithms.mb_mpc.dynamics import DynamicsModel  # deterministic MLP with normalization
from algorithms.mb_mpc.buffer import ReplayBuffer
from .planner import NagabandiCEMPlanner


class MBMPCNagabandiTrainer(BaseTrainer):
    """
    Nagabandi-style MB-MPC in PyTorch, integrated with this repo's training loop:
      - Deterministic MLP dynamics (MSE on Δs) with input/output normalization
      - CEM MPC planner matching classic semantics
      - Uses env-provided model reward with forward_reward_weight scaled to match baseline
    """

    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_env()
        self.model = self._build_model()

    def _make_env(self):
        import envs  # ensure custom envs registered
        import gymnasium as gym

        env_id = self.config.get("env")
        # Scale forward reward by 5x to match classic code that divides Δx by 0.01 while dt=0.05
        return gym.make(env_id,
                        exclude_current_positions_from_observation=False,
                        forward_reward_weight=5.0,
                        ctrl_cost_weight=0.05)

    def _make_eval_env(self):
        import envs
        import gymnasium as gym
        env_id = self.config.get("env")
        return gym.make(env_id,
                        exclude_current_positions_from_observation=False,
                        forward_reward_weight=5.0,
                        ctrl_cost_weight=0.05)

    def _build_model(self):
        train_cfg = self.train_config
        env = self.env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        hidden_sizes = train_cfg.get("hidden_sizes", [512, 512])
        lr = float(train_cfg.get("lr", 1e-3))
        batch_size = int(train_cfg.get("batch_size", 500))
        val_ratio = float(train_cfg.get("val_ratio", 0.2))
        device = train_cfg.get("device", "cpu")
        horizon = int(train_cfg.get("horizon", 10))
        n_candidates = int(train_cfg.get("n_candidates", train_cfg.get("num_candidates", 1024)))
        num_cem_iters = int(train_cfg.get("num_cem_iters", train_cfg.get("max_iters", 8)))
        percent_elites = float(train_cfg.get("percent_elites", 0.1))
        alpha = float(train_cfg.get("alpha", 0.1))
        warm_start = bool(train_cfg.get("warm_start", False))
        clip_rollouts = bool(train_cfg.get("clip_rollouts", False))

        self.device = torch.device(device)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))
        self.global_step = 0

        # Build deterministic dynamics model
        self.dynamics = DynamicsModel(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        # Replay buffer
        self.buffer = ReplayBuffer(state_dim, action_dim)

        # Reward function from env
        reward_fn = None
        get_r = getattr(getattr(env, "unwrapped", env), "get_model_reward_fn", None)
        if callable(get_r):
            reward_fn = get_r()

        # Planner
        self.planner = NagabandiCEMPlanner(
            dynamics_model=self.dynamics,
            action_space=env.action_space,
            horizon=horizon,
            n_candidates=n_candidates,
            num_cem_iters=num_cem_iters,
            percent_elites=percent_elites,
            alpha=alpha,
            device=device,
            reward_fn=reward_fn,
            warm_start=warm_start,
            clip_rollouts=clip_rollouts,
        )

        return self

    def collect_rollouts(self, env, num_steps=1000, use_planner=False):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_len = 0
        ep_rewards = []
        ep_lengths = []
        log_prefix = "planner" if use_planner else "random"
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

            ep_reward += float(reward)
            ep_len += 1
            if done:
                state, _ = env.reset()
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reward = 0.0
                ep_len = 0

        if ep_rewards:
            import numpy as _np
            mean_rew = float(_np.mean(ep_rewards))
            mean_len = float(_np.mean(ep_lengths))
            print(f"Rollout {log_prefix}: ep_rew_mean={mean_rew:.2f} ep_len_mean={mean_len:.1f} episodes={len(ep_rewards)}")
            self.writer.add_scalar(f"{log_prefix}/ep_rew_mean", mean_rew, self.global_step)
            self.writer.add_scalar(f"{log_prefix}/ep_len_mean", mean_len, self.global_step)

    def train_dynamics(self, epochs=50):
        states, actions, next_states = self.buffer.get_all()
        train, val = self.buffer.train_val_split(self.val_ratio)

        # Fit normalization
        self.dynamics.fit_normalization(states, actions, next_states)

        train_states, train_actions, train_next_states = train
        val_states, val_actions, val_next_states = val

        train_states = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        train_actions = torch.tensor(train_actions, dtype=torch.float32, device=self.device)
        train_next_states = torch.tensor(train_next_states, dtype=torch.float32, device=self.device)

        val_states = torch.tensor(val_states, dtype=torch.float32, device=self.device)
        val_actions = torch.tensor(val_actions, dtype=torch.float32, device=self.device)
        val_next_states = torch.tensor(val_next_states, dtype=torch.float32, device=self.device)

        pbar = tqdm(range(epochs), desc="Train dynamics (epochs)")
        for epoch in pbar:
            # Shuffle indices
            n_train = len(train_states)
            idxs = torch.randperm(n_train, device=self.device)
            total_loss = 0.0
            total_batches = 0
            for start in range(0, n_train, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]
                s = train_states[batch_idx]
                a = train_actions[batch_idx]
                ns = train_next_states[batch_idx]

                loss = self.dynamics.loss_fn(s, a, ns)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1

            with torch.no_grad():
                val_loss = float(self.dynamics.loss_fn(val_states, val_actions, val_next_states).item())

            mean_train = (total_loss / total_batches) if total_batches > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs} | train_mean={mean_train:.4f} | val={val_loss:.4f}")
            self.writer.add_scalar("loss/train_mean", mean_train, self.global_step)
            self.writer.add_scalar("loss/val", val_loss, self.global_step)
            self.global_step += 1
            pbar.set_postfix(train=f"{mean_train:.4f}", val=f"{val_loss:.4f}")

    def train(self):
        cfg = self.train_config
        n_iterations = int(cfg.get("total_iterations", cfg.get("iterations", 50)))
        init_random_steps = int(cfg.get("init_random_steps", 5000))
        rollout_steps = int(cfg.get("rollout_steps", 32000))
        epochs = int(cfg.get("epochs", 50))

        print(
            f"🚀 Starting Nagabandi MB-MPC: iterations={n_iterations}, "
            f"init_random_steps={init_random_steps}, rollout_steps={rollout_steps}, epochs={epochs}"
        )

        for itr in range(n_iterations):
            print(f"\n=== Iteration {itr+1}/{n_iterations} ===")
            if itr == 0:
                print("Collecting initial random rollouts...")
                self.collect_rollouts(self.env, num_steps=init_random_steps, use_planner=False)
            else:
                print("Collecting planner-based rollouts...")
                self.collect_rollouts(self.env, num_steps=rollout_steps, use_planner=True)

            print("Training dynamics model...")
            self.train_dynamics(epochs=epochs)

        self.writer.close()

    # Action selection for evaluation
    def _predict(self, obs, deterministic: bool):
        return self.planner.plan(obs)

    def save(self):
        import torch as _torch
        save_path = os.path.join(self.output_dir, "model_nagabandi.pt")
        m = self.dynamics
        ckpt = {
            "state_dict": m.state_dict(),
            "normalization": {
                "state_mean": m.state_mean,
                "state_std": m.state_std,
                "action_mean": m.action_mean,
                "action_std": m.action_std,
                "delta_mean": m.delta_mean,
                "delta_std": m.delta_std,
            },
        }
        _torch.save(ckpt, save_path)

    def load(self, path: str):
        import torch as _torch
        ckpt = _torch.load(path, map_location=self.device)
        self.dynamics.load_state_dict(ckpt["state_dict"])
        norm = ckpt.get("normalization")
        if norm:
            self.dynamics.state_mean = norm.get("state_mean")
            self.dynamics.state_std = norm.get("state_std")
            self.dynamics.action_mean = norm.get("action_mean")
            self.dynamics.action_std = norm.get("action_std")
            self.dynamics.delta_mean = norm.get("delta_mean")
            self.dynamics.delta_std = norm.get("delta_std")
            # Refresh cached tensors
            states = np.zeros((1, len(self.dynamics.state_mean)), dtype=np.float32)
            actions = np.zeros((1, len(self.dynamics.action_mean)), dtype=np.float32)
            next_states = np.zeros_like(states)
            self.dynamics.fit_normalization(states, actions, next_states)
