import os
import time
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from algorithms.base_trainer import BaseTrainer
from .dynamics import DynamicsModel  # deterministic MLP with normalization  # Added for Nagabandi fidelity
from algorithms.mb_mpc.buffer import ReplayBuffer
from .planner import NagabandiCEMPlanner, RandomShootingPlanner
from utils.seeding import set_seed, seed_env  # Added for Nagabandi fidelity


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
        from stable_baselines3.common.env_util import make_vec_env

        env_id = self.config.get("env")
        n_envs = int(self.train_config.get("n_envs", self.config.get("n_envs", 1)))
        # Added for Nagabandi fidelity: pass seed through VecEnv constructor when possible
        seed_cfg = self.config.get("seed", self.train_config.get("seed", None))  # Added for Nagabandi fidelity
        env_kwargs = dict(
            exclude_current_positions_from_observation=False,
        )
        if n_envs > 1:
            try:
                return make_vec_env(env_id, n_envs=n_envs, env_kwargs=env_kwargs, seed=seed_cfg)
            except TypeError:
                # Older SB3 versions may not accept seed=...
                return make_vec_env(env_id, n_envs=n_envs, env_kwargs=env_kwargs)
        return gym.make(env_id, **env_kwargs)

    def _make_eval_env(self):
        import envs
        import gymnasium as gym
        env_id = self.config.get("env")
        return gym.make(env_id,
                        exclude_current_positions_from_observation=False)

    def _build_model(self):
        train_cfg = self.train_config
        env = self.env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        hidden_sizes = train_cfg.get("hidden_sizes", [512, 512])
        lr = float(train_cfg.get("lr", 1e-3))
        batch_size = int(train_cfg.get("batch_size", 500))
        val_ratio = float(train_cfg.get("val_ratio", 0.2))
        # Resolve device robustly: support 'auto' and fallbacks for unavailable backends
        device_cfg = str(train_cfg.get("device", "auto")).lower()
        resolved_device = None
        try:
            if device_cfg in ("auto", "best"):
                if torch.cuda.is_available():
                    resolved_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved_device = "mps"
                else:
                    resolved_device = "cpu"
            elif device_cfg.startswith("cuda"):
                if torch.cuda.is_available():
                    resolved_device = device_cfg
                else:
                    print("[MBMPC-Nagabandi] Warning: CUDA requested but not available; falling back to CPU/MPS")
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        resolved_device = "mps"
                    else:
                        resolved_device = "cpu"
            elif device_cfg == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved_device = "mps"
                else:
                    print("[MBMPC-Nagabandi] Warning: MPS requested but not available; falling back to CPU")
                    resolved_device = "cpu"
            else:
                resolved_device = "cpu"
        except Exception:
            # Any unexpected issue: be safe and use CPU
            resolved_device = "cpu"
        device = resolved_device
        horizon = int(train_cfg.get("horizon", 10))
        n_candidates = int(train_cfg.get("n_candidates", train_cfg.get("num_candidates", 1024)))
        num_cem_iters = int(train_cfg.get("num_cem_iters", train_cfg.get("max_iters", 8)))
        percent_elites = float(train_cfg.get("percent_elites", 0.1))
        alpha = float(train_cfg.get("alpha", 0.1))
        clip_rollouts = bool(train_cfg.get("clip_rollouts", False))

        # Added for Nagabandi fidelity: planner type and training overrides
        planner_type = str(train_cfg.get("planner_type", "cem")).lower()
        # Optional exact-fidelity overrides
        if bool(train_cfg.get("nagabandi_overrides", False)):
            batch_size = 128
            val_ratio = 0.1
            # Note: epochs override applied in train() where epochs is used
            self._epochs_override = int(train_cfg.get("epochs_override", 100))
        else:
            self._epochs_override = None

        # If RS is selected and user did not specify N/H, apply defaults 2000/20
        if planner_type == "rs":
            if "n_candidates" not in train_cfg and "num_candidates" not in train_cfg:
                n_candidates = 2000
            if "horizon" not in train_cfg:
                horizon = 20

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

        # Reward function from env (supports single env or SB3 VecEnv)
        reward_fn = None
        # Try direct on unwrapped (Gymnasium API)
        get_r = getattr(getattr(env, "unwrapped", env), "get_model_reward_fn", None)
        if callable(get_r):
            reward_fn = get_r()
        # Try SB3 VecEnv API: call method on the first sub-env
        if reward_fn is None and hasattr(env, "env_method"):
            try:
                fns = env.env_method("get_model_reward_fn")  # returns list per sub-env
                if isinstance(fns, (list, tuple)) and len(fns) > 0 and callable(fns[0]):
                    reward_fn = fns[0]
            except Exception:
                pass

        # Planner
        if planner_type == "rs":
            # Added for Nagabandi fidelity: Random Shooting option
            self.planner = RandomShootingPlanner(
                dynamics_model=self.dynamics,
                action_space=env.action_space,
                horizon=horizon,
                n_candidates=n_candidates,
                device=str(self.device),
                reward_fn=reward_fn,
                rng=None,  # will be set in train() after seeding
            )
        else:
            self.planner = NagabandiCEMPlanner(
                dynamics_model=self.dynamics,
                action_space=env.action_space,
                horizon=horizon,
                n_candidates=n_candidates,
                num_cem_iters=num_cem_iters,
                percent_elites=percent_elites,
                alpha=alpha,
                device=str(self.device),
                reward_fn=reward_fn,
                clip_rollouts=clip_rollouts,
                rng=None,  # will be set in train() after seeding
            )

        return self

    def collect_rollouts(self, env, num_steps=1000, use_planner=False):
        log_prefix = "planner" if use_planner else "random"
        is_vec = hasattr(env, "num_envs")

        if not is_vec:
            # Single-env path (Gymnasium API)
            # Added for Nagabandi fidelity: seed env reset
            try:
                state, _ = env.reset(seed=int(self.seed))
            except Exception:
                state, _ = env.reset()
            ep_reward = 0.0
            ep_len = 0
            ep_rewards = []
            ep_lengths = []
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
            return

        # Vectorized env path (SB3 VecEnv API)
        # Reset returns obs only
        # Added for Nagabandi fidelity: seed vec reset
        try:
            states = env.reset(seed=int(self.seed))
        except Exception:
            states = env.reset()
        # Episode stats per env
        n_envs = int(getattr(env, "num_envs", len(states)))
        ep_rewards = np.zeros(n_envs, dtype=np.float64)
        ep_lengths = np.zeros(n_envs, dtype=np.int64)
        completed_rewards = []
        completed_lengths = []
        step_iter = tqdm(range(num_steps), desc="Collecting planner rollouts (vec)", leave=False) if use_planner else range(num_steps)
        for _ in step_iter:
            if use_planner:
                actions = self.planner.plan(states)  # (n_envs, A)
            else:
                # Sample per-env actions
                low, high = env.action_space.low, env.action_space.high
                actions = np.random.uniform(low=low, high=high, size=(n_envs, low.shape[0]))

            # VecEnv step signature: obs, rewards, dones, infos
            out = env.step(actions)
            if len(out) == 4:
                next_states, rewards, dones, infos = out
            elif len(out) == 5:
                # Gymnasium-style (unlikely for VecEnv)
                next_states, rewards, terminated, truncated, infos = out
                dones = np.asarray(terminated) | np.asarray(truncated)
            else:
                raise RuntimeError("Unexpected VecEnv.step return format")

            # Store transitions
            self.buffer.add_batch(states, actions, next_states)
            states = next_states

            # Update episode stats
            ep_rewards += rewards.astype(np.float64)
            ep_lengths += 1
            if np.any(dones):
                done_idx = np.where(dones)[0]
                for di in done_idx.tolist():
                    completed_rewards.append(float(ep_rewards[di]))
                    completed_lengths.append(int(ep_lengths[di]))
                    ep_rewards[di] = 0.0
                    ep_lengths[di] = 0

        if completed_rewards:
            import numpy as _np
            mean_rew = float(_np.mean(completed_rewards))
            mean_len = float(_np.mean(completed_lengths))
            print(f"Rollout {log_prefix} (vec): ep_rew_mean={mean_rew:.2f} ep_len_mean={mean_len:.1f} episodes={len(completed_rewards)}")
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

        # Added for Nagabandi fidelity: early stopping with rolling average + patience
        rolling_persistency = float(self.train_config.get("rolling_average_persitency", 0.99))
        patience = int(self.train_config.get("early_stop_patience", 5))
        best_ema = None
        best_state = None
        no_improve = 0

        pbar = tqdm(range(epochs), desc="Train dynamics (epochs)")
        for epoch in pbar:
            # Shuffle indices
            n_train = len(train_states)
            # Added for Nagabandi fidelity: use seeded generator for reproducibility (CPU, then move)
            idxs = torch.randperm(n_train, generator=self._torch_gen)
            if str(self.device) != "cpu":
                idxs = idxs.to(self.device)
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

            # Added for Nagabandi fidelity: update rolling EMA and early stop
            if best_ema is None:
                ema = val_loss
                best_ema = ema
                best_state = copy.deepcopy(self.dynamics.state_dict())
                no_improve = 0
            else:
                ema = rolling_persistency * best_ema + (1.0 - rolling_persistency) * val_loss
                if ema < best_ema - 1e-8:
                    best_ema = ema
                    best_state = copy.deepcopy(self.dynamics.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (patience={patience}).")
                break

        # Added for Nagabandi fidelity: restore best parameters
        if best_state is not None:
            self.dynamics.load_state_dict(best_state)

    def train(self):
        start_time = time.time()
        cfg = self.train_config
        n_iterations = int(cfg.get("total_iterations", cfg.get("iterations", 50)))
        init_random_steps = int(cfg.get("init_random_steps", 5000))
        rollout_steps = int(cfg.get("rollout_steps", 32000))
        epochs = int(cfg.get("epochs", 50))
        # Added for Nagabandi fidelity: override epochs if requested
        if getattr(self, "_epochs_override", None):
            epochs = int(self._epochs_override)

        # Added for Nagabandi fidelity: global seeding
        self.seed = int(self.config.get("seed", cfg.get("seed", 42)))
        set_seed(self.seed)
        # Seeded Torch generator on the active device
        try:
            self._torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
            # Propagate RNG to planner for CEM/RS sampling
            if hasattr(self, "planner") and hasattr(self.planner, "rng"):
                self.planner.rng = self._torch_gen  # ensure seeded sampling on same device
        except Exception:
            # Fallback to CPU generator; keep planner using global RNG to avoid device mismatch
            self._torch_gen = torch.Generator().manual_seed(self.seed)
            if hasattr(self, "planner") and hasattr(self.planner, "rng"):
                self.planner.rng = None

        print(
            f"🚀 Starting Nagabandi MB-MPC: iterations={n_iterations}, "
            f"init_random_steps={init_random_steps}, rollout_steps={rollout_steps}, epochs={epochs}"
        )

        for itr in range(n_iterations):
            print(f"\n=== Iteration {itr+1}/{n_iterations} ===")
            if itr == 0:
                print("Collecting initial random rollouts...")
                # Added for Nagabandi fidelity: seed env(s) per collection call
                seed_env(self.env, self.seed)
                self.collect_rollouts(self.env, num_steps=init_random_steps, use_planner=False)
            else:
                print("Collecting planner-based rollouts...")
                seed_env(self.env, self.seed)
                self.collect_rollouts(self.env, num_steps=rollout_steps, use_planner=True)

            print("Training dynamics model...")
            self.train_dynamics(epochs=epochs)

        self.writer.close()

        # Print elapsed training time in HH:MM:SS
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"✅ Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")

    # Action selection for evaluation
    def _predict(self, obs, deterministic: bool):
        return self.planner.plan(obs)

    # Use base evaluation; planner is stateless across steps by design

    def save(self):
        import torch as _torch
        save_path = os.path.join(self.output_dir, "model.pt")
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
        import os as _os
        # Accept either a direct file path or a run directory; expect 'model.pt'
        model_path = path
        if _os.path.isdir(model_path):
            model_path = _os.path.join(model_path, "model.pt")
            if not _os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No checkpoint found in directory '{path}' (expected model.pt)"
                )
        elif not _os.path.exists(model_path):
            # Common caller pattern passes .../model — resolve to model.pt in same dir
            base_dir = _os.path.dirname(model_path)
            alt = _os.path.join(base_dir, "model.pt")
            if _os.path.exists(alt):
                model_path = alt
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: '{path}' (also tried '{alt}')"
                )

        # Torch 2.6: default weights_only=True can block loading numpy objects in our ckpt
        ckpt = _torch.load(model_path, map_location=self.device, weights_only=False)
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
        return self
