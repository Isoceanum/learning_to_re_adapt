import os
import time
import copy
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.base_trainer import BaseTrainer
from .dynamics import DynamicsModel  # deterministic MLP with normalization  # Added for Nagabandi fidelity
from algorithms.mb_mpc.buffer import ReplayBuffer
from .planner import RandomShootingPlanner
from utils.seeding import set_seed, seed_env  # Added for Nagabandi fidelity


class MBMPCTrainer(BaseTrainer):
    """
    Nagabandi-style MB-MPC in PyTorch, integrated with this repo's training loop:
      - Deterministic MLP dynamics (MSE on Δs) with input/output normalization
      - CEM MPC planner matching classic semantics
      - Uses env-provided model reward with forward_reward_weight scaled to match baseline
    """

    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        # MB-MPC relies on deterministic torch ops for reproducibility regardless of config defaults
        if self.seed is not None:
            set_seed(self.seed, deterministic_torch=True)
            self._deterministic_torch = True
        self.env = self._make_env()
        self.model = self._build_model()
        # Persistent datasets to mirror original TF behavior (accumulate splits)
        self._dataset_train = None  # dict(states, actions, next_states)
        self._dataset_val = None    # dict(states, actions, next_states)
        # Cache of the most recent collection chunk (this iteration only)
        self._last_collected = None

    def _make_env(self):
        import envs  # ensure custom envs registered
        import gymnasium as gym
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        env_id = self.config.get("env")
        n_envs = int(self.train_config.get("n_envs", self.config.get("n_envs", 1)))
        # Added for Nagabandi fidelity: pass seed through VecEnv constructor when possible
        seed_cfg = self.seed  # Added for Nagabandi fidelity
        env_kwargs = dict(
            exclude_current_positions_from_observation=True,
        )
        if n_envs > 1:
            base_seed = seed_cfg if seed_cfg is not None else int(time.time())
            try:
                return make_vec_env(
                    env_id,
                    n_envs=n_envs,
                    env_kwargs=env_kwargs,
                    seed=base_seed,
                    vec_env_cls=SubprocVecEnv,
                )
            except TypeError:
                # Older SB3 versions may not accept vec_env_cls/seed kwargs
                return make_vec_env(
                    env_id,
                    n_envs=n_envs,
                    env_kwargs=env_kwargs,
                    vec_env_cls=SubprocVecEnv,
                )
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
        norm_eps = float(train_cfg.get("norm_eps", 1e-10))
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
        discount = float(train_cfg.get("discount", 1.0))

        # No hidden defaults: horizon and n_candidates must be provided in YAML if desired

        self.device = torch.device(device)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))
        self.global_step = 0

        # Build deterministic dynamics model
        self.dynamics = DynamicsModel(state_dim, action_dim, hidden_sizes, norm_eps=norm_eps).to(self.device)
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        # Replay buffer
        self.buffer = ReplayBuffer(state_dim, action_dim)

        # Reward function from env (supports single env or SB3 VecEnv)
        reward_fn = None
        use_reward_model = bool(train_cfg.get("use_reward_model", False))
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

        if use_reward_model:
            print("[MBMPC-Nagabandi] use_reward_model=True requested, but no reward model is provided; using env reward.")

        # Planner: Random Shooting only (CEM removed)
        self.planner = RandomShootingPlanner(
            dynamics_model=self.dynamics,
            action_space=env.action_space,
            horizon=horizon,
            n_candidates=n_candidates,
            device=str(self.device),
            reward_fn=reward_fn,
            discount=discount,
            rng=None,  # will be set in train() after seeding
        )

        return self

    def _record_collected_chunk(self, states_list, actions_list, next_states_list):
        """Cache the most recently collected transitions as numpy arrays.
        Mirrors TF trainer where each iteration adds a new chunk to datasets.
        """
        if len(states_list) == 0:
            self._last_collected = None
            return
        states = np.asarray(states_list, dtype=np.float32)
        actions = np.asarray(actions_list, dtype=np.float32)
        next_states = np.asarray(next_states_list, dtype=np.float32)
        self._last_collected = (states, actions, next_states)

    def _normalize_step_budget(self, total_steps: int) -> int:
        """Convert a total-transition budget into per-env loop iterations."""
        total_steps = int(total_steps)
        if total_steps <= 0:
            return 0
        n_envs = int(getattr(self.env, "num_envs", 1))
        if n_envs <= 1:
            return total_steps
        # Divide by n_envs so the aggregate transitions match the single-env baseline.
        return max(1, int(math.ceil(total_steps / float(n_envs))))

    def _apply_terminal_observations(self, next_states, infos, dones):
        """Swap VecEnv reset obs with terminal obs before storing transitions."""
        if not np.any(dones):
            return next_states
        corrected = np.array(next_states, copy=True)
        info_seq = infos if isinstance(infos, (list, tuple)) else [infos] * len(corrected)
        for idx, done in enumerate(dones):
            if not done or idx >= len(info_seq):
                continue
            info = info_seq[idx] or {}
            if not isinstance(info, dict):
                continue
            terminal_obs = info.get("terminal_observation")
            if terminal_obs is None:
                terminal_obs = info.get("final_observation")
            if terminal_obs is None:
                continue
            if isinstance(terminal_obs, torch.Tensor):
                terminal_obs = terminal_obs.detach().cpu().numpy()
            corrected[idx] = terminal_obs
        return corrected

    def collect_rollouts(self, env, num_steps=1000, use_planner=False, max_path_length: int = 100):
        log_prefix = "planner" if use_planner else "random"
        is_vec = hasattr(env, "num_envs")

        if not is_vec:
            # Single-env path (Gymnasium API)
            # Added for Nagabandi fidelity: seed env reset
            try:
                state, _ = env.reset(seed=int(self.seed))
            except Exception:
                state, _ = env.reset()
            # Ensure random action sampling is reproducible when not using planner
            if not use_planner:
                try:
                    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                        env.action_space.seed(int(self.seed))
                except Exception:
                    pass
            ep_reward = 0.0
            ep_len = 0
            ep_rewards = []
            ep_lengths = []
            step_iter = range(num_steps)
            # Track this call's collected chunk
            _col_s, _col_a, _col_ns = [], [], []
            for _ in step_iter:
                if use_planner:
                    action = self.planner.plan(state)
                else:
                    # Reproducible uniform sampling via torch Generator
                    low = env.action_space.low
                    high = env.action_space.high
                    a = torch.rand((1, low.shape[0]), generator=self._cpu_gen)
                    a = torch.as_tensor(low, dtype=torch.float32) + a * (
                        torch.as_tensor(high, dtype=torch.float32) - torch.as_tensor(low, dtype=torch.float32)
                    )
                    action = a.squeeze(0).cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated or (ep_len + 1 >= int(max_path_length))

                self.buffer.add(state, action, next_state)
                _col_s.append(state)
                _col_a.append(action)
                _col_ns.append(next_state)
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
            # Cache this chunk
            self._record_collected_chunk(_col_s, _col_a, _col_ns)
            return

        # Vectorized env path (SB3 VecEnv API)
        # Reset returns obs only
        # Added for Nagabandi fidelity: seed vec reset
        states = env.reset()
        # Episode stats per env
        n_envs = int(getattr(env, "num_envs", len(states)))
        ep_rewards = np.zeros(n_envs, dtype=np.float64)
        ep_lengths = np.zeros(n_envs, dtype=np.int64)
        completed_rewards = []
        completed_lengths = []
        step_iter = range(num_steps)
        _col_s, _col_a, _col_ns = [], [], []
        for _ in step_iter:
            if use_planner:
                actions = self.planner.plan(states)  # (n_envs, A)
            else:
                # Sample per-env actions
                low, high = env.action_space.low, env.action_space.high
                a = torch.rand((n_envs, low.shape[0]), generator=self._cpu_gen)
                low_t = torch.as_tensor(low, dtype=torch.float32)
                high_t = torch.as_tensor(high, dtype=torch.float32)
                actions = (low_t + a * (high_t - low_t)).cpu().numpy()

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

            dones_arr = np.asarray(dones, dtype=bool)
            corrected_next_states = self._apply_terminal_observations(next_states, infos, dones_arr)
            # Store transitions using true terminal states before VecEnv resets.
            self.buffer.add_batch(states, actions, corrected_next_states)
            _col_s.extend(list(states))
            _col_a.extend(list(actions))
            _col_ns.extend(list(corrected_next_states))
            states = next_states

            # Update episode stats
            ep_rewards += rewards.astype(np.float64)
            ep_lengths += 1
            if np.any(dones_arr):
                done_idx = np.where(dones_arr)[0]
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
        # Cache chunk
        self._record_collected_chunk(_col_s, _col_a, _col_ns)

    def collect_rollouts_episodes(self, env, num_rollouts=10, use_planner=False, max_path_length: int = 100):
        """Episode-based collection: exactly num_rollouts episodes capped at max_path_length.
        Caches the collected chunk for dataset accumulation.
        """
        log_prefix = "planner" if use_planner else "random"
        is_vec = hasattr(env, "num_envs")

        _col_s, _col_a, _col_ns = [], [], []

        if not is_vec:
            try:
                state, _ = env.reset(seed=int(self.seed))
            except Exception:
                state, _ = env.reset()

            ep_rewards = []
            ep_lengths = []
            for _ in range(int(num_rollouts)):
                ep_len = 0
                ep_rew = 0.0
                while True:
                    if use_planner:
                        action = self.planner.plan(state)
                    else:
                        low = env.action_space.low
                        high = env.action_space.high
                        a = torch.rand((1, low.shape[0]), generator=self._cpu_gen)
                        a = torch.as_tensor(low, dtype=torch.float32) + a * (
                            torch.as_tensor(high, dtype=torch.float32) - torch.as_tensor(low, dtype=torch.float32)
                        )
                        action = a.squeeze(0).cpu().numpy()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or (ep_len + 1 >= int(max_path_length))

                    self.buffer.add(state, action, next_state)
                    _col_s.append(state)
                    _col_a.append(action)
                    _col_ns.append(next_state)
                    state = next_state
                    ep_len += 1
                    ep_rew += float(reward)
                    if done:
                        ep_rewards.append(ep_rew)
                        ep_lengths.append(ep_len)
                        state, _ = env.reset()
                        break

            if ep_rewards:
                import numpy as _np
                mean_rew = float(_np.mean(ep_rewards))
                mean_len = float(_np.mean(ep_lengths))
                print(f"Rollout {log_prefix} (episodes): ep_rew_mean={mean_rew:.2f} ep_len_mean={mean_len:.1f} episodes={len(ep_rewards)}")
                self.writer.add_scalar(f"{log_prefix}/ep_rew_mean", mean_rew, self.global_step)
                self.writer.add_scalar(f"{log_prefix}/ep_len_mean", mean_len, self.global_step)

            self._record_collected_chunk(_col_s, _col_a, _col_ns)
            return

        # Vectorized env episode collection
        states = env.reset()

        n_envs = int(getattr(env, "num_envs", len(states)))
        ep_lens = np.zeros(n_envs, dtype=np.int64)
        ep_rewards = np.zeros(n_envs, dtype=np.float64)
        episodes_collected = 0
        while episodes_collected < int(num_rollouts):
            if use_planner:
                actions = self.planner.plan(states)
            else:
                low, high = env.action_space.low, env.action_space.high
                a = torch.rand((n_envs, low.shape[0]), generator=self._cpu_gen)
                low_t = torch.as_tensor(low, dtype=torch.float32)
                high_t = torch.as_tensor(high, dtype=torch.float32)
                actions = (low_t + a * (high_t - low_t)).cpu().numpy()

            out = env.step(actions)
            if len(out) == 4:
                next_states, rewards, dones, infos = out
            elif len(out) == 5:
                next_states, rewards, terminated, truncated, infos = out
                dones = np.asarray(terminated) | np.asarray(truncated)
            else:
                raise RuntimeError("Unexpected VecEnv.step return format")

            dones_arr = np.asarray(dones, dtype=bool)
            corrected_next_states = self._apply_terminal_observations(next_states, infos, dones_arr)
            # Store transitions using the true terminal frames.
            self.buffer.add_batch(states, actions, corrected_next_states)
            _col_s.extend(list(states))
            _col_a.extend(list(actions))
            _col_ns.extend(list(corrected_next_states))

            ep_lens += 1
            ep_rewards += rewards.astype(np.float64)
            timeouts = ep_lens >= int(max_path_length)
            finished = dones_arr | timeouts
            if np.any(finished):
                episodes_collected += int(np.sum(finished))
                ep_lens[finished] = 0
                ep_rewards[finished] = 0.0
            states = next_states

        self._record_collected_chunk(_col_s, _col_a, _col_ns)

    def train_dynamics(self, epochs=50):
        # Use only the latest collected chunk to update persistent datasets
        if self._last_collected is None:
            # Fallback to all data (first iteration safety)
            states, actions, next_states = self.buffer.get_all()
            if len(states) == 0:
                print("[MBMPC-Nagabandi] No data available to train dynamics.")
                return
        else:
            states, actions, next_states = self._last_collected

        # Split current chunk into train/val
        N = states.shape[0]
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        split = int(N * (1.0 - float(self.val_ratio)))
        train_idx, val_idx = idxs[:split], idxs[split:]
        cur_train = dict(states=states[train_idx], actions=actions[train_idx], next_states=next_states[train_idx])
        cur_val = dict(states=states[val_idx], actions=actions[val_idx], next_states=next_states[val_idx])

        # Accumulate persistent datasets
        if self._dataset_train is None:
            self._dataset_train = cur_train
            self._dataset_val = cur_val
        else:
            self._dataset_train = dict(
                states=np.concatenate([self._dataset_train['states'], cur_train['states']]),
                actions=np.concatenate([self._dataset_train['actions'], cur_train['actions']]),
                next_states=np.concatenate([self._dataset_train['next_states'], cur_train['next_states']]),
            )
            self._dataset_val = dict(
                states=np.concatenate([self._dataset_val['states'], cur_val['states']]),
                actions=np.concatenate([self._dataset_val['actions'], cur_val['actions']]),
                next_states=np.concatenate([self._dataset_val['next_states'], cur_val['next_states']]),
            )

        # Fit normalization on all accumulated training data
        all_states = self._dataset_train['states']
        all_actions = self._dataset_train['actions']
        all_next_states = self._dataset_train['next_states']
        self.dynamics.fit_normalization(all_states, all_actions, all_next_states)

        train_states = torch.tensor(self._dataset_train['states'], dtype=torch.float32, device=self.device)
        train_actions = torch.tensor(self._dataset_train['actions'], dtype=torch.float32, device=self.device)
        train_next_states = torch.tensor(self._dataset_train['next_states'], dtype=torch.float32, device=self.device)

        val_states = torch.tensor(self._dataset_val['states'], dtype=torch.float32, device=self.device)
        val_actions = torch.tensor(self._dataset_val['actions'], dtype=torch.float32, device=self.device)
        val_next_states = torch.tensor(self._dataset_val['next_states'], dtype=torch.float32, device=self.device)

        # Early stopping to mirror original: stop when validation EMA worsens; no restore-best
        rolling_persistency = float(self.train_config.get("rolling_average_persitency", 0.99))
        use_ema_early_stop = bool(self.train_config.get("use_ema_early_stop", True))
        ema = None
        prev_ema = None

        for epoch in range(epochs):
            # Shuffle indices
            n_train = len(train_states)
            # Added for Nagabandi fidelity: use a CPU generator for reproducible shuffling, then move to device
            idxs = torch.randperm(n_train, generator=self._cpu_gen)
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
            # Progress bar removed; keep plain logging only

            # Update rolling EMA and stop if it worsens (original TF behavior)
            if ema is None:
                # Initialize with a higher value to avoid too-early stopping
                # Match TF heuristic: 1.5*val_loss (or val_loss/1.5 if negative)
                if val_loss >= 0:
                    ema = 1.5 * val_loss
                    prev_ema = 2.0 * val_loss
                else:
                    ema = val_loss / 1.5
                    prev_ema = val_loss / 2.0
            else:
                ema = rolling_persistency * ema + (1.0 - rolling_persistency) * val_loss

            # Stop when EMA worsens compared to previous EMA (optional)
            if use_ema_early_stop and prev_ema is not None and (prev_ema < ema):
                print('Stopping Training of Model since its valid_loss_rolling_average decreased')
                break
            prev_ema = ema

    def train(self):
        start_time = time.time()
        cfg = self.train_config
        n_iterations = int(cfg.get("total_iterations", cfg.get("iterations", 50)))
        init_random_steps = int(cfg.get("init_random_steps", 5000))
        rollout_steps = int(cfg.get("rollout_steps", 32000))
        epochs = int(cfg.get("epochs", 50))
        max_path_length = int(cfg.get("max_path_length", 100))

        # Added for Nagabandi fidelity: global seeding
        if self.seed is None:
            self.seed = int(self.config.get("seed", cfg.get("seed", 42)))
        # Enforce deterministic torch ops for full reproducibility
        set_seed(self.seed, deterministic_torch=True)
        # Added for Nagabandi fidelity: separate CPU and device generators
        self._cpu_gen = torch.Generator().manual_seed(self.seed)
        try:
            self._dev_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        except Exception:
            self._dev_gen = self._cpu_gen
        # Propagate RNG to planner for CEM/RS sampling
        if hasattr(self, "planner") and hasattr(self.planner, "rng"):
            self.planner.rng = self._dev_gen

        print(f"🚀 Starting Nagabandi MB-MPC: iterations={n_iterations}, epochs={epochs}")

        for itr in range(n_iterations):
            print(f"\n=== Iteration {itr+1}/{n_iterations} ===")
            if itr == 0:
                print("Collecting initial random rollouts...")
                seed_env(self.env, self.seed)
                num_rollouts = self.train_config.get("num_rollouts")
                if num_rollouts is not None:
                    self.collect_rollouts_episodes(self.env, num_rollouts=int(num_rollouts), use_planner=False, max_path_length=max_path_length)
                else:
                    # Fallback to step budget
                    total_steps = int(self.train_config.get("init_random_steps", 0))
                    if total_steps <= 0:
                        # Reasonable default matching paper: 10x100
                        total_steps = 1000
                    steps = self._normalize_step_budget(total_steps)  # Keep aggregate steps consistent across n_envs.
                    self.collect_rollouts(self.env, num_steps=steps, use_planner=False, max_path_length=max_path_length)
            else:
                print("Collecting planner-based rollouts...")
                seed_env(self.env, self.seed)
                num_rollouts = self.train_config.get("num_rollouts")
                if num_rollouts is not None:
                    self.collect_rollouts_episodes(self.env, num_rollouts=int(num_rollouts), use_planner=True, max_path_length=max_path_length)
                else:
                    steps = int(self.train_config.get("rollout_steps", 0))
                    if steps <= 0:
                        steps = 1000
                    steps = self._normalize_step_budget(steps)  # Ensure total planner rollouts match single-env budget.
                    self.collect_rollouts(self.env, num_steps=steps, use_planner=True, max_path_length=max_path_length)

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
        # Reset planner RNG for evaluation reproducibility across runs
        if not getattr(self, "_eval_rng_initialized", False):
            try:
                import torch as _torch
                base_seed = int(self.config.get("seed", getattr(self, "seed", 42)))
                self._eval_rng = _torch.Generator(device=self.device).manual_seed(base_seed)
            except Exception:
                import torch as _torch
                base_seed = int(self.config.get("seed", getattr(self, "seed", 42)))
                self._eval_rng = _torch.Generator().manual_seed(base_seed)
            if hasattr(self, "planner") and hasattr(self.planner, "rng"):
                self.planner.rng = self._eval_rng
            self._eval_rng_initialized = True
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
            # Restore saved normalization arrays
            self.dynamics.state_mean = norm.get("state_mean")
            self.dynamics.state_std = norm.get("state_std")
            self.dynamics.action_mean = norm.get("action_mean")
            self.dynamics.action_std = norm.get("action_std")
            self.dynamics.delta_mean = norm.get("delta_mean")
            self.dynamics.delta_std = norm.get("delta_std")
            # Rebuild cached tensors on the current device without recomputing stats
            if hasattr(self.dynamics, "refresh_cached_normalization"):
                self.dynamics.refresh_cached_normalization()
            else:
                # Fallback: construct tensors directly
                device = self.device
                dtype = _torch.float32
                self.dynamics._state_mean_t = _torch.as_tensor(self.dynamics.state_mean, dtype=dtype, device=device)
                self.dynamics._state_std_t = _torch.as_tensor(self.dynamics.state_std, dtype=dtype, device=device)
                self.dynamics._action_mean_t = _torch.as_tensor(self.dynamics.action_mean, dtype=dtype, device=device)
                self.dynamics._action_std_t = _torch.as_tensor(self.dynamics.action_std, dtype=dtype, device=device)
                self.dynamics._delta_mean_t = _torch.as_tensor(self.dynamics.delta_mean, dtype=dtype, device=device)
                self.dynamics._delta_std_t = _torch.as_tensor(self.dynamics.delta_std, dtype=dtype, device=device)
        return self
