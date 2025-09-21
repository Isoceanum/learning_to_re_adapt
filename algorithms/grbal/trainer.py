"""GrBAL trainer integrated with the repository's BaseTrainer."""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from algorithms.base_trainer import BaseTrainer
from algorithms.grbal.buffers import MetaSequenceBuffer
from algorithms.grbal.models import GrBALDynamicsModel, NormalizationStats, detach_params
from algorithms.mb_mpc.planner import RandomShootingPlanner
from utils.seeding import set_seed


@dataclass
class AdaptationConfig:
    context_length: int
    target_length: int
    batch_size: int
    use_second_order: bool


class AdaptedDynamicsWrapper:
    """Adapter that lets the planner query adapted parameters."""

    def __init__(self, model: GrBALDynamicsModel) -> None:
        self.model = model
        self.params: Optional[Dict[str, torch.Tensor]] = None

    @property
    def num_models(self) -> int:
        return 1

    def set_params(self, params: Optional[Dict[str, torch.Tensor]]) -> None:
        self.params = params

    def predict_next_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        model_indices=None,
        sample: bool = False,
    ) -> torch.Tensor:
        return self.model.predict_next_state(states, actions, params=self.params)


class GrBALTrainer(BaseTrainer):
    """PyTorch implementation of Nagabandi et al.'s GrBAL."""

    def __init__(self, config: Dict, output_dir: str) -> None:
        super().__init__(config, output_dir)
        self.device = torch.device(self.train_config.get("device", "cpu"))
        if self.seed is not None:
            set_seed(self.seed, deterministic_torch=True)
        self.env = self._make_env()
        if getattr(self.env, "num_envs", 1) != 1:
            raise ValueError("GrBALTrainer currently supports only single-environment training")
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        hidden_sizes = tuple(self.train_config.get("hidden_sizes", [512, 512]))
        inner_lr_init = float(self.train_config.get("inner_lr", 1e-2))
        learnable_inner_lr = bool(self.train_config.get("learnable_inner_lr", True))

        self.model = GrBALDynamicsModel(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_sizes=hidden_sizes,
            inner_lr_init=inner_lr_init,
            learnable_inner_lr=learnable_inner_lr,
        ).to(self.device)

        outer_lr = float(self.train_config.get("outer_lr", 1e-3))
        self.optimizer = Adam(self.model.parameters(), lr=outer_lr)

        context_len = int(self.train_config.get("adapt_window", 32))
        target_len = int(self.train_config.get("target_window", context_len))
        batch_size = int(self.train_config.get("meta_batch_size", 32))
        second_order = bool(self.train_config.get("second_order", True))
        self.adapt_cfg = AdaptationConfig(context_len, target_len, batch_size, second_order)

        self.meta_buffer = MetaSequenceBuffer()

        self.dynamics_adapter = AdaptedDynamicsWrapper(self.model)
        reward_fn = self._resolve_reward_fn(self.env)
        horizon = int(self.train_config.get("horizon", 15))
        n_candidates = int(self.train_config.get("n_candidates", 1000))
        discount = float(self.train_config.get("discount", 1.0))
        self.planner = RandomShootingPlanner(
            dynamics_model=self.dynamics_adapter,
            action_space=self.env.action_space,
            horizon=horizon,
            n_candidates=n_candidates,
            device=str(self.device),
            reward_fn=reward_fn,
            discount=discount,
        )

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))
        self.global_step = 0

        self._eval_history: deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.adapt_cfg.context_length
        )

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        total_iterations = int(self.train_config.get("total_iterations", 50))
        num_rollouts = int(self.train_config.get("num_rollouts", 16))
        max_path_length = int(self.train_config.get("max_path_length", 1000))
        meta_updates = int(self.train_config.get("meta_updates_per_iter", 64))

        for itr in range(total_iterations):
            print(f"\n=== GrBAL Iteration {itr + 1}/{total_iterations} ===")
            rollout_stats = self._collect_rollouts(num_rollouts, max_path_length)
            self.writer.add_scalar("rollout/avg_return", rollout_stats["avg_return"], itr)
            self.writer.add_scalar("rollout/avg_length", rollout_stats["avg_length"], itr)

            losses = self._meta_update(meta_updates)
            for step_idx, (inner_loss, meta_loss) in enumerate(losses):
                self.writer.add_scalar("loss/inner", inner_loss, self.global_step)
                self.writer.add_scalar("loss/meta", meta_loss, self.global_step)
                self.global_step += 1

        self.writer.close()

    def save(self) -> None:
        stats = self.model.get_normalization()
        payload = {
            "state_dict": self.model.state_dict(),
            "normalization": {
                "state_mean": stats.state_mean.cpu(),
                "state_std": stats.state_std.cpu(),
                "action_mean": stats.action_mean.cpu(),
                "action_std": stats.action_std.cpu(),
                "delta_mean": stats.delta_mean.cpu(),
                "delta_std": stats.delta_std.cpu(),
            },
            "adapt_config": self.adapt_cfg.__dict__,
        }
        path = os.path.join(self.output_dir, "model.pt")
        torch.save(payload, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        norm = checkpoint.get("normalization")
        if norm is not None:
            stats = NormalizationStats(
                state_mean=norm["state_mean"].to(self.device),
                state_std=norm["state_std"].to(self.device),
                action_mean=norm["action_mean"].to(self.device),
                action_std=norm["action_std"].to(self.device),
                delta_mean=norm["delta_mean"].to(self.device),
                delta_std=norm["delta_std"].to(self.device),
            )
            self.model.load_normalization(stats)

    # ------------------------------------------------------------------
    # Rollout collection with online adaptation
    # ------------------------------------------------------------------
    def _collect_rollouts(self, num_rollouts: int, max_path_length: int) -> Dict[str, float]:
        returns = []
        lengths = []
        all_states = []
        all_actions = []
        all_next_states = []

        for ep in range(num_rollouts):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            ep_return = 0.0
            history: deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self.adapt_cfg.context_length)
            path_states, path_actions, path_next_states = [], [], []

            while not done and steps < max_path_length:
                adapted_params = self._compute_adapted_params(history)
                action = self._select_action(obs, adapted_params)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                path_states.append(obs.astype(np.float32, copy=True))
                path_actions.append(action.astype(np.float32, copy=True))
                path_next_states.append(next_obs.astype(np.float32, copy=True))

                history.append((obs, action, next_obs))

                obs = next_obs
                steps += 1
                ep_return += float(reward)

            if path_states:
                states = np.asarray(path_states, dtype=np.float32)
                actions = np.asarray(path_actions, dtype=np.float32)
                next_states = np.asarray(path_next_states, dtype=np.float32)
                self.meta_buffer.add_path(states, actions, next_states)
                all_states.append(states)
                all_actions.append(actions)
                all_next_states.append(next_states)

            returns.append(ep_return)
            lengths.append(steps)

        if all_states:
            states_cat = torch.tensor(np.concatenate(all_states, axis=0), dtype=torch.float32, device=self.device)
            actions_cat = torch.tensor(np.concatenate(all_actions, axis=0), dtype=torch.float32, device=self.device)
            next_states_cat = torch.tensor(np.concatenate(all_next_states, axis=0), dtype=torch.float32, device=self.device)
            self.model.update_normalization(states_cat, actions_cat, next_states_cat)

        avg_return = float(np.mean(returns)) if returns else 0.0
        avg_length = float(np.mean(lengths)) if lengths else 0.0
        return {"avg_return": avg_return, "avg_length": avg_length}

    def _compute_adapted_params(self, history: deque) -> Optional[Dict[str, torch.Tensor]]:
        if len(history) < self.adapt_cfg.context_length:
            return None
        states = torch.tensor(np.stack([item[0] for item in history], axis=0), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([item[1] for item in history], axis=0), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([item[2] for item in history], axis=0), dtype=torch.float32, device=self.device)
        params = detach_params(self.model.named_parameter_dict())
        loss = self.model.loss(states, actions, next_states, params=params)
        return self.model.adapt(params, loss, create_graph=False)

    def _select_action(self, obs: np.ndarray, adapted_params: Optional[Dict[str, torch.Tensor]]) -> np.ndarray:
        self.dynamics_adapter.set_params(adapted_params)
        action = self.planner.plan(obs.astype(np.float32))
        return action

    # ------------------------------------------------------------------
    # Meta-update
    # ------------------------------------------------------------------
    def _meta_update(self, updates: int) -> Tuple[Tuple[float, float], ...]:
        losses = []
        self.model.train()
        for _ in range(updates):
            batch = self.meta_buffer.sample(
                batch_size=self.adapt_cfg.batch_size,
                context_len=self.adapt_cfg.context_length,
                target_len=self.adapt_cfg.target_length,
                device=self.device,
            )

            inner_losses = []
            meta_loss = 0.0

            for idx in range(self.adapt_cfg.batch_size):
                ctx_states = batch.context_states[idx]
                ctx_actions = batch.context_actions[idx]
                ctx_next_states = batch.context_next_states[idx]
                tgt_states = batch.target_states[idx]
                tgt_actions = batch.target_actions[idx]
                tgt_next_states = batch.target_next_states[idx]

                params = self.model.named_parameter_dict()
                inner_loss = self.model.loss(ctx_states, ctx_actions, ctx_next_states, params=params)
                adapted_params = self.model.adapt(params, inner_loss, create_graph=self.adapt_cfg.use_second_order)
                target_loss = self.model.loss(tgt_states, tgt_actions, tgt_next_states, params=adapted_params)

                inner_losses.append(inner_loss.detach().item())
                meta_loss = meta_loss + target_loss

            meta_loss = meta_loss / self.adapt_cfg.batch_size

            self.optimizer.zero_grad()
            meta_loss.backward()
            self.optimizer.step()

            losses.append((float(np.mean(inner_losses)), float(meta_loss.detach().item())))
        return tuple(losses)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _reset_eval_history(self) -> None:
        self._eval_history.clear()

    def evaluate(self) -> Dict[str, float]:
        eval_env = self._make_eval_env()
        episodes = int(self.eval_config.get("episodes", 10))
        seeds = self.eval_config.get("seeds", [None])
        if not isinstance(seeds, list) or not seeds:
            seeds = [None]
        gamma = float(self.train_config.get("gamma", 0.99))
        save_csv = bool(self.eval_config.get("save_csv", True))

        rewards: List[float] = []
        forward_progresses: List[float] = []
        rows = []

        total_episodes = len(seeds) * episodes
        print(
            f"🎯 Evaluating over {episodes} episode(s) per seed × {len(seeds)} seed(s) = {total_episodes} episodes"
        )

        eval_start = time.time()

        for seed in seeds:
            for ep in range(episodes):
                if seed is None:
                    obs, _ = eval_env.reset()
                else:
                    obs, _ = eval_env.reset(seed=int(seed))

                self._reset_eval_history()

                done = False
                total_reward = 0.0
                discounted_return = 0.0
                discount = 1.0
                steps = 0
                policy_time_ep = 0.0
                env_time_ep = 0.0

                com_start = float(obs[-3]) if obs.shape[0] >= 3 else None
                last_com = com_start

                while not done:
                    t0 = time.time()
                    adapted_params = self._compute_adapted_params(self._eval_history)
                    action = self._select_action(obs, adapted_params)
                    policy_time_ep += time.time() - t0

                    t1 = time.time()
                    next_obs, reward, terminated, truncated, info = eval_env.step(action)
                    env_time_ep += time.time() - t1
                    done = terminated or truncated

                    self._eval_history.append(
                        (
                            obs.astype(np.float32, copy=True),
                            action.astype(np.float32, copy=True),
                            next_obs.astype(np.float32, copy=True),
                        )
                    )

                    total_reward += float(reward)
                    discounted_return += discount * float(reward)
                    discount *= gamma
                    steps += 1
                    obs = next_obs
                    if obs.shape[0] >= 3:
                        last_com = float(obs[-3])

                rewards.append(total_reward)
                fp = (last_com - com_start) if (com_start is not None and last_com is not None) else 0.0
                forward_progresses.append(fp)

                rows.append(
                    {
                        "episode": ep,
                        "seed": "" if seed is None else int(seed),
                        "reward": total_reward,
                        "discounted_return": discounted_return,
                        "forward_progress": fp,
                        "steps": steps,
                        "policy_exec_time_s": policy_time_ep,
                        "env_step_time_s": env_time_ep,
                    }
                )

        reward_mean = float(np.mean(rewards)) if rewards else 0.0
        reward_std = float(np.std(rewards)) if rewards else 0.0
        fp_mean = float(np.mean(forward_progresses)) if forward_progresses else 0.0
        fp_std = float(np.std(forward_progresses)) if forward_progresses else 0.0

        elapsed = time.time() - eval_start
        print("✅ Evaluation summary:")
        print(f"- reward_mean: {reward_mean:.4f}")
        print(f"- reward_std: {reward_std:.4f}")
        print(f"- forward_progress_mean: {fp_mean:.4f}")
        print(f"- forward_progress_std: {fp_std:.4f}")
        print(f"- elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        if save_csv and rows:
            self._write_eval_csv(rows)

        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "forward_progress_mean": fp_mean,
            "forward_progress_std": fp_std,
        }

    def _predict(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        adapted_params = self._compute_adapted_params(self._eval_history)
        action = self._select_action(obs, adapted_params)
        return action

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _write_eval_csv(self, rows: List[Dict]) -> None:
        csv_path = os.path.join(self.output_dir, "evaluation.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _resolve_reward_fn(self, env) -> Optional:
        reward_fn = None
        base_env = getattr(env, "unwrapped", env)
        get_fn = getattr(base_env, "get_model_reward_fn", None)
        if callable(get_fn):
            reward_fn = get_fn()
        if reward_fn is None and hasattr(env, "env_method"):
            try:
                fns = env.env_method("get_model_reward_fn")
                if isinstance(fns, (list, tuple)) and fns:
                    reward_fn = fns[0]
            except Exception:
                pass
        if reward_fn is None:
            raise RuntimeError("Environment does not expose a model reward function required for planning")
        return reward_fn
