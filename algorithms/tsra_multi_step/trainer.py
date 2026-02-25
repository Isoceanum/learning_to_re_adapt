import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_  # gradient clipping safeguard

from algorithms.base_trainer import BaseTrainer
from algorithms.untils import make_dynamics_model
from algorithms.tsra_multi_step.planner import (
    CrossEntropyMethodPlanner,
    MPPIPlanner,
    RandomShootingPlanner,
)
from algorithms.tsra_multi_step.transition_buffer import KStepWindowBuffer
from algorithms.tsra_multi_step.tsra_residual_adapter import TSRAResidualAdapter
from algorithms.tsra.residual_dynamics_wrapper import ResidualDynamicsWrapper
from evaluation.model_error import compute_k_step_rmse_for_episode
from utils.seed import set_seed


class TSRAMultiStepTrainer(BaseTrainer):
    """Task-Specific Residual Adapter trained with true K-step open-loop loss for MPPI."""

    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()

        # core configs
        self.iterations = int(self.train_config.get("iterations", 1))
        self.steps_per_iter = int(self.train_config.get("steps_per_iteration", 1000))
        self.max_episode_length = int(self.train_config.get("max_episode_length", 1000))
        self.train_epochs = int(self.train_config.get("train_epochs", 30))
        self.batch_size = int(self.train_config.get("batch_size", 256))

        # rollout / loss horizon
        self.k_horizon = int(self.train_config.get("k_horizon", 5))
        self.lambda_anchor = float(self.train_config.get("lambda_anchor", 0.1))
        self.k_curriculum = self.train_config.get("k_curriculum", None)

        # safety knobs
        self.grad_clip_norm = float(self.train_config.get("grad_clip_norm", 0.0))
        self.residual_bound = float(self.train_config.get("residual_bound", 0.0))
        self.correction_l2 = float(self.train_config.get("correction_l2", 0.0))
        self.l2_penalty_weight = float(self.train_config.get("l2_penalty_weight", 0.0))

        # evaluation
        self.quick_eval_episodes = int(self.train_config.get("quick_eval_episodes", 0))

        # models
        self.pretrained_dynamics_model = self._load_pretrained_dynamics_model()
        self.residual_adapter = self._make_residual_adapter()
        self.optimizer = self._make_optimizer()
        self.residual_wrapper = ResidualDynamicsWrapper(self.pretrained_dynamics_model, self.residual_adapter)

        # planner (MPPI)
        self.planner = self._make_planner(self.residual_wrapper.predict_next_state)
        self.base_planner = self._make_planner(self.pretrained_dynamics_model.predict_next_state)

        # buffer
        self.buffer = KStepWindowBuffer(self.train_config.get("valid_split_ratio", 0.1), self.train_seed)

    # ------------------------------------------------------------------
    # Builders
    def _load_pretrained_dynamics_model(self):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]

        with open(config_path, "r") as f:
            dyn_cfg = yaml.safe_load(f)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        model = make_dynamics_model(dyn_cfg, obs_dim, act_dim, self.train_seed).to(self.device)
        model.load_saved_model(model_path)
        model.freeze()
        return model

    def _make_residual_adapter(self):
        ra_cfg = self.train_config.get("residual_adapter", {})
        hidden = ra_cfg.get("hidden_sizes", [64, 64])
        activation = ra_cfg.get("activation", "relu")
        input_mode = ra_cfg.get("input_mode", "sabase")
        output_mode = ra_cfg.get("output_mode", "delta_next_obs")
        zero_init = bool(ra_cfg.get("zero_init_last_layer", True))
        bound_type = ra_cfg.get("bound_type", "none")  # none | tanh | clamp
        bound_scale = ra_cfg.get("bound_scale", 1.0)
        bound_max_abs = ra_cfg.get("bound_max_abs", 0.0)

        return TSRAResidualAdapter(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            input_mode=input_mode,
            hidden_sizes=hidden,
            activation=activation,
            output_mode=output_mode,
            zero_init_last_layer=zero_init,
            bound_type=bound_type,
            bound_scale=bound_scale,
            bound_max_abs=bound_max_abs,
        ).to(self.device)

    def _make_optimizer(self):
        lr = float(self.train_config.get("learning_rate", 1e-3))
        wd = float(self.train_config.get("weight_decay", 0.0))
        return optim.AdamW(self.residual_adapter.parameters(), lr=lr, weight_decay=wd)

    def _make_planner(self, dynamics_fn):
        plan_cfg = self.train_config.get("planner")
        base_env = getattr(self.env, "unwrapped", self.env)
        reward_fn = base_env.get_model_reward_fn()
        planner_type = plan_cfg.get("type")
        horizon = int(plan_cfg.get("horizon"))
        n_candidates = int(plan_cfg.get("n_candidates"))
        discount = float(plan_cfg.get("discount"))
        action_space = self.env.action_space

        if planner_type == "rs":
            return RandomShootingPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                action_space.low,
                action_space.high,
                self.device,
                discount,
            )

        if planner_type == "cem":
            return CrossEntropyMethodPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                action_space.low,
                action_space.high,
                self.device,
                discount,
                int(plan_cfg.get("num_cem_iters")),
                float(plan_cfg.get("percent_elites")),
                float(plan_cfg.get("alpha")),
                self.train_seed,
            )

        if planner_type == "mppi":
            return MPPIPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                action_space.low,
                action_space.high,
                self.device,
                discount,
                float(plan_cfg.get("noise_sigma")),
                float(plan_cfg.get("lambda_")),
                self.train_seed,
            )

        raise AttributeError(f"Planner type {planner_type} not supported")

    # ------------------------------------------------------------------
    # Data collection
    def _collect_env_steps(self, iteration_idx, use_base_only=False):
        start = time.time()
        steps = 0
        ep_returns, ep_fp, ep_len = [], [], []
        planner = self.base_planner if use_base_only else self.planner

        ess_list = []
        sat_list = []
        ret_mean_list = []
        ret_std_list = []

        while steps < self.steps_per_iter:
            obs, _ = self.env.reset()
            done = False
            episode_return = 0.0
            x0 = None
            traj_obs, traj_act, traj_next = [], [], []

            while not done and len(traj_act) < self.max_episode_length and steps < self.steps_per_iter:
                act, plan_info = self._plan_with_info(planner, obs)
                next_obs, reward, terminated, truncated, info = self._step_env(act)
                done = terminated or truncated

                if plan_info:
                    if "ess_ratio" in plan_info:
                        ess_list.append(plan_info["ess_ratio"])
                    if "sat_ratio" in plan_info:
                        sat_list.append(plan_info["sat_ratio"])
                    if "returns_mean" in plan_info:
                        ret_mean_list.append(plan_info["returns_mean"])
                    if "returns_std" in plan_info:
                        ret_std_list.append(plan_info["returns_std"])

                traj_obs.append(obs)
                traj_act.append(act)
                traj_next.append(next_obs)

                episode_return += float(reward)
                x_pos = float(self._get_forward_position(info))
                if x0 is None:
                    x0 = x_pos
                obs = next_obs
                steps += 1

            self.buffer.add_trajectory(traj_obs, traj_act, traj_next)
            ep_returns.append(episode_return)
            ep_fp.append(float(x_pos - x0) if (x0 is not None and len(traj_act) > 0) else 0.0)
            ep_len.append(len(traj_act))

        elapsed = time.time() - start
        ess_mean = np.mean(ess_list) if len(ess_list) else float("nan")
        sat_mean = np.mean(sat_list) if len(sat_list) else float("nan")
        ret_mean = np.mean(ret_mean_list) if len(ret_mean_list) else float("nan")
        ret_std = np.mean(ret_std_list) if len(ret_std_list) else float("nan")
        print(
            f"collect iter={iteration_idx} base_only={use_base_only} steps={steps} episodes={len(ep_returns)} "
            f"reward_mean={np.mean(ep_returns):.3f} ± {np.std(ep_returns):.3f} "
            f"forward_mean={np.mean(ep_fp):.3f} ± {np.std(ep_fp):.3f} time={elapsed:.1f}s "
            f"ess={ess_mean:.3f} sat={sat_mean:.3f} ret_mean={ret_mean:.3f} ret_std={ret_std:.3f}"
        )

    def _plan_with_info(self, planner, obs):
        try:
            act, info = planner.plan(obs, return_info=True)
        except TypeError:
            act = planner.plan(obs)
            info = getattr(planner, "last_plan_info", None)
        if torch.is_tensor(act):
            act = act.detach().cpu().numpy()
        return act, info

    # ------------------------------------------------------------------
    # Training helpers
    def _k_rollout_loss(self, obs_b, act_seq_b, tgt_seq_b, k_use):
        base = self.pretrained_dynamics_model
        eps = 1e-8

        # normalize inputs once
        obs_norm = (obs_b - base.mean_obs) / base.std_obs

        total_k_loss = 0.0
        total_steps = 0

        # open-loop rollout
        pred_state = obs_b
        base_pred_state = obs_b
        base_norm_state = obs_norm

        for step in range(k_use):
            act_t = act_seq_b[:, step, :]
            tgt_next = tgt_seq_b[:, step, :]

            # base prediction
        with torch.no_grad():
            base_next = base.predict_next_state(pred_state, act_t)

            base_next_norm = (base_next - base.mean_obs) / base.std_obs

            # adapter correction
            act_norm = (act_t - base.mean_act) / base.std_act
            corr = self.residual_adapter(base_norm_state, act_norm, base_next_norm)
            if self.residual_bound > 0:
                corr = torch.clamp(corr, -self.residual_bound, self.residual_bound)

            pred_next_norm = base_next_norm + corr
            pred_next = pred_next_norm * base.std_obs + base.mean_obs

            tgt_next_norm = (tgt_next - base.mean_obs) / base.std_obs

            step_loss = torch.mean((pred_next_norm - tgt_next_norm) ** 2)
            total_k_loss = total_k_loss + step_loss
            total_steps += 1

            # advance
            pred_state = pred_next.detach()  # open-loop
            base_norm_state = (pred_state - base.mean_obs) / base.std_obs

        rollout_mse = total_k_loss / max(1, total_steps)

        # 1-step anchor (teacher forcing)
        with torch.no_grad():
            base_one = base.predict_next_state(obs_b, act_seq_b[:, 0, :])
        base_one_norm = (base_one - base.mean_obs) / base.std_obs
        tgt_one_norm = (tgt_seq_b[:, 0, :] - base.mean_obs) / base.std_obs
        act_one_norm = (act_seq_b[:, 0, :] - base.mean_act) / base.std_act
        corr_one = self.residual_adapter(obs_norm, act_one_norm, base_one_norm)
        if self.residual_bound > 0:
            corr_one = torch.clamp(corr_one, -self.residual_bound, self.residual_bound)
        pred_one_norm = base_one_norm + corr_one
        one_step_mse = torch.mean((pred_one_norm - tgt_one_norm) ** 2)

        # combine
        total_loss = self.lambda_anchor * one_step_mse + (1.0 - self.lambda_anchor) * rollout_mse

        # optional correction penalties
        corr_norm_sq = torch.sum(corr_one ** 2, dim=1).mean()
        if self.correction_l2 > 0:
            total_loss = total_loss + self.correction_l2 * corr_norm_sq
        if self.l2_penalty_weight > 0:
            total_loss = total_loss + self.l2_penalty_weight * corr_norm_sq

        return {
            "loss": total_loss,
            "one_step_mse": one_step_mse.detach(),
            "rollout_mse": rollout_mse.detach(),
            "corr_ratio": (corr_one.norm(dim=1) / (base_one_norm.norm(dim=1) + eps)).mean().detach(),
            "corr_norm": corr_one.norm(dim=1).mean().detach(),
        }

    def _run_epoch(self, loader, k_use, train=True):
        stats = {"loss": [], "one": [], "k": [], "corr_norm": [], "corr_ratio": []}
        self.residual_adapter.train(mode=train)
        for obs_b, act_seq_b, tgt_seq_b in loader:
            obs_b = obs_b.to(self.device)
            act_seq_b = act_seq_b.to(self.device)
            tgt_seq_b = tgt_seq_b.to(self.device)

            out = self._k_rollout_loss(obs_b, act_seq_b, tgt_seq_b, k_use)
            loss = out["loss"]

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(self.residual_adapter.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            stats["loss"].append(float(loss.item()))
            stats["one"].append(float(out["one_step_mse"].item()))
            stats["k"].append(float(out["rollout_mse"].item()))
            stats["corr_norm"].append(float(out["corr_norm"].item()))
            stats["corr_ratio"].append(float(out["corr_ratio"].item()))

        return {k: float(np.mean(v)) if len(v) else float("nan") for k, v in stats.items()}

    # ------------------------------------------------------------------
    def train(self):
        print("Starting TSRA K-step MPPI training")
        start_time = time.time()

        for it in range(self.iterations):
            print(f"\n==== Iteration {it+1}/{self.iterations} ====")
            use_base_only = bool(self.train_config.get("bootstrap_base_only", True)) and it == 0
            self._collect_env_steps(it, use_base_only=use_base_only)

            k_use = self._resolve_curriculum_k(it)
            train_loader, val_loader, stats = self.buffer.build_dataloaders(k_use, self.batch_size)
            print(
                f"dataset: K={k_use} train_windows={stats['train_windows']} val_windows={stats['val_windows']} "
                f"split={self.buffer.num_trajectories()}"
            )

            for ep in range(self.train_epochs):
                train_stats = self._run_epoch(train_loader, k_use, train=True)
                val_stats = self._run_epoch(val_loader, k_use, train=False)
                bound_cfg = self.residual_adapter.bound_config()
                print(
                    f"epoch {ep+1}/{self.train_epochs} "
                    f"train_loss={train_stats['loss']:.6f} one={train_stats['one']:.6f} k={train_stats['k']:.6f} "
                    f"corr_norm={train_stats['corr_norm']:.4f} corr_ratio={train_stats['corr_ratio']:.4f} "
                    f"val_loss={val_stats['loss']:.6f} one={val_stats['one']:.6f} k={val_stats['k']:.6f} "
                    f"corr_norm={val_stats['corr_norm']:.4f} corr_ratio={val_stats['corr_ratio']:.4f} "
                    f"bound={bound_cfg['type']} scale={bound_cfg['scale']:.3f} max_abs={bound_cfg['max_abs']:.3f}"
                )

            if self.quick_eval_episodes > 0:
                self._quick_eval(self.quick_eval_episodes)

        elapsed = time.time() - start_time
        print(f"Training finished in {elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    def _resolve_curriculum_k(self, iteration_idx):
        if self.k_curriculum is None:
            return self.k_horizon
        schedule = self.k_curriculum
        # schedule: list of {iter: int, k: int}
        k_val = self.k_horizon
        for item in schedule:
            if iteration_idx >= int(item.get("iter", 0)):
                k_val = int(item["k"])
        return k_val

    def _quick_eval(self, episodes):
        env = self._make_eval_env(seed=self.eval_config.get("seeds", [self.train_seed])[0])
        rewards, fps = [], []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_r, x0, xl = 0.0, None, None
            while not done:
                action = self.predict(obs)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                ep_r += float(r)
                x = float(self._get_forward_position(info))
                if x0 is None:
                    x0 = x
                xl = x
            rewards.append(ep_r)
            fps.append(float(xl - x0) if x0 is not None and xl is not None else 0.0)
        env.close()
        print(
            f"[quick-eval] episodes={episodes} reward_mean={np.mean(rewards):.3f} ± {np.std(rewards):.3f} "
            f"fp_mean={np.mean(fps):.3f} ± {np.std(fps):.3f}"
        )

    # ------------------------------------------------------------------
    def predict(self, obs):
        action = self.planner.plan(obs)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        return action

    def save(self):
        payload = {"residual_adapter_state": self.residual_adapter.state_dict(), "optimizer_state": self.optimizer.state_dict()}
        path = os.path.join(self.output_dir, "tsra_kstep_adapter.pt")
        torch.save(payload, path)
        print(f"Saved adapter to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.residual_adapter.load_state_dict(ckpt.get("residual_adapter_state", ckpt))
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Loaded adapter from {path}")
        return self
