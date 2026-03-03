import math
import os
import time
import yaml

import numpy as np
import torch

from algorithms.base_trainer import BaseTrainer
from algorithms.reskilling.dynamics_model import DynamicsModel
from algorithms.reskilling.planner import CrossEntropyMethodPlanner, RandomShootingPlanner, MPPIPlanner
from algorithms.reskilling.transition_buffer import TransitionBuffer


class ReskillingTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()

        self.dynamics_model = self._load_pretrained_dynamics_model().to(self.device)
        self.optimizer = self._make_optimizer()

        # Keep a frozen base model for optional bootstrap collection.
        self.base_dynamics_model = self._load_pretrained_dynamics_model().to(self.device)
        self.base_dynamics_model.freeze()

        self.planner = self._make_planner()
        self.base_planner = self._make_base_planner()
        self.buffer = None

    def _make_dynamics_model(self, pretrained_config):
        dynamics_cfg = pretrained_config.get("train", {}).get("dynamics_model")
        if dynamics_cfg is None:
            raise AttributeError("Missing train.dynamics_model in pretrained config")

        hidden_sizes = dynamics_cfg.get("hidden_sizes")
        if hidden_sizes is None:
            raise AttributeError("Missing train.dynamics_model.hidden_sizes in pretrained config")

        learning_rate = float(self.train_config.get("learning_rate", dynamics_cfg.get("learning_rate", 1e-3)))
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        return DynamicsModel(obs_dim, act_dim, hidden_sizes, learning_rate, self.train_seed)

    def _load_pretrained_dynamics_model(self):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Pretrained model not found: {model_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Pretrained config not found: {config_path}")

        with open(config_path, "r") as f:
            pretrained_config = yaml.safe_load(f)

        model = self._make_dynamics_model(pretrained_config).to(self.device)
        model.load_saved_model(model_path)

        # Reskilling trains all parameters.
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()

        return model

    def _make_optimizer(self):
        learning_rate = float(self.train_config.get("learning_rate", 1e-4))
        return torch.optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)

    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing train.planner config in YAML")

        planner_type = planner_config.get("type")
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        act_low = self.env.action_space.low
        act_high = self.env.action_space.high
        dynamics_fn = self.dynamics_model.predict_next_state

        if planner_type == "rs":
            return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)

        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))
            return CrossEntropyMethodPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                act_low,
                act_high,
                self.device,
                discount,
                num_cem_iters,
                percent_elites,
                alpha,
                self.train_seed,
            )

        if planner_type == "mppi":
            noise_sigma = float(planner_config.get("noise_sigma"))
            lambda_ = float(planner_config.get("lambda_"))
            return MPPIPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                act_low,
                act_high,
                self.device,
                discount,
                noise_sigma,
                lambda_,
                self.train_seed,
            )

        raise AttributeError(f"Planner type {planner_type} not supported")

    def _make_base_planner(self):
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing train.planner config in YAML")

        planner_type = planner_config.get("type")
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        act_low = self.env.action_space.low
        act_high = self.env.action_space.high
        dynamics_fn = self.base_dynamics_model.predict_next_state

        if planner_type == "rs":
            return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)

        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))
            return CrossEntropyMethodPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                act_low,
                act_high,
                self.device,
                discount,
                num_cem_iters,
                percent_elites,
                alpha,
                self.train_seed,
            )

        if planner_type == "mppi":
            noise_sigma = float(planner_config.get("noise_sigma"))
            lambda_ = float(planner_config.get("lambda_"))
            return MPPIPlanner(
                dynamics_fn,
                reward_fn,
                horizon,
                n_candidates,
                act_low,
                act_high,
                self.device,
                discount,
                noise_sigma,
                lambda_,
                self.train_seed,
            )

        raise AttributeError(f"Planner type {planner_type} not supported")

    def _collect_env_steps(self, steps_target, max_episode_length, use_base_only=False):
        buffer = TransitionBuffer(
            valid_split_ratio=float(self.train_config["valid_split_ratio"]),
            seed=self.train_seed,
        )

        collect_start_time = time.time()
        steps_collected_this_iteration = 0
        log_episodes = 0
        log_episode_returns = []
        log_episode_forward_progress = []

        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()
            log_episodes += 1

            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None
            episode_steps = 0

            episode_obs = []
            episode_act = []
            episode_next_obs = []

            while episode_steps < max_episode_length:
                planner = self.base_planner if (use_base_only and self.base_planner is not None) else self.planner
                action = planner.plan(obs)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()

                next_obs, reward, terminated, truncated, info = self._step_env(action)
                episode_return += float(reward)

                x_position = float(self._get_forward_position(info))
                if episode_x_start is None:
                    episode_x_start = x_position
                episode_x_last = x_position

                episode_obs.append(obs)
                episode_act.append(action)
                episode_next_obs.append(next_obs)

                obs = next_obs
                episode_steps += 1
                steps_collected_this_iteration += 1

                if steps_collected_this_iteration >= steps_target:
                    break
                if terminated or truncated:
                    break

            buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            forward_progress = 0.0
            if episode_x_start is not None and episode_x_last is not None:
                forward_progress = float(episode_x_last - episode_x_start)
            log_episode_forward_progress.append(forward_progress)
            log_episode_returns.append(float(episode_return))

        collect_time = time.time() - collect_start_time
        reward_mean = float(np.mean(log_episode_returns)) if log_episode_returns else 0.0
        reward_std = float(np.std(log_episode_returns)) if log_episode_returns else 0.0
        forward_mean = float(np.mean(log_episode_forward_progress)) if log_episode_forward_progress else 0.0
        forward_std = float(np.std(log_episode_forward_progress)) if log_episode_forward_progress else 0.0

        print(
            f"Collected: steps={steps_collected_this_iteration} "
            f"episodes={log_episodes} "
            f"reward_mean={reward_mean:.3f} +- {reward_std:.3f} "
            f"forward_mean={forward_mean:.3f} +- {forward_std:.3f} "
            f"time={collect_time:.1f}s"
        )

        return buffer

    def _run_train_epoch(self, batch_size, steps_per_epoch):
        self.dynamics_model.train()
        loss_sum = 0.0

        for _ in range(steps_per_epoch):
            obs_b, act_b, next_obs_b = self.buffer.sample_transitions(batch_size, "train")
            obs_b = obs_b.to(self.device)
            act_b = act_b.to(self.device)
            next_obs_b = next_obs_b.to(self.device)
            delta_b = next_obs_b - obs_b

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.dynamics_model.loss(obs_b, act_b, delta_b)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item())

        return loss_sum / max(1, steps_per_epoch)

    @torch.no_grad()
    def _run_eval_epoch(self, eval_batch_size, eval_steps):
        if eval_batch_size <= 0 or eval_steps <= 0:
            return float("nan")

        self.dynamics_model.eval()
        loss_sum = 0.0

        for _ in range(eval_steps):
            obs_b, act_b, next_obs_b = self.buffer.sample_transitions(eval_batch_size, "eval")
            obs_b = obs_b.to(self.device)
            act_b = act_b.to(self.device)
            next_obs_b = next_obs_b.to(self.device)
            delta_b = next_obs_b - obs_b
            loss = self.dynamics_model.loss(obs_b, act_b, delta_b)
            loss_sum += float(loss.item())

        return loss_sum / eval_steps

    def _train_dynamics_for_iteration(self, train_epochs, batch_size, steps_per_epoch, eval_batch_size, eval_steps):
        for epoch in range(train_epochs):
            epoch_start = time.time()
            train_loss = self._run_train_epoch(batch_size, steps_per_epoch)
            eval_loss = self._run_eval_epoch(eval_batch_size, eval_steps)
            epoch_time = time.time() - epoch_start

            print(
                f"epoch {epoch + 1}/{train_epochs} "
                f"train={train_loss:.6f} "
                f"eval={eval_loss:.6f} "
                f"time={epoch_time:.2f}s"
            )

    def train(self):
        print("Starting Reskilling training")
        start_time = time.time()

        max_episode_length = int(self.train_config.get("max_episode_length", self.train_config.get("max_path_length", 1000)))
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"])
        bootstrap_base_only = bool(self.train_config.get("bootstrap_base_only", True))

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            use_base_only = bootstrap_base_only and iteration_index == 0
            if use_base_only:
                print("Collecting rollout data with base-only planner (bootstrap)")

            self.buffer = self._collect_env_steps(
                steps_per_iteration,
                max_episode_length,
                use_base_only=use_base_only,
            )

            num_train_transitions = int(sum(len(ep) for ep in self.buffer.train_observations))
            if num_train_transitions == 0:
                raise RuntimeError("No train transitions were collected; increase steps_per_iteration")

            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(
                {
                    "mean_obs": mean_obs,
                    "std_obs": std_obs,
                    "mean_act": mean_act,
                    "std_act": std_act,
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                }
            )

            steps_per_epoch = max(1, math.ceil(num_train_transitions / batch_size))
            num_eval_transitions = int(sum(len(ep) for ep in self.buffer.eval_observations))
            eval_batch_size = min(batch_size, num_eval_transitions) if num_eval_transitions > 0 else 0
            eval_steps = max(1, math.ceil(num_eval_transitions / max(1, eval_batch_size))) if eval_batch_size > 0 else 0

            self._train_dynamics_for_iteration(
                train_epochs,
                batch_size,
                steps_per_epoch,
                eval_batch_size,
                eval_steps,
            )

            self._reset_eval_planner()

        elapsed = int(time.time() - start_time)
        print(f"\nTraining finished. Elapsed: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")

    def _reset_eval_planner(self):
        self.planner = self._make_planner()
        self.base_planner = self._make_base_planner()

    def predict(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")
        self.dynamics_model._assert_normalization_stats()

        norm_stats = {
            "mean_obs": self.dynamics_model.mean_obs.detach().cpu(),
            "std_obs": self.dynamics_model.std_obs.detach().cpu(),
            "mean_act": self.dynamics_model.mean_act.detach().cpu(),
            "std_act": self.dynamics_model.std_act.detach().cpu(),
            "mean_delta": self.dynamics_model.mean_delta.detach().cpu(),
            "std_delta": self.dynamics_model.std_delta.detach().cpu(),
        }

        payload = {
            "state_dict": self.dynamics_model.state_dict(),
            "norm_stats": norm_stats,
        }
        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")

    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)

        normalization = checkpoint.get("norm_stats")
        if normalization is None:
            raise RuntimeError("Checkpoint is missing normalization stats.")

        self.dynamics_model.update_normalization_stats(normalization)
        self._reset_eval_planner()

        print(f"Loaded dynamics model from {model_path}")
        return self

    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.mean_obs, dm.std_obs, dm.mean_act, dm.std_act, dm.mean_delta, dm.std_delta)):
            return
        super().evaluate_checkpoint()
