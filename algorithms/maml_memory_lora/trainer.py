import os
import math
import time
import numpy as np

import torch
import torch.nn.utils as nn_utils

from algorithms.base_trainer import BaseTrainer
from algorithms.maml_memory_lora.dynamics_model import DynamicsModel
from algorithms.maml_memory_lora.memory import LoRAMemory
from algorithms.maml_memory_lora import sampler
from common.transition_buffer import TransitionBuffer
from common.planner import make_planner


class MAMLLoraMemoryTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)

        self.max_episode_length = int(self.environment_config["max_episode_length"])
        self.support_ratio = float(self.train_config.get("support_ratio", 0.5))
        self.inner_steps_k = int(self.train_config.get("inner_steps_k", 5))
        self.memory_lookup_steps = int(self.train_config.get("memory_lookup_steps", 50))
        self.use_memory = bool(self.train_config.get("use_memory", True))

        self.inner_learning_rate = float(self.train_config["inner_learning_rate"])
        self.outer_learning_rate = float(self.train_config["outer_learning_rate"])
        self.meta_batch_size = int(self.train_config["meta_batch_size"])

        self.lora_rank = int(self.train_config["lora_rank"])
        self.lora_alpha = float(self.train_config["lora_alpha"])

        self.support_window_size = max(1, int(self.support_ratio * self.max_episode_length))
        self.query_window_size = max(1, self.max_episode_length - self.support_window_size)
        self.meta_window_size = self.support_window_size + self.query_window_size

        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner = self._make_planner()
        self.buffer = self._make_buffer()
        self.memory = self._make_memory()

        # episode state
        self.episode_transitions = []
        self.adapted_parameters = None
        self.memory_match = False
        self.memory_lookup_done = False
        self._use_memory = False
        self._memory_hits = 0
        self.last_obs = None
        self.last_action = None
        self._episode_step = 0
        self._last_episode_return = None
        self._last_episode_length = None
        self._collecting = False
        self._iteration_episode_stats = []
        self._episode_pre_loss = None
        self._episode_post_loss = None
        self._episode_delta = None
        self._episode_update_norm = None
        self._episode_helpful = None
        self._episode_memory_lookup = False
        self._episode_memory_hit = False

    def _make_memory(self):
        relative_improvement = float(self.train_config["relative_improvement"])
        absolute_improvement = float(self.train_config["absolute_improvement"])
        return LoRAMemory(self.output_dir, relative_improvement, absolute_improvement)

    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)

    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None:
            raise AttributeError("Missing dynamics_model config in YAML")

        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        seed = self.train_seed
        weight_decay = float(dynamics_model_config.get("weight_decay", 0.0))
        norm_clip = float(dynamics_model_config.get("norm_clip", 10.0))
        return DynamicsModel(
            self.observation_dim,
            self.action_dim,
            hidden_sizes,
            self.outer_learning_rate,
            self.lora_rank,
            self.lora_alpha,
            seed,
            weight_decay=weight_decay,
            norm_clip=norm_clip,
        )

    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        base_env = getattr(self.env, "unwrapped", self.env)

        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        dynamics_fn = self.dynamics_model.predict_next_state_with_parameters

        return make_planner(planner_config, dynamics_fn, reward_fn, self.env.action_space, self.device, self.train_seed)

    def _evaluate(self, episodes, seeds):
        self._use_memory = bool(self.use_memory)
        self._memory_hits = 0
        try:
            metrics = super()._evaluate(episodes, seeds)
            if self._use_memory:
                print(f"memory: hits={self._memory_hits}")
            return metrics
        finally:
            self._use_memory = False

    def _compute_meta_loss(self, support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs):
        current_params = self.dynamics_model.get_parameter_dict()
        for _ in range(self.inner_steps_k):
            current_params = self.dynamics_model.compute_adapted_parameters_step(
                current_params, support_obs, support_act, support_next_obs, self.inner_learning_rate, create_graph=True
            )

        query_delta = query_next_obs - query_obs
        meta_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, current_params)
        return meta_loss

    def _outer_update(self, meta_loss):
        self.dynamics_model.optimizer.zero_grad()
        meta_loss.backward()
        nn_utils.clip_grad_norm_(self.dynamics_model.parameters(), max_norm=5.0)
        self.dynamics_model.optimizer.step()
        return meta_loss.item()

    def _evaluate_meta_batch(self, eval_batch):
        support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs = eval_batch
        eval_meta_loss = self._compute_meta_loss(support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs)
        return eval_meta_loss.item()

    def _log_epoch(self, epoch, train_loss, eval_loss, epoch_time_s, train_epochs, log_print_every_k_epochs):
        should_print = (epoch % log_print_every_k_epochs == 0) or (epoch == train_epochs - 1)
        if not should_print:
            return
        print(f"epoch {epoch}/{train_epochs}: train_meta_loss={train_loss:.6f} eval_meta_loss={eval_loss:.6f} time={epoch_time_s:.2f}s")

    def _train_dynamics_for_iteration(self, train_epochs, steps_per_epoch, eval_batch):
        log_print_every_k_epochs = 5
        for epoch in range(train_epochs):
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0

            for _ in range(steps_per_epoch):
                train_batch = sampler.sample_meta_batch(
                    self.buffer,
                    "train",
                    self.meta_batch_size,
                    self.support_window_size,
                    self.query_window_size,
                    self.device,
                )
                support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs = train_batch
                meta_loss = self._compute_meta_loss(support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs)
                train_loss_value = self._outer_update(meta_loss)
                epoch_loss_sum += train_loss_value

            avg_epoch_loss = epoch_loss_sum / steps_per_epoch
            epoch_time_s = time.time() - epoch_start_time
            eval_loss = self._evaluate_meta_batch(eval_batch)
            self._log_epoch(epoch, avg_epoch_loss, eval_loss, epoch_time_s, train_epochs, log_print_every_k_epochs)

    def train(self):
        print("Starting MAML memory LoRA training")
        start_time = time.time()
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            self.collect_steps(iteration_index, steps_per_iteration)

            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta)

            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            steps_per_epoch = max(1, math.ceil(num_train_transitions / (self.meta_batch_size * self.meta_window_size)))

            eval_batch = sampler.sample_meta_batch(
                self.buffer,
                "eval",
                self.meta_batch_size,
                self.support_window_size,
                self.query_window_size,
                self.device,
            )
            self._train_dynamics_for_iteration(train_epochs, steps_per_epoch, eval_batch)

        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")

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
        payload = {"state_dict": self.dynamics_model.state_dict(), "norm_stats": norm_stats}
        torch.save(payload, save_path)
        self.memory.save()
        print(f"Dynamics model saved to {save_path}")

    def predict(self, obs):
        if self.last_obs is not None and self.last_action is not None:
            self.episode_transitions.append((self.last_obs, self.last_action, obs))

        # memory lookup once when enough steps gathered
        if (
            self._use_memory
            and self.use_memory
            and not self.memory_match
            and not self.memory_lookup_done
            and len(self.episode_transitions) >= self.memory_lookup_steps
        ):
            self._episode_memory_lookup = True
            retrieved_lora = self.memory.retrieve(self.episode_transitions, self.dynamics_model)
            if retrieved_lora is not None:
                base_parameters = self.dynamics_model.get_parameter_dict()
                self.adapted_parameters = self._merge_lora(base_parameters, retrieved_lora)
                self.memory_match = True
                self._episode_memory_hit = True
                self._memory_hits += 1
            self.memory_lookup_done = True

        params_for_planning = self.dynamics_model.get_parameter_dict()

        # perform single adaptation after support window if no memory hit
        support_ready = len(self.episode_transitions) >= self.support_window_size
        if not self.memory_match and support_ready and self.adapted_parameters is None:
            support_obs, support_act, support_next_obs = zip(*self.episode_transitions[: self.support_window_size])
            support_obs = torch.as_tensor(np.stack(support_obs), dtype=torch.float32, device=self.device)
            support_act = torch.as_tensor(np.stack(support_act), dtype=torch.float32, device=self.device)
            support_next_obs = torch.as_tensor(np.stack(support_next_obs), dtype=torch.float32, device=self.device)

            base_params = self.dynamics_model.get_parameter_dict()
            support_delta = support_next_obs - support_obs
            with torch.no_grad():
                pre_loss = self.dynamics_model.compute_loss_with_parameters(
                    support_obs, support_act, support_delta, base_params
                )

            current_params = base_params
            for _ in range(self.inner_steps_k):
                current_params = self.dynamics_model.compute_adapted_parameters_step(
                    current_params, support_obs, support_act, support_next_obs, self.inner_learning_rate, create_graph=False
                )
            self.adapted_parameters = current_params
            with torch.no_grad():
                post_loss = self.dynamics_model.compute_loss_with_parameters(
                    support_obs, support_act, support_delta, current_params
                )

            pre_val = float(pre_loss.item())
            post_val = float(post_loss.item())
            delta_val = pre_val - post_val
            self._episode_pre_loss = pre_val
            self._episode_post_loss = post_val
            self._episode_delta = delta_val
            self._episode_helpful = delta_val > 0.0
            self._episode_update_norm = self._lora_update_norm(base_params, current_params)

        if self.adapted_parameters is not None:
            params_for_planning = self.adapted_parameters

        action = self.planner.plan(obs, parameters=params_for_planning)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.last_obs = obs
        self.last_action = action
        return action

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
            raise RuntimeError("Checkpoint is missing normalization stats. Re-train with updated save().")
        normalization = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in normalization.items()}
        self.dynamics_model.update_normalization_stats(
            normalization["mean_obs"],
            normalization["std_obs"],
            normalization["mean_act"],
            normalization["std_act"],
            normalization["mean_delta"],
            normalization["std_delta"],
        )
        print(f"Loaded dynamics model from {model_path}")
        return self

    def _reset_episode_state(self):
        if self._collecting and self._last_episode_length is not None:
            short_for_split = int(self._last_episode_length) < int(self.meta_window_size)
            self._iteration_episode_stats.append(
                {
                    "pre_loss": self._episode_pre_loss,
                    "post_loss": self._episode_post_loss,
                    "delta": self._episode_delta,
                    "helpful": self._episode_helpful,
                    "update_norm": self._episode_update_norm,
                    "short_for_split": short_for_split,
                    "memory_lookup": self._episode_memory_lookup,
                    "memory_hit": self._episode_memory_hit,
                }
            )

        if self.adapted_parameters is not None and not self.memory_match:
            lora_params = self._lora_param_dict(self.adapted_parameters)
            if lora_params:
                self.memory.insert(lora_params)
                self.memory.save()

        self.episode_transitions = []
        self.adapted_parameters = None
        self.memory_match = False
        self.memory_lookup_done = False
        self.last_obs = None
        self.last_action = None
        self._episode_step = 0
        self._last_episode_return = None
        self._last_episode_length = None
        self._episode_pre_loss = None
        self._episode_post_loss = None
        self._episode_delta = None
        self._episode_update_norm = None
        self._episode_helpful = None
        self._episode_memory_lookup = False
        self._episode_memory_hit = False

    def _rollout_episode(self, env, iteration_index, max_steps):
        obs, _ = env.reset()
        self._episode_step = 0
        episode_return = 0.0
        episode_steps = 0
        episode_obs = []
        episode_act = []
        episode_next_obs = []

        while episode_steps < max_steps:
            if iteration_index == 0:
                action = env.action_space.sample()
            else:
                action = self.predict(obs)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

            episode_obs.append(obs)
            episode_act.append(action)
            episode_next_obs.append(next_obs)

            obs = next_obs
            episode_steps += 1
            self._episode_step = episode_steps

            if terminated or truncated:
                break

        self._last_episode_return = float(episode_return)
        self._last_episode_length = int(episode_steps)
        return (episode_obs, episode_act, episode_next_obs), {"episode_return": float(episode_return)}

    def collect_steps(self, iteration_index, steps_target):
        self._collecting = True
        self._iteration_episode_stats = []
        super().collect_steps(iteration_index, steps_target)
        self._collecting = False
        self._log_iteration_summary(iteration_index)

    def _log_iteration_summary(self, iteration_index):
        if not self._iteration_episode_stats:
            return

        pre_losses = [e["pre_loss"] for e in self._iteration_episode_stats if e["pre_loss"] is not None]
        post_losses = [e["post_loss"] for e in self._iteration_episode_stats if e["post_loss"] is not None]
        deltas = [e["delta"] for e in self._iteration_episode_stats if e["delta"] is not None]
        helpful = [e["helpful"] for e in self._iteration_episode_stats if e["helpful"] is not None]
        update_norms = [e["update_norm"] for e in self._iteration_episode_stats if e["update_norm"] is not None]
        short_eps = sum(1 for e in self._iteration_episode_stats if e["short_for_split"])
        lookups = sum(1 for e in self._iteration_episode_stats if e["memory_lookup"])
        hits = sum(1 for e in self._iteration_episode_stats if e["memory_hit"])

        mean_pre = float(np.mean(pre_losses)) if pre_losses else 0.0
        mean_post = float(np.mean(post_losses)) if post_losses else 0.0
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        helpful_pct = 100.0 * float(np.mean(helpful)) if helpful else 0.0
        mean_norm = float(np.mean(update_norms)) if update_norms else 0.0
        max_norm = float(np.max(update_norms)) if update_norms else 0.0
        hit_rate = 100.0 * hits / lookups if lookups > 0 else 0.0

        print(
            f"maml_cfg: support_steps={self.support_window_size} "
            f"query_steps={self.query_window_size} support_ratio={self.support_ratio:.2f} "
            f"inner_steps_k={self.inner_steps_k} inner_lr={self.inner_learning_rate:.4f}"
        )
        print(
            f"memory: lookup_attempts={lookups} hits={hits} hit_rate={hit_rate:.1f}%"
        )
        print(
            f"adapt_stats: mean_pre_loss={mean_pre:.4f} mean_post_loss={mean_post:.4f} "
            f"mean_delta={mean_delta:.4f} helpful_pct={helpful_pct:.1f}%"
        )
        print(
            f"adapt_norm: mean_update_norm={mean_norm:.4f} max_update_norm={max_norm:.4f}"
        )
        print(f"skipped_episodes: short_for_split={short_eps}")

    def _lora_param_dict(self, params):
        if params is None:
            return None
        lora_params = {}
        for name, param in params.items():
            if "A.weight" in name or "B.weight" in name:
                lora_params[name] = param.detach().cpu()
        return lora_params

    def _merge_lora(self, base_parameters, lora_params):
        merged = dict(base_parameters)
        for name, param in lora_params.items():
            base_param = merged.get(name)
            if torch.is_tensor(param) and torch.is_tensor(base_param) and param.device != base_param.device:
                param = param.to(base_param.device)
            merged[name] = param
        return merged

    def _lora_update_norm(self, params_before, params_after):
        if params_before is None or params_after is None:
            return 0.0
        deltas = []
        for name, param in params_before.items():
            if "A.weight" in name or "B.weight" in name:
                deltas.append((params_after[name] - param).detach().reshape(-1))
        if not deltas:
            return 0.0
        return torch.norm(torch.cat(deltas)).item()
