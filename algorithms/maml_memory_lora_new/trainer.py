import os
import math
import time
import numpy as np

import torch
import torch.nn.utils as nn_utils

from algorithms.base_trainer import BaseTrainer
from algorithms.maml_memory_lora_new.dynamics_model import DynamicsModel
from algorithms.maml_memory_lora_new.memory import LoRAMemory
from algorithms.maml_memory_lora_new import sampler
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

        schedule = self.train_config.get("inner_learning_rate_schedule")
        if schedule is None:
            inner_lr_scalar = self.train_config.get("inner_learning_rate")
            if inner_lr_scalar is None:
                raise AttributeError("Missing inner_learning_rate_schedule in YAML")
            schedule = [float(inner_lr_scalar)] * self.inner_steps_k
            print(
                "cfg_compat: inner_learning_rate_schedule missing; "
                f"using scalar inner_learning_rate={float(inner_lr_scalar):.6f} "
                f"for all {self.inner_steps_k} inner steps"
            )
        if not isinstance(schedule, (list, tuple)) or len(schedule) == 0:
            raise ValueError("inner_learning_rate_schedule must be a non-empty list")
        self.inner_learning_rate_schedule = [float(lr) for lr in schedule]
        if len(self.inner_learning_rate_schedule) != self.inner_steps_k:
            raise ValueError(
                f"inner_learning_rate_schedule length ({len(self.inner_learning_rate_schedule)}) "
                f"must match inner_steps_k ({self.inner_steps_k})"
            )
        if any((not math.isfinite(lr)) or lr <= 0.0 for lr in self.inner_learning_rate_schedule):
            raise ValueError("inner_learning_rate_schedule values must be finite and > 0")
        self.outer_learning_rate = float(self.train_config["outer_learning_rate"])
        self.meta_batch_size = int(self.train_config["meta_batch_size"])

        self.lora_rank = int(self.train_config["lora_rank"])
        self.lora_alpha = float(self.train_config["lora_alpha"])
        self.iterations = int(self.train_config["iterations"])
        self.train_epochs = int(self.train_config["train_epochs"])

        self.support_window_size = max(1, int(self.support_ratio * self.max_episode_length))
        self.query_window_size = max(1, self.max_episode_length - self.support_window_size)
        self.meta_window_size = self.support_window_size + self.query_window_size

        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self._da_warmup_ratio = 0.4
        self._cosine_eta_min_ratio = 0.1
        self._msl_weight_bias = 0.7
        self._total_train_epochs = max(1, self.iterations * self.train_epochs)
        self._da_warmup_epochs = max(1, int(round(self._total_train_epochs * self._da_warmup_ratio)))
        self._global_epoch_index = 0
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dynamics_model.optimizer,
            T_max=self._total_train_epochs,
            eta_min=self.outer_learning_rate * self._cosine_eta_min_ratio,
        )
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

        # iteration-level logging state
        self._collect_reward_history = []
        self._collect_reward_std_history = []
        self._last_collect_reward_mean = None
        self._last_collect_reward_std = None
        self._last_collect_episode_count = 0
        self._outer_clip_step_count_iteration = 0
        self._outer_nonfinite_grad_step_count_iteration = 0
        self._nan_meta_loss_step_count_iteration = 0
        self._outer_update_steps_iteration = 0
        self._best_eval_meta_loss_iteration = None
        self._last_eval_meta_loss_iteration = None
        self._patience_counter_iteration = 0
        self._last_msl_step_query_losses_iteration = None

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

    def _current_outer_lr(self):
        return float(self.dynamics_model.optimizer.param_groups[0]["lr"])

    def _use_second_order(self, global_epoch_index):
        return bool(global_epoch_index >= self._da_warmup_epochs)

    def _msl_step_weights(self, global_epoch_index):
        if self.inner_steps_k <= 1:
            return [1.0]

        progress = min(1.0, max(0.0, float(global_epoch_index) / float(max(1, self._total_train_epochs - 1))))
        uniform_weight = 1.0 / float(self.inner_steps_k)

        raw = []
        for step_index in range(self.inner_steps_k):
            step_position = float(step_index + 1) / float(self.inner_steps_k)
            emphasis = 1.0 + self._msl_weight_bias * progress * step_position
            raw.append((1.0 - progress) * uniform_weight + progress * emphasis)

        total = float(sum(raw))
        if total <= 0.0:
            return [uniform_weight] * self.inner_steps_k
        return [w / total for w in raw]

    def _compute_meta_loss(
        self,
        support_obs,
        support_act,
        support_next_obs,
        query_obs,
        query_act,
        query_next_obs,
        create_graph,
        step_weights,
        return_step_losses=False,
    ):
        current_params = self.dynamics_model.get_parameter_dict()
        query_delta = query_next_obs - query_obs
        query_step_losses = []

        for inner_lr in self.inner_learning_rate_schedule:
            current_params = self.dynamics_model.compute_adapted_parameters_step(
                current_params, support_obs, support_act, support_next_obs, inner_lr, create_graph=create_graph
            )
            step_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, current_params)
            query_step_losses.append(step_loss)

        if len(step_weights) != len(query_step_losses):
            raise RuntimeError("MSL step weight count does not match number of inner steps")

        meta_loss = 0.0
        for weight, step_loss in zip(step_weights, query_step_losses):
            meta_loss = meta_loss + float(weight) * step_loss

        if return_step_losses:
            step_losses_values = [float(loss.detach().item()) for loss in query_step_losses]
            return meta_loss, step_losses_values
        return meta_loss

    def _outer_update(self, meta_loss):
        self.dynamics_model.optimizer.zero_grad()
        meta_loss_value = float(meta_loss.item())
        if not math.isfinite(meta_loss_value):
            self._nan_meta_loss_step_count_iteration += 1
        meta_loss.backward()
        grad_norm = nn_utils.clip_grad_norm_(self.dynamics_model.parameters(), max_norm=5.0)
        grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        if not math.isfinite(grad_norm_value):
            self._outer_nonfinite_grad_step_count_iteration += 1
        elif grad_norm_value > 5.0:
            self._outer_clip_step_count_iteration += 1
        self._outer_update_steps_iteration += 1
        self.dynamics_model.optimizer.step()
        return meta_loss_value

    def _evaluate_meta_batch(self, eval_batch, step_weights):
        support_obs, support_act, support_next_obs, query_obs, query_act, query_next_obs = eval_batch
        eval_meta_loss = self._compute_meta_loss(
            support_obs,
            support_act,
            support_next_obs,
            query_obs,
            query_act,
            query_next_obs,
            create_graph=False,
            step_weights=step_weights,
            return_step_losses=False,
        )
        return eval_meta_loss.item()

    def _log_epoch(
        self,
        epoch,
        train_loss,
        eval_loss,
        best_eval_loss_so_far,
        patience_counter,
        epoch_time_s,
        train_epochs,
        log_print_every_k_epochs,
        outer_lr,
        diff_order_tag,
        msl_w_first,
        msl_w_last,
    ):
        should_print = (epoch % log_print_every_k_epochs == 0) or (epoch == train_epochs - 1)
        if not should_print:
            return
        print(
            f"epoch {epoch}/{train_epochs}: train_meta_loss={train_loss:.6f} "
            f"eval_meta_loss={eval_loss:.6f} best_eval_meta_loss={best_eval_loss_so_far:.6f} "
            f"patience={patience_counter} time={epoch_time_s:.2f}s "
            f"outer_lr={outer_lr:.6f} diff_order={diff_order_tag} "
            f"msl_w_first={msl_w_first:.4f} msl_w_last={msl_w_last:.4f}"
        )

    def _train_dynamics_for_iteration(self, train_epochs, steps_per_epoch, eval_batch):
        log_print_every_k_epochs = 5
        best_eval_loss_so_far = float("inf")
        patience_counter = 0
        last_eval_loss = float("nan")
        self._outer_clip_step_count_iteration = 0
        self._outer_nonfinite_grad_step_count_iteration = 0
        self._nan_meta_loss_step_count_iteration = 0
        self._outer_update_steps_iteration = 0
        self._last_msl_step_query_losses_iteration = None

        for epoch in range(train_epochs):
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0
            global_epoch_index = self._global_epoch_index
            create_graph = self._use_second_order(global_epoch_index)
            diff_order_tag = "so" if create_graph else "fo"
            msl_weights = self._msl_step_weights(global_epoch_index)
            msl_step_loss_sum = [0.0] * self.inner_steps_k

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
                meta_loss, step_query_losses = self._compute_meta_loss(
                    support_obs,
                    support_act,
                    support_next_obs,
                    query_obs,
                    query_act,
                    query_next_obs,
                    create_graph=create_graph,
                    step_weights=msl_weights,
                    return_step_losses=True,
                )
                train_loss_value = self._outer_update(meta_loss)
                epoch_loss_sum += train_loss_value
                for step_index, step_loss in enumerate(step_query_losses):
                    msl_step_loss_sum[step_index] += step_loss

            avg_epoch_loss = epoch_loss_sum / steps_per_epoch
            epoch_time_s = time.time() - epoch_start_time
            avg_step_query_losses = [value / steps_per_epoch for value in msl_step_loss_sum]
            self._last_msl_step_query_losses_iteration = avg_step_query_losses
            eval_loss = self._evaluate_meta_batch(eval_batch, step_weights=msl_weights)
            last_eval_loss = eval_loss
            if eval_loss < best_eval_loss_so_far:
                best_eval_loss_so_far = eval_loss
                patience_counter = 0
            else:
                patience_counter += 1
            outer_lr = self._current_outer_lr()
            self._log_epoch(
                epoch,
                avg_epoch_loss,
                eval_loss,
                best_eval_loss_so_far,
                patience_counter,
                epoch_time_s,
                train_epochs,
                log_print_every_k_epochs,
                outer_lr,
                diff_order_tag,
                msl_weights[0],
                msl_weights[-1],
            )
            self.lr_scheduler.step()
            self._global_epoch_index += 1

        self._best_eval_meta_loss_iteration = float(best_eval_loss_so_far)
        self._last_eval_meta_loss_iteration = float(last_eval_loss)
        self._patience_counter_iteration = int(patience_counter)

    def train(self):
        print("Starting MAML memory LoRA training")
        print(
            f"run_cfg: lora_rank={self.lora_rank} lora_alpha={self.lora_alpha:.4f} "
            f"support_ratio={self.support_ratio:.2f} inner_steps_k={self.inner_steps_k} "
            f"outer_lr={self.outer_learning_rate:.6f} meta_batch_size={self.meta_batch_size} "
            f"inner_lr_schedule={self.inner_learning_rate_schedule} "
            f"random_windows=1 msl=1 da=1 cosine_lr=1 "
            f"da_warmup_ratio={self._da_warmup_ratio:.2f} "
            f"cosine_eta_min_ratio={self._cosine_eta_min_ratio:.2f}"
        )
        start_time = time.time()
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = self.iterations
        train_epochs = self.train_epochs

        for iteration_index in range(iterations):
            iteration_start_time = time.time()
            iteration_start_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(iteration_start_time))
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            print(f"iter_start: index={iteration_index} ts={iteration_start_stamp}")
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
            self._log_training_iteration_health(iteration_index)

            iteration_end_time = time.time()
            iteration_end_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(iteration_end_time))
            print(
                f"iter_end: index={iteration_index} ts={iteration_end_stamp} "
                f"duration_s={iteration_end_time - iteration_start_time:.2f}"
            )

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
            for inner_lr in self.inner_learning_rate_schedule:
                current_params = self.dynamics_model.compute_adapted_parameters_step(
                    current_params, support_obs, support_act, support_next_obs, inner_lr, create_graph=False
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
                    "episode_return": self._last_episode_return,
                    "episode_length": self._last_episode_length,
                }
            )

        if self.use_memory and self.adapted_parameters is not None and not self.memory_match:
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
        episode_returns = [float(e["episode_return"]) for e in self._iteration_episode_stats if e["episode_return"] is not None]
        short_eps = sum(1 for e in self._iteration_episode_stats if e["short_for_split"])
        lookups = sum(1 for e in self._iteration_episode_stats if e["memory_lookup"])
        hits = sum(1 for e in self._iteration_episode_stats if e["memory_hit"])

        mean_pre = float(np.mean(pre_losses)) if pre_losses else 0.0
        mean_post = float(np.mean(post_losses)) if post_losses else 0.0
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        helpful_pct = 100.0 * float(np.mean(helpful)) if helpful else 0.0
        non_helpful_pct = 100.0 - helpful_pct if helpful else 0.0
        mean_norm = float(np.mean(update_norms)) if update_norms else 0.0
        p25_delta = float(np.percentile(deltas, 25)) if deltas else 0.0
        p50_delta = float(np.percentile(deltas, 50)) if deltas else 0.0
        p75_delta = float(np.percentile(deltas, 75)) if deltas else 0.0
        p25_norm = float(np.percentile(update_norms, 25)) if update_norms else 0.0
        p50_norm = float(np.percentile(update_norms, 50)) if update_norms else 0.0
        p75_norm = float(np.percentile(update_norms, 75)) if update_norms else 0.0
        max_norm = float(np.max(update_norms)) if update_norms else 0.0
        hit_rate = 100.0 * hits / lookups if lookups > 0 else 0.0
        reward_mean = float(np.mean(episode_returns)) if episode_returns else 0.0
        reward_std = float(np.std(episode_returns)) if episode_returns else 0.0

        self._last_collect_reward_mean = reward_mean
        self._last_collect_reward_std = reward_std
        self._last_collect_episode_count = len(episode_returns)
        self._collect_reward_history.append(reward_mean)
        self._collect_reward_std_history.append(reward_std)
        ma5 = float(np.mean(self._collect_reward_history[-5:])) if self._collect_reward_history else 0.0
        ma10 = float(np.mean(self._collect_reward_history[-10:])) if self._collect_reward_history else 0.0

        print(
            f"maml_cfg: support_steps={self.support_window_size} "
            f"query_steps={self.query_window_size} support_ratio={self.support_ratio:.2f} "
            f"inner_steps_k={self.inner_steps_k}"
        )
        print(
            f"collect_stats: iteration={iteration_index} episodes={len(episode_returns)} "
            f"episode_reward_mean={reward_mean:.4f} episode_reward_std={reward_std:.4f} "
            f"reward_ma5={ma5:.4f} reward_ma10={ma10:.4f}"
        )
        print(
            f"memory: lookup_attempts={lookups} hits={hits} hit_rate={hit_rate:.1f}%"
        )
        print(
            f"adapt_stats: mean_pre_loss={mean_pre:.4f} mean_post_loss={mean_post:.4f} "
            f"mean_delta={mean_delta:.4f} delta_p25={p25_delta:.4f} delta_p50={p50_delta:.4f} "
            f"delta_p75={p75_delta:.4f} helpful_pct={helpful_pct:.1f}% non_helpful_pct={non_helpful_pct:.1f}%"
        )
        print(
            f"adapt_norm: mean_update_norm={mean_norm:.4f} p25_update_norm={p25_norm:.4f} "
            f"p50_update_norm={p50_norm:.4f} p75_update_norm={p75_norm:.4f} max_update_norm={max_norm:.4f}"
        )
        print(f"skipped_episodes: short_for_split={short_eps}")

    def _log_training_iteration_health(self, iteration_index):
        clip_rate = (
            100.0 * self._outer_clip_step_count_iteration / self._outer_update_steps_iteration
            if self._outer_update_steps_iteration > 0
            else 0.0
        )
        print(
            f"train_health: iteration={iteration_index} outer_updates={self._outer_update_steps_iteration} "
            f"clipped_grad_steps={self._outer_clip_step_count_iteration} clipped_grad_pct={clip_rate:.2f} "
            f"nonfinite_grad_steps={self._outer_nonfinite_grad_step_count_iteration} "
            f"nan_meta_loss_steps={self._nan_meta_loss_step_count_iteration}"
        )
        if self._best_eval_meta_loss_iteration is not None and self._last_eval_meta_loss_iteration is not None:
            print(
                f"train_meta_summary: iteration={iteration_index} "
                f"best_eval_meta_loss={self._best_eval_meta_loss_iteration:.6f} "
                f"last_eval_meta_loss={self._last_eval_meta_loss_iteration:.6f} "
                f"patience_counter={self._patience_counter_iteration}"
            )
        if self._last_msl_step_query_losses_iteration:
            step_means = self._last_msl_step_query_losses_iteration
            step_means_str = ",".join(f"{value:.5f}" for value in step_means)
            delta_final_first = float(step_means[-1] - step_means[0]) if len(step_means) >= 2 else 0.0
            print(
                f"msl_diag: iteration={iteration_index} "
                f"mean_query_losses_by_step=[{step_means_str}] "
                f"final_step_minus_first_step={delta_final_first:.6f}"
            )

    def evaluate(self):
        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]

        total_runs = len(seeds) * episodes
        print(f"Evaluating {episodes} episode(s) × {len(seeds)} seed(s) = {total_runs} total runs")

        metrics = self._evaluate(episodes, seeds)
        elapsed = metrics["elapsed"]
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"

        print("\nEvaluation summary:")
        print(f"- reward_mean: {metrics['reward_mean']:.4f}")
        print(f"- reward_std: {metrics['reward_std']:.4f}")
        print(f"- episode_length_mean: {metrics['episode_length_mean']:.2f}")
        print(f"- elapsed: {elapsed_str}")

        if self._last_collect_reward_mean is not None:
            gap = float(self._last_collect_reward_mean) - float(metrics["reward_mean"])
            print(
                f"eval_gap: train_reward_mean_last_collect={self._last_collect_reward_mean:.4f} "
                f"eval_reward_mean={float(metrics['reward_mean']):.4f} gap={gap:.4f}"
            )

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
