import os
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import torch.optim as optim
import math
import time
from utils.seed import seed_env, set_seed

from algorithms.residual_adapter.dynamics_model import DynamicsModel
from algorithms.residual_adapter.residual_adapter import ResidualAdapter
from algorithms.residual_adapter.planner import RandomShootingPlanner, CrossEntropyMethodPlanner
from algorithms.residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper
from algorithms.residual_adapter.transition_buffer import TransitionBuffer


class ResidualAdapterTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.norm_stats = None
        
        self.dynamics_model = self._load_dynamics_model()
        self.residual_adapter = self._make_residual_adapter()
    
        self.residual_optimizer = self._make_residual_optimizer()
 
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper()
        self.planner = self._make_planner()
        self.buffer = self._make_buffer()
        
    def _make_residual_optimizer(self):
        residual_adapter_config = self.train_config.get("residual_adapter")
        
        if residual_adapter_config is None:
            return None
        
        learning_rate = float(residual_adapter_config.get("learning_rate"))
        return optim.Adam(self.residual_adapter.parameters(), lr=learning_rate)
        
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
        
    def _make_residual_dynamics_wrapper(self):
        return ResidualDynamicsWrapper(self.dynamics_model, self.residual_adapter)
        
    def _make_residual_adapter(self):                
        residual_adapter_config = self.train_config.get("residual_adapter")
        
        if residual_adapter_config is None:
            return None
    
        hidden_sizes = residual_adapter_config.get("hidden_sizes")
        learning_rate = float(residual_adapter_config.get("learning_rate"))
        seed = self.train_seed
        
        residual_adapter = ResidualAdapter(self.observation_dim, self.action_dim, hidden_sizes, learning_rate, seed).to(self.device)
        
        residual_adapter.update_normalization_stats(
        self.norm_stats["mean_obs"],
        self.norm_stats["std_obs"],
        self.norm_stats["mean_act"],
        self.norm_stats["std_act"],
        self.norm_stats["mean_delta"],
        self.norm_stats["std_delta"])
        
        return residual_adapter
        
    def _load_dynamics_model_state_dict(self, model, model_path):
        payload = torch.load(model_path, map_location="cpu")
        state_dict = payload["state_dict"] 
        model.load_state_dict(state_dict)
        return model
    
    def _retrieve_norm_stats(self, model_path):
        payload = torch.load(model_path, map_location="cpu")
        norm_stats = payload["norm_stats"] 
        
        norm_stats = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in norm_stats.items()
        }
        
        return norm_stats
        
    def _apply_norm_stats_to_dynamics_model(self, dynamics_model, norm_stats):
        dynamics_model.update_normalization_stats(
            norm_stats["mean_obs"],
            norm_stats["std_obs"],
            norm_stats["mean_act"],
            norm_stats["std_act"],
            norm_stats["mean_delta"],
            norm_stats["std_delta"],
        )
        return dynamics_model
        
    def _freeze_dynamics_model(self, dynamics_model):
        for p in dynamics_model.parameters():
            p.requires_grad = False
        dynamics_model.eval()
        return dynamics_model
                  
    def _make_dynamics_model_from_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        seed = config["train"]["seed"]
        hidden_sizes = config["train"]["dynamics_model"]["hidden_sizes"]
        learning_rate = config["train"]["dynamics_model"]["learning_rate"]
        observation_dim = self.observation_dim
        action_dim = self.action_dim
    
        return DynamicsModel(observation_dim, action_dim, hidden_sizes, learning_rate, seed)

    def _load_dynamics_model(self):
        dynamics_model_path = self.train_config["dynamics_model_path"]
        
        if not os.path.isdir(dynamics_model_path): raise NotADirectoryError(f"dynamics_model_path must be a directory, got: {dynamics_model_path}")
        
        model_path = os.path.join(dynamics_model_path, "model.pt")
        if not os.path.isfile(model_path): raise FileNotFoundError(f"Missing model.pt in {dynamics_model_path}")
        
        config_path = os.path.join(dynamics_model_path, "config.yaml")
        if not os.path.isfile(config_path): raise FileNotFoundError(f"Missing config.yaml in {dynamics_model_path}")
        
        dynamics_model = self._make_dynamics_model_from_config(config_path).to(self.device)
        dynamics_model = self._load_dynamics_model_state_dict(dynamics_model, model_path)
        norm_stats = self._retrieve_norm_stats(model_path)
        dynamics_model = self._apply_norm_stats_to_dynamics_model(dynamics_model, norm_stats)
        dynamics_model = self._freeze_dynamics_model(dynamics_model)
        self.norm_stats = norm_stats
        
        return dynamics_model
        
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing planner config in YAML")
        
        planner_type = planner_config.get("type")         
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))
    
        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        
        action_space = self.env.action_space
        act_low = action_space.low
        act_high = action_space.high
        
        residual_adapter_config = self.train_config.get("residual_adapter")
        dynamics_fn = self.dynamics_model.predict_next_state if residual_adapter_config is None else self.residual_dynamics_wrapper.predict_next_state
        
        if planner_type == "rs":
            return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
        
        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))
            seed = self.train_seed
                
            return CrossEntropyMethodPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount, num_cem_iters, percent_elites, alpha, seed)
            
        raise AttributeError(f"Planner type {planner_type} not supported")
  
    def _collect_steps(self):
        self.buffer.clear_that_shit()
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        data_collection_policy = self.train_config["data_collection_policy"]
        
        steps_collected_this_iteration = 0
            
        log_collect_start_time = time.time()
        log_episodes = 0
        log_episode_forward_progress = []
        log_episode_velocity = []
        log_episode_returns = []
        
        while steps_collected_this_iteration < steps_per_iteration:
            obs, _ = self.env.reset()
            log_episodes += 1
            
            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None
            episode_velocity = 0.0

            episode_steps = 0          
            episode_obs = []
            episode_act = []
            episode_next_obs = []
            
            while episode_steps < max_path_length:
                
                if data_collection_policy == "planner":
                    action = self.planner.plan(obs)
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                else:
                    action = self.env.action_space.sample()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                episode_return += float(reward)
                
                x_position = float(self._get_forward_position(info))
                if episode_x_start is None:
                    episode_x_start = x_position
                episode_x_last = x_position
                
                episode_velocity += self._get_x_velocity(info)
                
                episode_obs.append(obs)
                episode_act.append(action)
                episode_next_obs.append(next_obs)
                
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if steps_collected_this_iteration >= steps_per_iteration:
                    break
                
                if terminated or truncated:
                    break
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_velocity.append(float(episode_velocity))
            log_episode_returns.append(float(episode_return))
                    
        collect_stats = {
            "log_episodes": log_episodes,
            "log_collect_time":  time.time() - log_collect_start_time, 
            "steps_collected_this_iteration": steps_collected_this_iteration,
            "avg_reward": sum(log_episode_returns) / max(1, len(log_episode_returns)),
            "avg_forward_progress": sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress)),
            "avg_velocity": sum(log_episode_velocity) / max(1, len(log_episode_velocity)),
        }
        
        return collect_stats
  
  
    def _evaluate_dynamics_k_step(self):
        k_list=(1, 2, 5, 10, 15)
        k_max = max(k_list)
        
        episodes = self.buffer.eval_observations
        total_starts = sum(max(0, len(ep) - k_max + 1) for ep in episodes)
        if total_starts <= 0:
            print("RMSE: (skipping k-step eval; not enough eval data)")
            return
        eval_batch_size = min(5000, total_starts)
        
        obs_batch, action_batch, target_batch = self.buffer.sample_k_step_batch(k_max, eval_batch_size, "eval")
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        pred_state = obs_batch
        base_pred_state = obs_batch

    
        sum_squared_error_by_k = {k: 0.0 for k in k_list}
        base_sum_squared_error_by_k = {k: 0.0 for k in k_list}
        count_by_k = {k: 0 for k in k_list}
        for t in range(k_max):
            act_t = action_batch[:, t, :]
            pred_next_state = self.residual_dynamics_wrapper.predict_next_state(pred_state, act_t)
            base_pred_next_state = self.dynamics_model.predict_next_state(base_pred_state, act_t)
            true_next_state = target_batch[:, t, :]
            error = pred_next_state - true_next_state
            base_error = base_pred_next_state - true_next_state
            step_index = t + 1
            if step_index in k_list:
                sum_squared_error_by_k[step_index] += (error ** 2).mean().item()
                base_sum_squared_error_by_k[step_index] += (base_error ** 2).mean().item()
                count_by_k[step_index] += 1
                
            pred_state = pred_next_state
            base_pred_state = base_pred_next_state
            
        mse_by_k = {k: (sum_squared_error_by_k[k] / max(1, count_by_k[k])) for k in k_list}
        rmse_by_k = {k: math.sqrt(mse_by_k[k]) for k in k_list}
        base_mse_by_k = {k: (base_sum_squared_error_by_k[k] / max(1, count_by_k[k])) for k in k_list}
        base_rmse_by_k = {k: math.sqrt(base_mse_by_k[k]) for k in k_list}
        
        print("RMSE:       ", " | ".join([f"k-{k} {rmse_by_k[k]:.4f}" for k in k_list]))
        print("RMSE (base):", " | ".join([f"k-{k} {base_rmse_by_k[k]:.4f}" for k in k_list]))
  

    def _train_residual_adapter_for_iteration(self, train_epochs, batch_size, steps_per_epoch, eval_batch):
        eval_obs, eval_act, eval_next_obs = eval_batch
        log_print_every_k_epochs = 5
        
        for _epoch in range(train_epochs):
            epoch_start_time = time.time()
            
            epoch_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                batch_obs, batch_act, batch_next_obs = self.buffer.sample_transitions(batch_size, "train")
                
                batch_obs = batch_obs.to(self.device)
                batch_act = batch_act.to(self.device)
                batch_next_obs = batch_next_obs.to(self.device)
                
                loss_value = self._train_on_batch(batch_obs, batch_act, batch_next_obs)
    
                epoch_loss_sum += loss_value
                    
            avg_epoch_loss = epoch_loss_sum / steps_per_epoch                    
            epoch_time_s = time.time() - epoch_start_time
            should_print = (_epoch % log_print_every_k_epochs == 0) or (_epoch == train_epochs - 1)
            if should_print:
                with torch.no_grad():
                    eval_loss = self.residual_dynamics_wrapper.loss(eval_obs, eval_act, eval_next_obs).item()
                    base_pred = self.dynamics_model.predict_next_state(eval_obs, eval_act)
                    base_rmse = torch.sqrt(torch.mean((base_pred - eval_next_obs) ** 2)).item()
                    # NOTE: Training-time RMSE metrics are computed in raw next-state space for both models
                    # (comparable units); the optimization loss remains in normalized-delta space.
                    residual_pred = self.residual_dynamics_wrapper.predict_next_state(eval_obs, eval_act)
                    residual_rmse = torch.sqrt(torch.mean((residual_pred - eval_next_obs) ** 2)).item()
                    rmse_improvement = base_rmse - residual_rmse 
                    rmse_impr_pct = (rmse_improvement / (base_rmse + 1e-12)) * 100.0
                    
                print(
                    f"epoch {_epoch}/{train_epochs}: "
                    f"train={avg_epoch_loss:.6f} "
                    f"eval={eval_loss:.6f} "
                    f"rmse_impr={rmse_improvement:+.6f} ({rmse_impr_pct:+.1f}%) "
                    f"time={epoch_time_s:.2f}s"
                )
                
        self._evaluate_dynamics_k_step()

    def _train_on_batch(self, batch_obs, batch_act, batch_next_obs):
        loss = self.residual_dynamics_wrapper.loss(batch_obs, batch_act, batch_next_obs)
        self.residual_optimizer.zero_grad()
        loss.backward()
        self.residual_optimizer.step()
        return float(loss.item())

    def train(self):
        print("Starting Task Specific Task Residual Adapter training")
        

        if self.residual_adapter is None:
            print("No Residual Adapter spesifed for training")
            return
            
             
        start_time = time.time()

        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"]) 

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            
            collect_stats = self._collect_steps()
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            avg_reward = collect_stats["avg_reward"]
            avg_forward_progress = collect_stats["avg_forward_progress"]
            avg_velocity = collect_stats["avg_velocity"]
            steps_collected_this_iteration = collect_stats["steps_collected_this_iteration"]
            log_collect_time = collect_stats["log_collect_time"]
            log_episodes = collect_stats["log_episodes"]
            
            print(f"collect: dataset={num_train_transitions} " f"steps={steps_collected_this_iteration} " f"episodes={log_episodes} " f"avg_rew={avg_reward:.3f} " f"avg_fp={avg_forward_progress:.3f} " f"avg_v={avg_velocity:.3f} " f"time={log_collect_time:.1f}s")
            
            steps_per_epoch = math.ceil(num_train_transitions / batch_size)
                        
            with torch.no_grad():
                num_eval_transitions = sum(len(ep) for ep in self.buffer.eval_observations)
                eval_batch_size = min(num_eval_transitions, 50_000)
                eval_obs, eval_act, eval_next_obs = self.buffer.sample_transitions(eval_batch_size, "eval")
                eval_obs = eval_obs.to(self.device)
                eval_act = eval_act.to(self.device)
                eval_next_obs = eval_next_obs.to(self.device)
                eval_batch = (eval_obs, eval_act, eval_next_obs)
                
            self._train_residual_adapter_for_iteration(train_epochs, batch_size, steps_per_epoch, eval_batch)
            
            # remove
            with torch.no_grad():
                base_pred = self.dynamics_model.predict_next_state(eval_obs, eval_act)
                base_rmse = torch.sqrt(torch.mean((base_pred - eval_next_obs) ** 2)).item()
                residual_pred = self.residual_dynamics_wrapper.predict_next_state(eval_obs, eval_act)
                residual_rmse = torch.sqrt(torch.mean((residual_pred - eval_next_obs) ** 2)).item()
            print(f"iter {iteration_index}/{iterations}: base_rmse={base_rmse:.6f} residual_rmse={residual_rmse:.6f}")
            
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")  

    def evaluate_rmse(self, transitions, model):
        obs = np.stack([t[0] for t in transitions], axis=0)
        act = np.stack([t[1] for t in transitions], axis=0)
        next_obs = np.stack([t[2] for t in transitions], axis=0)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_next = model.predict_next_state(obs_t, act_t)
            rmse = torch.sqrt(torch.mean((pred_next - next_obs_t) ** 2))

        return float(rmse.item())

    def evaluate(self):
        print("Overwriting base evaluate to predict model error")
        
        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]
        max_steps = int(self.train_config.get("max_path_length"))
        data_collection_policy = self.train_config["data_collection_policy"]
        
        all_rmses = []
        base_all_rmses = []
        all_episode_sequences = []
        
        for seed in seeds:
            set_seed(seed)
            eval_env = self._make_eval_env(seed=seed)
            transitions = [] 
            seed_sequences = []
            
            for episode in range(episodes):
                obs, _ = eval_env.reset(seed=seed + episode)
                
                done = False
                steps = 0
                episode_transitions = []
                while not done and steps < max_steps:                   
                    if data_collection_policy == "planner":
                        act = self.planner.plan(obs)
                        if torch.is_tensor(act):
                            act = act.detach().cpu().numpy()
                    else:
                        act = eval_env.action_space.sample()
                        
                    next_obs, reward, terminated, truncated, info = eval_env.step(act)

                    transitions.append((obs, act, next_obs))
                    episode_transitions.append((obs, act, next_obs))

                    obs = next_obs
                    done = terminated or truncated
                    steps += 1

                seed_sequences.append(episode_transitions)
            rmse = self.evaluate_rmse(transitions, self.residual_dynamics_wrapper)
            base_rmse = self.evaluate_rmse(transitions, self.dynamics_model)
            all_rmses.append(rmse)
            base_all_rmses.append(base_rmse)
            all_episode_sequences.extend(seed_sequences)
            
        mean_rmse = float(np.mean(all_rmses)) if all_rmses else float("nan")
        std_rmse = float(np.std(all_rmses)) if all_rmses else float("nan")
        base_mean_rmse = float(np.mean(base_all_rmses)) if base_all_rmses else float("nan")
        base_std_rmse = float(np.std(base_all_rmses)) if base_all_rmses else float("nan")
        rmse_improvement = base_mean_rmse - mean_rmse
        rmse_improvement_pct = (rmse_improvement / (base_mean_rmse + 1e-12)) * 100.0
        
        
        print("\n=== eval completed: one-step RMSE (all seeds) ===")
        print(f"RMSE improvement (base - residual): {rmse_improvement:+.6f} ({rmse_improvement_pct:+.1f}%)")
        
        
        print(f"{'model':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
        print("-" * 54)

        print(f"{'base':<12} {base_mean_rmse:>10.6f} {base_std_rmse:>10.6f} {float(np.min(base_all_rmses)):>10.6f} {float(np.max(base_all_rmses)):>10.6f}")
        print(f"{'residual':<12} {mean_rmse:>10.6f} {std_rmse:>10.6f} {float(np.min(all_rmses)):>10.6f} {float(np.max(all_rmses)):>10.6f}")
        
        k_horizon = 15
        base_k_metrics = self._compute_k_step_errors(all_episode_sequences, self.dynamics_model, k_horizon)
        residual_k_metrics = self._compute_k_step_errors(all_episode_sequences, self.residual_dynamics_wrapper, k_horizon)

        print(f"\n=== eval completed: open-loop K-step RMSE (horizon={k_horizon}) ===")
        print(f"{'model':<12} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'final':>10}")
        print("-" * 66)
        print(
            f"{'base':<12} {base_k_metrics[0]:>10.6f} {base_k_metrics[1]:>10.6f} "
            f"{base_k_metrics[2]:>10.6f} {base_k_metrics[3]:>10.6f} {base_k_metrics[4]:>10.6f}"
        )
        print(
            f"{'residual':<12} {residual_k_metrics[0]:>10.6f} {residual_k_metrics[1]:>10.6f} "
            f"{residual_k_metrics[2]:>10.6f} {residual_k_metrics[3]:>10.6f} {residual_k_metrics[4]:>10.6f}"
        )
        
        print("\nAlso running BaseTrainer.evaluate() for reward / progress stats")
        super().evaluate()
                
    def _compute_k_step_errors(self, episode_sequences, model, horizon):
        step_errors = []
        final_errors = []
        for episode in episode_sequences:
            if not episode:
                continue
            obs = episode[0][0]
            steps = min(len(episode), horizon)
            for idx in range(steps):
                action = episode[idx][1]
                true_next = episode[idx][2]
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    pred_next = model.predict_next_state(obs_t, act_t)
                pred_next_np = pred_next.detach().cpu().numpy()
                error = np.linalg.norm(pred_next_np - true_next)
                step_errors.append(error)
                if idx == steps - 1:
                    final_errors.append(error)
                obs = pred_next_np

        if not step_errors:
            nan = float("nan")
            return nan, nan, nan, nan, nan

        step_errors = np.asarray(step_errors, dtype=np.float32)
        final_errors = np.asarray(final_errors, dtype=np.float32) if final_errors else np.array([], dtype=np.float32)
        final_mean = float(np.mean(final_errors)) if final_errors.size else float("nan")
        return (
            float(np.mean(step_errors)),
            float(np.std(step_errors)),
            float(np.min(step_errors)),
            float(np.max(step_errors)),
            final_mean,
        )

    def predict(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action
        
    def save(self):
        if self.residual_adapter is None:
            return

        payload = { "residual_adapter_state": self.residual_adapter.state_dict()}
        if self.residual_optimizer is not None:
            payload["optimizer_state"] = self.residual_optimizer.state_dict()

        save_path = os.path.join(self.output_dir, "residual_adapter.pt")
        torch.save(payload, save_path)
        print(f"Residual adapter checkpoint written to {save_path}")

    def load(self, path):
        if os.path.isdir(path):
            checkpoint_path = os.path.join(path, "residual_adapter.pt")
        else:
            checkpoint_path = path

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No residual adapter checkpoint at {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location=self.device)
        if "residual_adapter_state" not in payload:
            raise KeyError("Checkpoint missing 'residual_adapter_state'")

        self.residual_adapter.load_state_dict(payload["residual_adapter_state"])
        if self.residual_optimizer is not None and "optimizer_state" in payload:
            self.residual_optimizer.load_state_dict(payload["optimizer_state"])

        print(f"Residual adapter checkpoint loaded from {checkpoint_path}")
        return self

 
   
