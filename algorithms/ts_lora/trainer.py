import os

import numpy as np
from algorithms.base_trainer import BaseTrainer
import yaml
import torch
import math
import time

from algorithms.ts_lora.dynamics_model import DynamicsModel
from algorithms.ts_lora.planner import RandomShootingPlanner, CrossEntropyMethodPlanner
from algorithms.ts_lora.transition_buffer import TransitionBuffer
from algorithms.ts_lora.lora_linear import LoRALinear



class TaskSpecificLowRankAdaptation(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        
        self.base_dynamics_model = self.load_pretrained_dynamics_model() 
        self.base_planner = self._make_planner(self.base_dynamics_model.predict_next_state)
        

        self.dynamics_model = self.load_pretrained_dynamics_model() # load a pretrained dynamics model
        self._swap_linear_layers_with_lora()
        self.optimizer = self._make_lora_optimizer()
        self.planner =self._make_planner(self.dynamics_model.predict_next_state)
        self.buffer = self._make_buffer()

    def _make_lora_optimizer(self):
        lora_lr = float(self.train_config.get("lora_lr"))
        
        lora_params = []
        for module in self.dynamics_model.model:
            if isinstance(module, LoRALinear):
                module.freeze_base()
                lora_params.extend(module.lora_parameters())
                
        return torch.optim.Adam(lora_params, lr=lora_lr)
        
    def _swap_linear_layers_with_lora(self):
        enable_low_rank_adaptation = self.train_config.get("enable_low_rank_adaptation", False)
        if not enable_low_rank_adaptation:
            return
        
        lora_rank = int(self.train_config.get("lora_rank"))
        lora_alpha = float(self.train_config.get("lora_alpha"))
        
        new_layers = []
        for layer in self.dynamics_model.model:
            if isinstance(layer, torch.nn.Linear):
                new_layers.append(LoRALinear(in_features=layer.in_features, out_features=layer.out_features, r=lora_rank, alpha=lora_alpha, bias=(layer.bias is not None), base_linear=layer))
            else:
                new_layers.append(layer)
                
        self.dynamics_model.model = torch.nn.Sequential(*new_layers)
        
    def make_dynamics_model(self, dynamics_model_config, obs_dim, action_dim, seed):
        if dynamics_model_config is None: 
            raise AttributeError("Missing dynamics_model config in YAML")
        
        hidden_sizes = dynamics_model_config["train"]["dynamics_model"]["hidden_sizes"]
        learning_rate = float(dynamics_model_config["train"]["dynamics_model"]["learning_rate"])

        return DynamicsModel(obs_dim, action_dim, hidden_sizes, learning_rate, seed)
        
    def load_pretrained_dynamics_model(self):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        
        with open(config_path, "r") as f:
            pretrained_dynamics_model_config = yaml.safe_load(f)
                        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        pretrained_dynamics_model = self.make_dynamics_model(pretrained_dynamics_model_config, obs_dim, action_dim, self.train_seed).to(self.device)
        pretrained_dynamics_model.load_saved_model(model_path)
        pretrained_dynamics_model.freeze()
        return pretrained_dynamics_model
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)

    def _make_planner(self, predict_next_state_fn):
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
        
        if planner_type == "rs":
            return RandomShootingPlanner(predict_next_state_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
        
        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))        
            return CrossEntropyMethodPlanner(predict_next_state_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount, num_cem_iters, percent_elites, alpha)
        
            
        raise AttributeError(f"Planner type {planner_type} not supported")
    
    def _collect_steps(self, iteration_index, steps_target, max_path_length):
        steps_collected_this_iteration = 0
            
        log_collect_start_time = time.time()
        log_episodes = 0
        log_episode_forward_progress = []
        log_episode_velocity = []
        log_episode_returns = []
        log_episodes_terminated = 0
        log_episodes_maxlen = 0
        
        while steps_collected_this_iteration < steps_target:
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
                if iteration_index == 0:
                    action = self.base_planner.plan(obs)
                else:
                    action = self.planner.plan(obs)
                    
                    
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                    
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
                            
                if steps_collected_this_iteration >= steps_target:
                    break
                
                if terminated or truncated:
                    break
            if terminated or truncated:
                log_episodes_terminated += 1
            elif episode_steps >= max_path_length:
                log_episodes_maxlen += 1
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_velocity.append(float(episode_velocity))
            log_episode_returns.append(float(episode_return))
                    
        collect_stats = {
            "log_episodes": log_episodes,
            "log_collect_time":  time.time() - log_collect_start_time, 
            "steps_collected_this_iteration": steps_collected_this_iteration,
            "avg_reward": sum(log_episode_returns) / max(1, len(log_episode_returns)),
            "reward_mean": float(np.mean(log_episode_returns)) if len(log_episode_returns) > 0 else 0.0,
            "reward_std": float(np.std(log_episode_returns)) if len(log_episode_returns) > 0 else 0.0,
            "forward_mean": float(np.mean(log_episode_forward_progress)) if len(log_episode_forward_progress) > 0 else 0.0,
            "forward_std": float(np.std(log_episode_forward_progress)) if len(log_episode_forward_progress) > 0 else 0.0,
            "avg_forward_progress": sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress)),
            "avg_velocity": sum(log_episode_velocity) / max(1, len(log_episode_velocity)),
            "episodes_terminated": log_episodes_terminated,
            "episodes_maxlen": log_episodes_maxlen,
        }
        
        return collect_stats
    
    def _evaluate_dynamics_k_step(self, dynamics_model, label):
        k_list=(1, 2, 5, 10, 15)
        k_max = max(k_list)
        
        episodes = self.buffer.eval_observations
        total_starts = sum(max(0, len(ep) - k_max + 1) for ep in episodes)
        eval_batch_size = min(5000, total_starts)
        
        obs_batch, action_batch, target_batch = self.buffer.sample_k_step_batch(k_max, eval_batch_size, "eval")
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        pred_state = obs_batch

    
        sum_squared_error_by_k = {k: 0.0 for k in k_list}
        count_by_k = {k: 0 for k in k_list}
        for t in range(k_max):
            act_t = action_batch[:, t, :]
            pred_next_state = dynamics_model.predict_next_state(pred_state, act_t)
            true_next_state = target_batch[:, t, :]
            error = pred_next_state - true_next_state
            step_index = t + 1
            if step_index in k_list:
                sum_squared_error_by_k[step_index] += (error ** 2).mean().item()
                count_by_k[step_index] += 1
                
            pred_state = pred_next_state
            
        mse_by_k = {k: (sum_squared_error_by_k[k] / max(1, count_by_k[k])) for k in k_list}
        rmse_by_k = {k: math.sqrt(mse_by_k[k]) for k in k_list}
        
        print(f"RMSE[{label}]:", " | ".join([f"k-{k} {rmse_by_k[k]:.4f}" for k in k_list]))

    def _evaluate_one_step_rmse(self, dynamics_model):
        obs_batch, action_batch, target_batch = self.buffer.sample_k_step_batch(1, 1024, "eval")
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        with torch.no_grad():
            pred_next = dynamics_model.predict_next_state(obs_batch, action_batch[:, 0, :])
            true_next = target_batch[:, 0, :]
            rmse = torch.sqrt(torch.mean((pred_next - true_next) ** 2)).item()
        return float(rmse)

    def _print_data_collection_stats(self, collect_stats):
        # compute dataset size (train split) for logging
        num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations) 
        # unpack collection logs
        avg_reward = collect_stats["avg_reward"]
        reward_mean = collect_stats["reward_mean"]
        reward_std = collect_stats["reward_std"]
        forward_mean = collect_stats["forward_mean"]
        forward_std = collect_stats["forward_std"]
        avg_forward_progress = collect_stats["avg_forward_progress"]
        avg_velocity = collect_stats["avg_velocity"]
        steps_collected_this_iteration = collect_stats["steps_collected_this_iteration"]
        log_collect_time = collect_stats["log_collect_time"]
        log_episodes = collect_stats["log_episodes"]
        episodes_terminated = collect_stats["episodes_terminated"]
        episodes_maxlen = collect_stats["episodes_maxlen"]
        
        # print collection summary for this iteration
        print(
            "Collected: "
            f"steps={steps_collected_this_iteration} "
            f"reward_mean={reward_mean:.3f} ± {reward_std:.3f} "
            f"forward_mean={forward_mean:.3f} ± {forward_std:.3f} "
            f"episodes={log_episodes} "
            f"term={episodes_terminated} "
            f"max={episodes_maxlen} "
            f"time={log_collect_time:.1f}s "

        )
        
    def train(self):
        print("Starting TS-LoRA training")
        start_time = time.time()
        
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"])
        
        enable_low_rank_adaptation = self.train_config.get("enable_low_rank_adaptation", False)
        
        
        if not enable_low_rank_adaptation:
            print("low rank adaptation disabled, training skipped" )
            return 
        

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            collect_stats = self._collect_steps(iteration_index, steps_per_iteration, max_path_length)
            # print collection stats
            self._print_data_collection_stats(collect_stats)
            self.dynamics_model.train()
            
            epoch_losses = []
            for epoch_index in range(train_epochs):
                self.optimizer.zero_grad()
                obs_batch, act_batch, next_obs_batch = self.buffer.sample_transitions(batch_size, "train")
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                next_obs_batch = next_obs_batch.to(self.device)
                
                pred_next_obs_batch = self.dynamics_model.predict_next_state(obs_batch, act_batch)
                loss = torch.mean((pred_next_obs_batch - next_obs_batch) ** 2)
                
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                print(f"[epoch {epoch_index+1}/{train_epochs}] train_mse={loss.item():.6f}")
                
            mean_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            print(f"train_lora_mse={mean_epoch_loss:.6f}")
            
            base_k1_rmse = self._evaluate_one_step_rmse(self.base_dynamics_model)
            lora_k1_rmse = self._evaluate_one_step_rmse(self.dynamics_model)
            print(f"eval_k1_rmse base={base_k1_rmse:.4f} lora={lora_k1_rmse:.4f}")

            self._evaluate_dynamics_k_step(self.base_dynamics_model, "base")
            self._evaluate_dynamics_k_step(self.dynamics_model, "lora")
 
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
           
    def save(self):
        save_path = os.path.join(self.output_dir, "lora_adapters.pt")
        lora_state = {}
        for idx, module in enumerate(self.dynamics_model.model):
            if isinstance(module, LoRALinear):
                lora_state[f"layer_{idx}"] = module.get_lora_state_dict()

        payload = {
            "lora_state": lora_state,
        }
        torch.save(payload, save_path)
        print(f"LoRA adapters saved to {save_path}")
    
    def predict(self, obs):
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

    def load(self, path):
        adapter_path = path
        if os.path.isdir(adapter_path):
            adapter_path = os.path.join(adapter_path, "lora_adapters.pt")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"No LoRA adapter checkpoint found at {adapter_path}")

        # Load base model (pretrained, frozen)
        self.dynamics_model = self.load_pretrained_dynamics_model()

        # Ensure LoRA layers are present
        self._swap_linear_layers_with_lora()

        payload = torch.load(adapter_path, map_location=self.device)
        lora_state = payload.get("lora_state", {})

        for idx, module in enumerate(self.dynamics_model.model):
            if isinstance(module, LoRALinear):
                key = f"layer_{idx}"
                if key in lora_state:
                    module.load_lora_state_dict(lora_state[key])

        # Refresh planner to use adapted model
        self.planner = self._make_planner(self.dynamics_model.predict_next_state)

        print(f"Loaded LoRA adapters from {adapter_path}")
        return self
     
