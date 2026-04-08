import os

import numpy as np
from algorithms.base_trainer import BaseTrainer
import torch
import math
import time

from common.dynamics_model import DynamicsModel
from algorithms.ts_lora.lora_linear import LoRALinear
from common.task_specific_helpers import (
    load_pretrained_dynamics_model,
    eval_policy_rollout,
    compute_cross_task_rmse,
    eval_epoch_rmse,
    eval_epoch_rmse_per_dim,
    eval_horizon_rmse,
    load_dataset,
    make_planner_from_base_config,
)

class TaskSpecificLowRankAdaptation(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.base_dynamics_model = load_pretrained_dynamics_model(config_path, model_path, obs_dim, action_dim, self.train_seed, self.device)
        self.dynamics_model = load_pretrained_dynamics_model(config_path, model_path, obs_dim, action_dim, self.train_seed, self.device)
        
        self._swap_linear_layers_with_lora()
        self.optimizer = self._make_lora_optimizer()
        dataset_path = self.train_config.get("dataset_path")
        
        eval_split = float(self.train_config.get("eval_split"))
        batch_size = int(self.train_config.get("batch_size"))
        
        self.train_dataloader, self.eval_dataloader = load_dataset(dataset_path, eval_split, batch_size, self.train_seed)
        
        pretrained_cfg_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        self.planner = make_planner_from_base_config(pretrained_cfg_path, self.env, self.dynamics_model.predict_next_state, self.device, self.train_seed)



        base_total_params = sum(p.numel() for p in self.base_dynamics_model.parameters())
        total_params = sum(p.numel() for p in self.dynamics_model.parameters())
        trainable_params = sum(p.numel() for p in self.dynamics_model.parameters() if p.requires_grad)
        trainable_percent = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        self.adaptation_cost = {
            "total_params": base_total_params,
            "trainable_params": trainable_params,
            "trainable_percent": trainable_percent,
            "gradient_steps": 0,
            "learning_rate": float(self.train_config.get("learning_rate")),
            "train_dataset_size": len(self.train_dataloader.dataset),
            "batch_size": int(self.train_config.get("batch_size")),
        }

        # Track eval reward means when eval_policy_rollout is enabled
        self.eval_reward_means = []

    def _make_lora_optimizer(self):
        lora_lr = float(self.train_config.get("learning_rate"))
        
        lora_params = []
        for module in self.dynamics_model.model:
            if isinstance(module, LoRALinear):
                module.freeze_base()
                lora_params.extend(module.lora_parameters())
                
        return torch.optim.Adam(lora_params, lr=lora_lr)
        
    def _swap_linear_layers_with_lora(self):
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
        

    def train(self):
        print("Starting TS-LoRA offline training")
        start_time = time.time()
        epochs = int(self.train_config.get("epochs"))
        
        eval_policy_rollout_flagg = self.eval_config.get("eval_policy_rollout")

        train_steps = len(self.train_dataloader.dataset)
        eval_steps = len(self.eval_dataloader.dataset)
        dataset_path = self.train_config.get("dataset_path", "")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0] if dataset_path else "unknown"
        print(f"dataset[{dataset_name}]: train={train_steps} eval={eval_steps}")
        
        if eval_policy_rollout_flagg:
            metrics = eval_policy_rollout(self)
            self.eval_reward_means.append(metrics["reward_mean"])

        for epoch_index in range(epochs):
            
            self.dynamics_model.train()
            train_sum_mse = 0.0
            train_count = 0

            for obs_batch, act_batch, next_obs_batch in self.train_dataloader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                next_obs_batch = next_obs_batch.to(self.device)

                self.optimizer.zero_grad()
                pred_next_obs_batch = self.dynamics_model.predict_next_state(obs_batch, act_batch)
                loss = torch.mean((pred_next_obs_batch - next_obs_batch) ** 2)
                loss.backward()
                self.optimizer.step()
                self.adaptation_cost["gradient_steps"] += 1
                batch_size = obs_batch.shape[0]
                train_sum_mse += float(loss.item()) * batch_size
                train_count += batch_size

            train_mse = (train_sum_mse / train_count) if train_count > 0 else float("nan")
            train_rmse = math.sqrt(train_mse) if train_count > 0 else float("nan")
            eval_rmse, base_eval_rmse = eval_epoch_rmse(self)

            print(f"[epoch {epoch_index+1}/{epochs}] train_rmse={train_rmse:.6f} eval_rmse={eval_rmse:.6f} base_eval_rmse={base_eval_rmse:.6f}")

            if eval_policy_rollout_flagg:
                metrics = eval_policy_rollout(self)
                self.eval_reward_means.append(metrics["reward_mean"])

        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        if self.eval_reward_means:
            values = ",".join(["lora"] + [f"{v:.6f}" for v in self.eval_reward_means])
            print(values)
        print(f"\nadaptation_cost={self.adaptation_cost}")

        base_per_dim, adapt_per_dim = eval_epoch_rmse_per_dim(self)
        if base_per_dim is not None and adapt_per_dim is not None:
            print("per_dim_rmse")
            print("idx | base | adapt")
            print("******************")
            for idx, (base_val, adapt_val) in enumerate(zip(base_per_dim, adapt_per_dim)):
                print(f"{idx:>3} | {base_val:.4f} | {adapt_val:.4f}")

        horizons = [1, 2, 5, 10, 15]
        horizon_rmse = eval_horizon_rmse(self, horizons)
        print("horizon_rmse")
        print("h | base | adapt")
        print("****************")
        for h in horizons:
            base_rmse, adapt_rmse = horizon_rmse.get(h, (float('nan'), float('nan')))
            print(f"{h:>2} | {base_rmse:.4f} | {adapt_rmse:.4f}")
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

    def evaluate(self):
        print("Per-dim RMSE on eval split (base vs LoRA)")
        compute_cross_task_rmse(self)
        

    def load(self, path):
        adapter_path = path
        if os.path.isdir(adapter_path):
            adapter_path = os.path.join(adapter_path, "lora_adapters.pt")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"No LoRA adapter checkpoint found at {adapter_path}")

        # Load base model (pretrained, frozen)
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.dynamics_model = load_pretrained_dynamics_model(
            config_path, model_path, obs_dim, action_dim, self.train_seed, self.device
        )

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
        self.planner = self._make_planner()

        print(f"Loaded LoRA adapters from {adapter_path}")
        return self
    
    def predict(self, obs):
        action = self.planner.plan(obs)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action
    
