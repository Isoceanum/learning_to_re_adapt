import os

import numpy as np
from algorithms.base_trainer import BaseTrainer
import torch
import math
import time

from common.dynamics_model import DynamicsModel
from common.task_specific_helpers import (
    load_pretrained_dynamics_model,
    eval_policy_rollout,
    compute_cross_task_rmse,
    eval_epoch_rmse,
    load_dataset,
    make_planner_from_base_config,
)

class TaskSpecificBiasTermAdaptationTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.base_dynamics_model = load_pretrained_dynamics_model(config_path, model_path, obs_dim, action_dim, self.train_seed, self.device)
        self.dynamics_model = load_pretrained_dynamics_model(config_path, model_path, obs_dim, action_dim, self.train_seed, self.device)

        self._enable_bias_training(self.dynamics_model)
        self.optimizer = self._make_bitfit_optimizer()
        dataset_path = self.train_config.get("dataset_path")
        
        eval_split = float(self.train_config.get("eval_split"))
        batch_size = int(self.train_config.get("batch_size"))
        
        self.train_dataloader, self.eval_dataloader = load_dataset(dataset_path, eval_split, batch_size, self.train_seed)
        
        
        pretrained_cfg_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        self.planner = make_planner_from_base_config(pretrained_cfg_path, self.env, self.dynamics_model.predict_next_state, self.device, self.train_seed)

        total_params = sum(p.numel() for p in self.dynamics_model.parameters())
        trainable_params = sum(p.numel() for p in self.dynamics_model.parameters() if p.requires_grad)
        trainable_percent = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

        self.adaptation_cost = {
            "trainable_params": trainable_params,
            "trainable_percent": trainable_percent,
            "gradient_steps": 0,
            "learning_rate": float(self.train_config.get("learning_rate")),
            "train_dataset_size": len(self.train_dataloader.dataset),
            "batch_size": int(self.train_config.get("batch_size")),
        }

    def _enable_bias_training(self, model):
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(".bias"))
        model.train()

    def _bias_parameters(self):
        return [p for name, p in self.dynamics_model.named_parameters() if name.endswith(".bias")]

    def _make_bitfit_optimizer(self):
        lr = float(self.train_config.get("learning_rate"))
        bias_params = self._bias_parameters()
        if len(bias_params) == 0:
            raise RuntimeError("No bias parameters found for BitFit training.")
        return torch.optim.Adam(bias_params, lr=lr)
        
    
    def make_dynamics_model(self, dynamics_model_config, obs_dim, action_dim, seed):
        if dynamics_model_config is None: 
            raise AttributeError("Missing dynamics_model config in YAML")
        
        hidden_sizes = dynamics_model_config["train"]["dynamics_model"]["hidden_sizes"]
        learning_rate = float(dynamics_model_config["train"]["dynamics_model"]["learning_rate"])

        return DynamicsModel(obs_dim, action_dim, hidden_sizes, learning_rate, seed)
        

    def train(self):
        print("Starting TS-BitFit offline training")
        start_time = time.time()
        epochs = int(self.train_config.get("epochs"))
        
        eval_policy_rollout_flagg = self.eval_config.get("eval_policy_rollout")

        train_steps = len(self.train_dataloader.dataset)
        eval_steps = len(self.eval_dataloader.dataset)
        dataset_path = self.train_config.get("dataset_path", "")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0] if dataset_path else "unknown"
        print(f"dataset[{dataset_name}]: train={train_steps} eval={eval_steps}")

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

            print(
                f"[epoch {epoch_index+1}/{epochs}] "
                f"train_rmse={train_rmse:.6f} "
                f"eval_rmse={eval_rmse:.6f} "
                f"base_eval_rmse={base_eval_rmse:.6f}"
            )

            if eval_policy_rollout_flagg:
                eval_policy_rollout(self)

        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nadaptation_cost={self.adaptation_cost}")
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
           
    def save(self):
        save_path = os.path.join(self.output_dir, "bitfit_model.pt")
        payload = {
            "bitfit_model_state": self.dynamics_model.state_dict(),
        }
        if self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()
        torch.save(payload, save_path)
        print(f"BitFit model saved to {save_path}")

    def evaluate(self):
        print("Per-dim RMSE on eval split (base vs BitFit)")
        compute_cross_task_rmse(self)
        

    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "bitfit_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No BitFit checkpoint found at {model_path}")

        # Load base model (pretrained, frozen)
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.dynamics_model = load_pretrained_dynamics_model(
            config_path, model_path, obs_dim, action_dim, self.train_seed, self.device
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("bitfit_model_state", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)
        self._enable_bias_training(self.dynamics_model)

        self.optimizer = self._make_bitfit_optimizer()
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Refresh planner to use adapted model
        pretrained_cfg_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        self.planner = make_planner_from_base_config(
            pretrained_cfg_path,
            self.env,
            self.dynamics_model.predict_next_state,
            self.device,
            self.train_seed,
        )

        print(f"Loaded BitFit model from {model_path}")
        return self
    
    def predict(self, obs):
        action = self.planner.plan(obs)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action
    
