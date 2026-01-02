import copy
import os
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import torch.optim as optim
import time
from utils.seed import set_seed

from algorithms.simple_residual_adapter.dynamics_model import DynamicsModel
from algorithms.simple_residual_adapter.residual_adapter import ResidualAdapter
from algorithms.simple_residual_adapter.planner import RandomShootingPlanner
from algorithms.simple_residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper
from algorithms.simple_residual_adapter.transition_buffer import TransitionBuffer

class SimpleResidualAdapterTrainer(BaseTrainer):
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
        if self.residual_adapter is None:
            return None
        
        learning_rate = float(self.train_config["lr"])
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
        seed = self.train_seed
        residual_adapter = ResidualAdapter(self.observation_dim, self.action_dim, hidden_sizes, seed).to(self.device)
        residual_adapter.update_normalization_stats(self.norm_stats["mean_obs"],self.norm_stats["std_obs"],self.norm_stats["mean_act"],self.norm_stats["std_act"],self.norm_stats["mean_delta"],self.norm_stats["std_delta"])
        return residual_adapter
        
    def _load_dynamics_model(self):
        dynamics_model_path = self.train_config["dynamics_model_path"]
        if not os.path.isdir(dynamics_model_path): raise NotADirectoryError(f"dynamics_model_path must be a directory, got: {dynamics_model_path}")
        model_path = os.path.join(dynamics_model_path, "model.pt")
        if not os.path.isfile(model_path): raise FileNotFoundError(f"Missing model.pt in {dynamics_model_path}")
        config_path = os.path.join(dynamics_model_path, "config.yaml")
        if not os.path.isfile(config_path): raise FileNotFoundError(f"Missing config.yaml in {dynamics_model_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        payload = torch.load(model_path, map_location="cpu")
        norm_stats = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in payload["norm_stats"].items()
        }

        train_config = config["train"]
        dynamics_config = train_config["dynamics_model"]
        dynamics_model = DynamicsModel(self.observation_dim, self.action_dim, dynamics_config["hidden_sizes"], dynamics_config["learning_rate"], train_config["seed"]).to(self.device)

        dynamics_model.load_state_dict(payload["state_dict"])
        dynamics_model.update_normalization_stats( norm_stats["mean_obs"], norm_stats["std_obs"], norm_stats["mean_act"], norm_stats["std_act"], norm_stats["mean_delta"], norm_stats["std_delta"])
        for p in dynamics_model.parameters():
            p.requires_grad = False
        dynamics_model.eval()

        self.norm_stats = norm_stats
        return dynamics_model
        
    def _make_planner(self):
        dynamics_model_path = self.train_config["dynamics_model_path"]
                 
        config_path = os.path.join(dynamics_model_path, "config.yaml")
        if not os.path.isfile(config_path): raise FileNotFoundError(f"Missing config.yaml in {dynamics_model_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        planner_config = config["train"]["planner"]
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
            return RandomShootingPlanner(self.residual_dynamics_wrapper.predict_next_state, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
            
        raise AttributeError(f"Planner type {planner_type} not supported")
  
    def _collect_training_data(self):
        max_path_length = int(self.train_config["max_path_length"])
        real_env_steps = int(self.train_config["real_env_steps"])
        data_collection_policy = self.train_config["data_collection_policy"]
        
        steps_collected = 0 
        log_collect_start_time = time.time()
        log_episodes = 0

        while steps_collected < real_env_steps:
            obs, _ = self.env.reset()
            log_episodes += 1
            
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
                episode_obs.append(obs)
                episode_act.append(action)
                episode_next_obs.append(next_obs)
                obs = next_obs
                episode_steps += 1
                steps_collected += 1
                            
                if steps_collected >= real_env_steps or terminated or truncated:
                    break
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
         
        log_collect_time = time.time() - log_collect_start_time
        
        print(f"collect: steps={steps_collected} episodes={log_episodes} time={log_collect_time:.1f}s")

        
    def train(self):
        print("Starting Task Specific Task Residual Adapter training")
    
        if self.residual_adapter is None:
            print("No Residual Adapter spesifed for training")
            return
            
             
        start_time = time.time()
        self._collect_training_data()

        batch_size = int(self.train_config["batch_size"])
        max_epochs = int(self.train_config["max_epochs"])
        patience = int(self.train_config["patience"])
        target_nominal_rmse = float(self.train_config["target_nominal_rmse"])
        target_margin_pct = float(self.train_config["target_margin_pct"])
        rmse_threshold = target_nominal_rmse * (1.0 + target_margin_pct / 100.0)
        val_obs_full, val_act_full, val_next_obs_full = self.buffer.get_all_transitions("eval")
        val_obs_full = val_obs_full.to(self.device)
        val_act_full = val_act_full.to(self.device)
        val_next_obs_full = val_next_obs_full.to(self.device)

        best_val_rmse = float("inf")
        best_state = None
        epochs_since_improve = 0

        for epoch in range(1, max_epochs + 1):
            obs, act, next_obs = self.buffer.sample_transitions(batch_size, split="train")
            obs = obs.to(self.device)
            act = act.to(self.device)
            next_obs = next_obs.to(self.device)

            self.residual_adapter.train()
            train_loss = self.residual_dynamics_wrapper.loss(obs, act, next_obs)
            self.residual_optimizer.zero_grad()
            train_loss.backward()
            self.residual_optimizer.step()

            with torch.no_grad():
                val_pred_next = self.residual_dynamics_wrapper.predict_next_state(val_obs_full, val_act_full)
                val_rmse = torch.sqrt(torch.mean((val_pred_next - val_next_obs_full) ** 2)).item()

            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse = val_rmse
                epochs_since_improve = 0
                best_state = {
                    "adapter": copy.deepcopy(self.residual_adapter.state_dict()),
                    "optimizer": copy.deepcopy(self.residual_optimizer.state_dict()) if self.residual_optimizer else None,
                }
            else:
                epochs_since_improve += 1

            print(f"Epoch {epoch:03d}: train_loss={float(train_loss.item()):.6f} val_rmse={val_rmse:.6f} best_val_rmse={best_val_rmse:.6f}")

            if best_val_rmse <= rmse_threshold:
                print(f"Early stopping: validation RMSE {best_val_rmse:.6f} reached threshold {rmse_threshold:.6f}")
                break

            if epochs_since_improve >= patience:
                print(f"Early stopping: no val improvement for {patience} epoch(s)")
                break

        if best_state is not None:
            self.residual_adapter.load_state_dict(best_state["adapter"])
            if self.residual_optimizer is not None and best_state["optimizer"] is not None:
                self.residual_optimizer.load_state_dict(best_state["optimizer"])


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
        
        k_horizon = 30
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

 
   
