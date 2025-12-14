import os
import torch
from algorithms.base_trainer import BaseTrainer

import time
import numpy as np
from collections import deque


from algorithms.grbal.dynamics_model import DynamicsModel
from algorithms.grbal.planner import CrossEntropyMethodPlanner, MPPIPlanner, RandomShootingPlanner

from utils.seed import set_seed

class GrBALTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        """
        Skeleton trainer for a faithful PyTorch reimplementation of Nagabandi et al.'s GrBAL loop.

        This mirrors learning_to_adapt.trainers.mb_trainer.Trainer: build env -> dynamics model
        -> MPC planner, then run one random warmup iteration followed by repeated
        sample-then-fit iterations. The actual rollout/fit logic lives in train(), but the
        wiring here must stay compatible with BaseTrainer so that _step_env triggers evals.
        """
        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing planner config in YAML")
        self.planner =self._make_planner(planner_config)
        self.eval_planner = None

        # Evaluation-time adaptation state (used by predict, not by train)
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None

        self.num_rollouts = int(self.train_config["num_rollouts"])
        self.max_path_length = int(self.train_config["max_path_length"])
        self.steps_per_iter = self.num_rollouts * self.max_path_length
        
    def _make_dynamics_model(self):
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_sizes = self.train_config.get("hidden_sizes")
        learning_rate = float(self.train_config.get("learning_rate"))
        train_epochs = int(self.train_config.get("train_epochs"))
        valid_split_ratio = float(self.train_config.get("valid_split_ratio"))
        meta_batch_size = int(self.train_config.get("meta_batch_size"))
        adapt_batch_size = int(self.train_config.get("adapt_batch_size"))
        inner_lr = float(self.train_config.get("inner_lr"))
        inner_steps = int(self.train_config.get("inner_steps"))
        rolling_average_persitency = float(self.train_config.get("rolling_average_persitency"))
        seed = self.train_seed
    
        return DynamicsModel(observation_dim, action_dim, hidden_sizes, learning_rate, train_epochs, valid_split_ratio, meta_batch_size, adapt_batch_size, inner_lr, inner_steps, rolling_average_persitency, seed)

    def _make_planner(self, planner_config):        
        planner_type = planner_config.get("type")         
        horizon = None if (v := planner_config.get("horizon")) is None else int(v)
        n_candidates = None if (v := planner_config.get("n_candidates")) is None else int(v)
        discount = None if (v := planner_config.get("discount")) is None else float(v)

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        
        action_space = self.env.action_space
        act_low = action_space.low
        act_high = action_space.high
        
        if planner_type == "cem":
            num_cem_iters = None if (v := planner_config.get("num_cem_iters")) is None else int(v)
            percent_elites = None if (v := planner_config.get("percent_elites")) is None else float(v)
            alpha = None if (v := planner_config.get("alpha")) is None else float(v)
            
            return CrossEntropyMethodPlanner(
                dynamics_fn=self.dynamics_model.predict_next_state_with_parameters,
                reward_fn=reward_fn,
                horizon=horizon,
                n_candidates=n_candidates,
                act_low=act_low,
                act_high=act_high,
                discount=discount,
                seed=self.train_seed,
                num_cem_iters = num_cem_iters,
                percent_elites=percent_elites,
                alpha=alpha,
                device = self.device
            )
            
        if planner_type == "rs":
            return RandomShootingPlanner(
                dynamics_fn=self.dynamics_model.predict_next_state_with_parameters,
                reward_fn=reward_fn,
                horizon=horizon,
                n_candidates=n_candidates,
                act_low=act_low,
                act_high=act_high,
                discount=discount,
                seed=self.train_seed,
                device = self.device
            )
            
        if planner_type == "mppi":            
            mppi_lambda = None if (v := planner_config.get("mppi_lambda")) is None else float(v) 
            mppi_sigma = None if (v := planner_config.get("mppi_sigma")) is None else float(v)
            warm_start = None if (v := planner_config.get("warm_start")) is None else bool(v)
 
            return MPPIPlanner(
                dynamics_fn=self.dynamics_model.predict_next_state_with_parameters,
                reward_fn=reward_fn,
                horizon=horizon,
                n_candidates=n_candidates,
                act_low=act_low,
                act_high=act_high,
                device = self.device,
                discount=discount,
                lambda_=mppi_lambda,
                sigma=mppi_sigma,
                warm_start = warm_start,
                seed=self.train_seed
            )
            
        raise AttributeError(f"Planner type {planner_type} not supported")
    
    def _stack_and_tensorize(self, obs_paths, act_paths, next_obs_paths):
        """Pad to common length, stack to 3D (num_paths, path_len, dim), then convert to torch."""
        if len(obs_paths) == 0:
            return None, None, None
        max_len = max(path.shape[0] for path in obs_paths)
        obs_dim = obs_paths[0].shape[1]
        act_dim = act_paths[0].shape[1]

        obs_batch = np.zeros((len(obs_paths), max_len, obs_dim), dtype=np.float32)
        act_batch = np.zeros((len(act_paths), max_len, act_dim), dtype=np.float32)
        next_obs_batch = np.zeros((len(next_obs_paths), max_len, obs_dim), dtype=np.float32)

        for idx, (o, a, no) in enumerate(zip(obs_paths, act_paths, next_obs_paths)):
            T = o.shape[0]
            obs_batch[idx, :T] = o
            act_batch[idx, :T] = a
            next_obs_batch[idx, :T] = no

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_tensor = torch.as_tensor(act_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        return obs_tensor, act_tensor, next_obs_tensor
    
    def train(self):
        print("Starting GrBAL training")
        start_time = time.time()
        
        
        gradient_descent_steps = 0
        
    
        # === Setup ===
        train_iterations = int(self.train_config["train_iterations"])
        adapt_batch_size = int(self.train_config["adapt_batch_size"])

        iteration_idx = 0
        total_steps_collected = 0
        
        warmup_obs_paths = []
        warmup_act_paths = []
        warmup_next_obs_paths = []
        
        collected_steps = 0
        while collected_steps < self.steps_per_iter:
            
            episode_obs = []
            episode_act = []
            episode_next_obs = []

            obs, _ = self.env.reset()
            
            for _ in range(self.max_path_length):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                executed_action = info.get("executed_action", action)

                episode_obs.append(obs)
                episode_act.append(executed_action)
                episode_next_obs.append(next_obs)

                obs = next_obs
                collected_steps += 1
                total_steps_collected += 1

                if terminated or truncated or collected_steps >= self.steps_per_iter:
                    break
            
            if len(episode_obs) > 0:
                warmup_obs_paths.append(np.array(episode_obs))
                warmup_act_paths.append(np.array(episode_act))
                warmup_next_obs_paths.append(np.array(episode_next_obs))

        warmup_obs_tensor, warmup_act_tensor, warmup_next_obs_tensor = self._stack_and_tensorize(
            warmup_obs_paths, warmup_act_paths, warmup_next_obs_paths
        )
        if warmup_obs_tensor is  None:
            raise RuntimeError("No data collected this iteration; iter_obs_tensor is None.")
        
        fit_stats = self.dynamics_model.fit(warmup_obs_tensor, warmup_act_tensor, warmup_next_obs_tensor)
        
        gradient_descent_steps += fit_stats['gradient_descent_steps']
        
        print(
            f"Finished iteration {iteration_idx}/{train_iterations - 1} | "
            f"epochs_trained={fit_stats['epochs_trained']} "
            f"train_loss={fit_stats['train_loss']:.4f} "
            f"val_loss={fit_stats['val_loss']:.4f} "
            f"GD_steps={fit_stats['gradient_descent_steps']} ")
            
        iteration_idx = 1
            
        # === Main training iterations (iteration_idx >= 1) ===
        while iteration_idx < train_iterations:
            iteration_start_time = time.time()
            
            steps_this_iter = 0
            iter_obs_paths = []
            iter_act_paths = []
            iter_next_obs_paths = []
            iter_returns = []

            while steps_this_iter < self.steps_per_iter:
                episode_obs = []
                episode_act = []
                episode_next_obs = []
                episode_return = 0.0
                adapt_window = deque(maxlen=adapt_batch_size)

                obs, _ = self.env.reset()

                for _ in range(self.max_path_length):
                    params_for_planning = self.dynamics_model.get_parameter_dict()
                    if len(adapt_window) >= adapt_batch_size:
                        window_obs, window_act, window_next_obs = zip(*adapt_window)
                        support_obs = np.stack(window_obs, axis=0)
                        support_act = np.stack(window_act, axis=0)
                        support_next_obs = np.stack(window_next_obs, axis=0)

                        theta_prime = self.dynamics_model.compute_adapted_params(support_obs, support_act, support_next_obs)
                        params_for_planning = theta_prime

                    action = self.planner.plan(obs, parameters=params_for_planning)
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                    next_obs, reward, terminated, truncated, info = self._step_env(action)
                    executed_action = info.get("executed_action", action)

                    episode_obs.append(obs)
                    episode_act.append(executed_action)
                    episode_next_obs.append(next_obs)
                    episode_return += reward
                    adapt_window.append((obs, executed_action, next_obs))

                    obs = next_obs

                    steps_this_iter += 1
                    total_steps_collected += 1

                    if terminated or truncated or steps_this_iter >= self.steps_per_iter:
                        break
                    
                if len(episode_obs) > 0:
                    iter_obs_paths.append(np.array(episode_obs))
                    iter_act_paths.append(np.array(episode_act))
                    iter_next_obs_paths.append(np.array(episode_next_obs))
                    iter_returns.append(episode_return)

            iter_obs_tensor, iter_act_tensor, iter_next_obs_tensor = self._stack_and_tensorize(
                iter_obs_paths, iter_act_paths, iter_next_obs_paths
            )
            if iter_obs_tensor is None:
                raise RuntimeError("No data collected this iteration; iter_obs_tensor is None.")

            fit_stats = self.dynamics_model.fit(iter_obs_tensor, iter_act_tensor, iter_next_obs_tensor)
            gradient_descent_steps += fit_stats['gradient_descent_steps']
            elapsed = time.time() - iteration_start_time
            
            print(
                f"Finished iteration {iteration_idx}/{train_iterations - 1} | "
                f"epochs_trained={fit_stats['epochs_trained']} "
                f"train_loss={fit_stats['train_loss']:.4f} "
                f"val_loss={fit_stats['val_loss']:.4f} "
                f"mean_return={np.mean(iter_returns):.2f} "
                f"GD_steps={fit_stats['gradient_descent_steps']} "
                f"in {elapsed:.2f}s"
            )

            iteration_idx += 1
            
        print("Total gradient descent steps :", gradient_descent_steps)
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
        
        # Training loop complete; metrics and logging are handled by BaseTrainer.

    def predict(self, obs):
        """
        Evaluation-time action selection mirroring the training-time online adaptation + MPC.
        Called by BaseTrainer._evaluate() inside rollout loops; must NOT write to buffers or run fit().
        """
        
        if self.eval_planner is None:
            planner_config = self.eval_config.get("planner")
            
            if planner_config is not None:
                self.eval_planner = self._make_planner(planner_config)
            else:
                self.eval_planner = self._make_planner(self.train_config.get("planner"))
                
        adapt_batch_size = int(self.train_config["adapt_batch_size"])
        if self.eval_adapt_window is None:
            self.eval_adapt_window = deque(maxlen=adapt_batch_size)
            self.eval_last_obs = None
            self.eval_last_action = None

        # If we have a previous (obs, action), treat (last_obs, last_action, current_obs) as a transition.
        if self.eval_last_obs is not None and self.eval_last_action is not None:
            self.eval_adapt_window.append((self.eval_last_obs, self.eval_last_action, obs))

        params_for_planning = self.dynamics_model.get_parameter_dict()
        if len(self.eval_adapt_window) >= adapt_batch_size:
            window_obs, window_act, window_next_obs = zip(*self.eval_adapt_window)
            support_obs = np.stack(window_obs, axis=0)
            support_act = np.stack(window_act, axis=0)
            support_next_obs = np.stack(window_next_obs, axis=0)

            theta_prime = self.dynamics_model.compute_adapted_params(
                support_obs,
                support_act,
                support_next_obs,
            )
            params_for_planning = theta_prime

        action = self.eval_planner.plan(obs, parameters=params_for_planning)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        # Remember current (obs, action) for the next predict() call so we can form a transition.
        self.eval_last_obs = obs
        self.eval_last_action = action

        return action
           
    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")

        # Grab weights and normalization tensors
        payload = {"state_dict": self.dynamics_model.state_dict()}
        norm_stats = {
            "observations_mean": self.dynamics_model.observations_mean,
            "observations_std": self.dynamics_model.observations_std,
            "actions_mean": self.dynamics_model.actions_mean,
            "actions_std": self.dynamics_model.actions_std,
            "delta_mean": self.dynamics_model.delta_mean,
            "delta_std": self.dynamics_model.delta_std,
        }

        if any(v is None for v in norm_stats.values()):
            raise RuntimeError("Normalization stats are missing; train the dynamics model before saving.")

        # Detach to CPU tensors for portability
        payload["normalization"] = {k: v.detach().cpu() for k, v in norm_stats.items()}

        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")
        
    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        # Restore model weights
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)

        # Restore normalization stats required for planning/adaptation
        normalization = checkpoint.get("normalization")
        if normalization is None:
            raise RuntimeError(
                "Checkpoint is missing normalization stats. Re-train with the updated save() so the stats are stored."
            )

        normalization = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in normalization.items()
        }
        self.dynamics_model.observations_mean = normalization["observations_mean"]
        self.dynamics_model.observations_std = normalization["observations_std"]
        self.dynamics_model.actions_mean = normalization["actions_mean"]
        self.dynamics_model.actions_std = normalization["actions_std"]
        self.dynamics_model.delta_mean = normalization["delta_mean"]
        self.dynamics_model.delta_std = normalization["delta_std"]

        print(f"Loaded dynamics model from {model_path}")
        return self
    
    def _reset_eval_adaptation(self):
        """Optional hook for trainers that keep eval-time adaptation state."""
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
        self.eval_planner = None
        
    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.observations_mean, dm.observations_std, dm.actions_mean, dm.actions_std, dm.delta_mean, dm.delta_std)):
            return
        super().evaluate_checkpoint()
