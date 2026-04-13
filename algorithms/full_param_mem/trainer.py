from collections import deque, OrderedDict
import os

import numpy as np
from algorithms.base_trainer import BaseTrainer

import torch
import math
import time
from collections import deque

from algorithms.full_param_mem.dynamics_model import DynamicsModel
from algorithms.full_param_mem.memory import FullParamMemory
from common.transition_buffer import TransitionBuffer
from algorithms.full_param_mem import sampler
from common.planner import make_planner

class FullParamMemoryTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
    
        self.support_window_size = int(self.train_config["support_window_size"])
        self.query_window_size = int(self.train_config["query_window_size"])
        
        self.inner_learning_rate = float(self.train_config["inner_learning_rate"]) # step size for the inner (adaptation) gradient update
        self.use_online_adaptation = self.train_config["use_online_adaptation"] # enable/disable online adaptation during planning/eval

        # Eval-only memory (configured in YAML; required keys, no defaults).
        self.use_memory = bool(self.train_config["use_memory"])
        self.abs_improvement_threshold = float(self.train_config["abs_improvement_threshold"])
        self.slow_inner_learning_rate = float(self.train_config["slow_inner_learning_rate"])
        
        self.outer_learning_rate = float(self.train_config["outer_learning_rate"]) 
        self.meta_batch_size = int(self.train_config["meta_batch_size"])
        
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner = self._make_planner()
        self.buffer = self._make_buffer()
        self.memory = self._make_memory()
        
        self.support_window = None
        self.last_obs = None
        self.last_action = None
        
        self._eval_mode = False
        self._memory_selected_this_episode = False
        self._episode_memory_id = None
        self._episode_params = None
        
        

    def _evaluate(self, episodes, seeds):
        self._eval_mode = bool(self.use_memory)
        try:
            return super()._evaluate(episodes, seeds)
        finally:
            self._eval_mode = False
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)

    def _make_memory(self):
        return FullParamMemory(self.output_dir, self.abs_improvement_threshold)
    
    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None:
            raise AttributeError("Missing dynamics_model config in YAML")
    
        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        seed = self.train_seed
        
        return DynamicsModel(self.observation_dim, self.action_dim, hidden_sizes, self.outer_learning_rate, seed)
    
    
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        base_env = getattr(self.env, "unwrapped", self.env)
        
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        dynamics_fn = self.dynamics_model.predict_next_state_with_parameters

        return make_planner(planner_config, dynamics_fn, reward_fn, self.env.action_space, self.device, self.train_seed)

    def _compute_meta_loss(self, support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations, inner_learning_rate):
        # compute target delta
        query_delta = query_next_observations - query_observations
        
        meta_loss_terms = []
        # iterate over each window
        for batch_index in range(support_observations.shape[0]):
            
            # unpack support and query window
            support_obs_task = support_observations[batch_index]
            support_act_task = support_actions[batch_index]
            support_next_obs_task = support_next_observations[batch_index]

            query_obs_task = query_observations[batch_index]
            query_act_task = query_actions[batch_index]
            query_delta_task = query_delta[batch_index]

            # adapt model parameters on support window
            adapted_parameters = self.dynamics_model.compute_adapted_parameters(
                support_obs_task,
                support_act_task,
                support_next_obs_task,
                inner_learning_rate,
                create_graph=True,
            )

            # evaluate adapted parameters on query window
            query_loss_task = self.dynamics_model.compute_loss_with_parameters(query_obs_task, query_act_task, query_delta_task, adapted_parameters)
            meta_loss_terms.append(query_loss_task)
            
        # combined window loss into one scalar meta-loss
        meta_loss = torch.stack(meta_loss_terms).mean()  
        return meta_loss
 
    def _outer_update(self, meta_loss):
        # clears old gradients
        self.dynamics_model.optimizer.zero_grad()
        
        # backpropagate meta-loss through the inner adaptation step
        meta_loss.backward()
        
        # apply the outer update to the parameters
        self.dynamics_model.optimizer.step()
        
        # return scalar loss value
        return meta_loss.item()
      
    def _evaluate_meta_batch(self, eval_batch, inner_learning_rate):
        # unpack eval_batch into support and query windows
        support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations = eval_batch
        eval_meta_loss = self._compute_meta_loss(support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations, inner_learning_rate)
        return eval_meta_loss.item()
    
    def _log_epoch(self, epoch, train_loss, eval_loss, epoch_time_s, train_epochs, log_print_every_k_epochs):
        should_print = (epoch % log_print_every_k_epochs == 0) or (epoch == train_epochs - 1)
        if not should_print: 
            return
        
        print(f"epoch {epoch}/{train_epochs}: train_meta_loss={train_loss:.6f} eval_meta_loss={eval_loss:.6f} time={epoch_time_s:.2f}s")
 
    def _train_dynamics_for_iteration(self, train_epochs, steps_per_epoch, eval_batch):
        # print progress every k epochs
        log_print_every_k_epochs = 5
        
        for epoch in range(train_epochs):
            # start timer for this epoch
            epoch_start_time = time.time()
            # accumulate training loss over this epoch
            epoch_loss_sum = 0.0
            
            for _ in range(steps_per_epoch):
                # sample one train meta batch of support and query windows                
                train_meta_batch = sampler.sample_meta_batch(self.buffer, "train", self.meta_batch_size, self.support_window_size, self.query_window_size, self.device)
                # sample one train meta batch of support/query windows
                support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations = train_meta_batch
                # compute the scalar meta-loss for this train meta-batch
                meta_loss = self._compute_meta_loss(support_observations, support_actions, support_next_observations, query_observations, query_actions, query_next_observations, self.inner_learning_rate)
                # apply the outer update and get the train meta-loss value
                train_loss_value = self._outer_update(meta_loss)
                # accumulate train meta-loss for the epoch average
                epoch_loss_sum += train_loss_value
              
            # average train meta-loss across all meta-updates in this epoch  
            avg_epoch_loss = epoch_loss_sum / steps_per_epoch
            # compute time for this epoch
            epoch_time_s = time.time() - epoch_start_time
            # evaluate the current model on the fixed eval meta-batch
            eval_loss = self._evaluate_meta_batch(eval_batch, self.inner_learning_rate)
            # log epoch-level meta-loss and timing
            self._log_epoch(epoch, avg_epoch_loss, eval_loss, epoch_time_s, train_epochs, log_print_every_k_epochs)

    def train(self):
        print("Starting Full Param Memory training")
        start_time = time.time() # overall train timer
        
        # read training hyperparameters from config
        steps_per_iteration = int(self.train_config["steps_per_iteration"]) # env steps collected per iteration
        iterations = int(self.train_config["iterations"]) # number of outer training iterations
        train_epochs = int(self.train_config["train_epochs"]) # dynamics training epochs per iteration
        
        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            # collect rollouts and add them to the replay buffer
            self.collect_steps(iteration_index, steps_per_iteration)
            
            # update dynamics-model normalization stats from the current replay buffer
            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta)
            
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            steps_per_epoch = max(1, math.ceil(num_train_transitions / (self.meta_batch_size * (self.support_window_size + self.query_window_size))))
            
            # build a fixed eval batch to use during training
            eval_batch = sampler.sample_meta_batch(self.buffer, "eval", self.meta_batch_size, self.support_window_size, self.query_window_size, self.device)
            
            # run GrBAL meta-training for this iteration
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
   
        payload = {
            "state_dict": self.dynamics_model.state_dict(),
            "norm_stats": norm_stats,
        }
        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")
        
        
        
    def _predict_with_prior(self, obs):
        params_for_planning = self.dynamics_model.get_parameter_dict()

        if len(self.support_window) >= self.support_window_size and self.use_online_adaptation:
            window_obs, window_act, window_next_obs = zip(*self.support_window)

            support_obs_np = np.stack(window_obs, axis=0)
            support_act_np = np.stack(window_act, axis=0)
            support_next_obs_np = np.stack(window_next_obs, axis=0)

            support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
            support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
            support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)

            params_for_planning = self.dynamics_model.compute_adapted_parameters(
                support_obs,
                support_act,
                support_next_obs,
                self.inner_learning_rate,
                create_graph=False,
            )

        action = self.planner.plan(obs, parameters=params_for_planning)
        return action
    
    def _predict_with_memory(self, obs):
        prior_parameters = self.dynamics_model.get_parameter_dict()
        
        if not self._memory_selected_this_episode:
            memory_match = self.memory.retrieve(list(self.support_window), self.dynamics_model)
            if memory_match is None:
                self._episode_memory_id = None
                self._episode_params = prior_parameters
            else:
                memory_id, memory_parameters = memory_match
                self._episode_memory_id = memory_id
                memory_parameters_device = OrderedDict()
                for name, param in memory_parameters.items():
                    if not torch.is_tensor(param):
                        continue
                    memory_parameters_device[name] = (
                        param.detach().to(self.device).clone().requires_grad_(True)
                    )
                self._episode_params = memory_parameters_device
                
                
        self._memory_selected_this_episode = True
            
        params_for_planning = self._episode_params  # or self._episode_phi_params if you renamed it
        
        if self.use_online_adaptation:
            window_obs, window_act, window_next_obs = zip(*self.support_window)
            support_obs_np = np.stack(window_obs, axis=0)
            support_act_np = np.stack(window_act, axis=0)
            support_next_obs_np = np.stack(window_next_obs, axis=0)
            
            support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
            support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
            support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)
            
            params_for_planning = self.dynamics_model.compute_adapted_parameters_step(params_for_planning, support_obs, support_act, support_next_obs, self.inner_learning_rate, create_graph=False)
        action = self.planner.plan(obs, parameters=params_for_planning)
        return action
        
    def predict(self, obs):
        if self.support_window is None:
            self.support_window = deque(maxlen=self.support_window_size)
            self.last_obs = None
            self.last_action = None
            
        if self.last_obs is not None and self.last_action is not None:
            self.support_window.append((self.last_obs, self.last_action, obs))
            
        if self.use_memory and self._eval_mode and len(self.support_window) >= self.support_window_size:
            action = self._predict_with_memory(obs)
        else: 
            action = self._predict_with_prior(obs)
            
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.last_obs = obs
        self.last_action = action
        return action
        
    def load(self, path):
        model_path = path
        if os.path.isdir(model_path): model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path): raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        # Restore model weights
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)
        # Restore normalization stats required for planning
        normalization = checkpoint.get("norm_stats")
        if normalization is None: raise RuntimeError("Checkpoint is missing normalization stats. Re-train with updated save() so stats are stored.")
        # Convert to tensors on correct device
        normalization = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in normalization.items()}
        self.dynamics_model.update_normalization_stats(normalization["mean_obs"], normalization["std_obs"], normalization["mean_act"], normalization["std_act"], normalization["mean_delta"], normalization["std_delta"])

        print(f"Loaded dynamics model from {model_path}")
        return self
    
    

    def _permanent_step_task_parameters(self):
        if (not self._eval_mode) or (not self.use_memory):
            return

        if self._episode_params is None or self.support_window is None or len(self.support_window) < self.support_window_size:
            return

        window_obs, window_act, window_next_obs = zip(*self.support_window)
        support_obs_np = np.stack(window_obs, axis=0)
        support_act_np = np.stack(window_act, axis=0)
        support_next_obs_np = np.stack(window_next_obs, axis=0)
        support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
        support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
        support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)
        
        support_delta = support_next_obs - support_obs
        
        slow_params = self.dynamics_model.compute_adapted_parameters_step(
            self._episode_params,
            support_obs,
            support_act,
            support_next_obs,
            self.slow_inner_learning_rate,
            create_graph=False,
        )

        with torch.no_grad():
            loss_before = self.dynamics_model.compute_loss_with_parameters(support_obs, support_act, support_delta, self._episode_params)
            loss_after = self.dynamics_model.compute_loss_with_parameters(support_obs, support_act, support_delta, slow_params)
            improvement = float(loss_before.item()) - float(loss_after.item())
            if improvement > self.abs_improvement_threshold:
                if self._episode_memory_id is None:
                    self.memory.insert(slow_params)
                else:
                    self.memory.update(self._episode_memory_id, slow_params)
                self.memory.save()


    def _reset_episode_state(self):
        self._permanent_step_task_parameters()

        self._memory_selected_this_episode = False
        self._episode_memory_id = None
        self._episode_params = None
        self.support_window = None
        self.last_obs = None
        self.last_action = None
