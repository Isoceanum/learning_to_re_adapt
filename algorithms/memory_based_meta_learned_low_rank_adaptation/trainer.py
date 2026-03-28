"""
Phase 3A guidance (trainer):

This file is the primary implementation target for Phase 3A.

Required changes for Phase 3A:
- Replace single-step support/query logic with **sequence-based** meta-training.
- Read new config keys:
  - `inner_update_interval` (N)
  - `inner_updates_k` (K)
- Define `meta_window_size = (K+1) * N` and use it for sampling.
- Meta-loss (Option 2 + include prior):
  - Chunk 0 uses the prior parameters (no adaptation).
  - For j = 1..K:
      * Support = all data up to chunk j-1 (prefix length j*N)
      * Apply ONE inner step (cumulative)
      * Query loss on chunk j
  - Meta-loss = mean of all chunk losses (K+1 terms).

Online adaptation (predict / collect):
- Maintain episode-local transition list.
- Every N steps, if updates_done < K, do **one** inner update using
  all transitions so far (full batch), cumulative on current params.
- After K updates, **freeze** adapted params for the rest of the episode.
- Reset all episode adaptation state on episode end.

Other required plumbing:
- Update `steps_per_epoch` based on `meta_window_size`.
- Use the new sequence sampler (no support/query split).
- Ensure planning always uses the most recent adapted params if available.
"""

from collections import deque
import os

import numpy as np
from algorithms.base_trainer import BaseTrainer

import torch
import math
import time

from algorithms.memory_based_meta_learned_low_rank_adaptation.dynamics_model import DynamicsModel
from common.transition_buffer import TransitionBuffer
from algorithms.memory_based_meta_learned_low_rank_adaptation import sampler
from common.planner import make_planner


class MemoryBasedMetaLearnedLowRankAdaptationTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.inner_update_interval = int(self.train_config["inner_update_interval"])  # N: steps per chunk / update interval
        self.inner_updates_k = int(self.train_config["inner_updates_k"])  # K: number of cumulative inner updates
        self.meta_window_size = (self.inner_updates_k + 1) * self.inner_update_interval  # full sequence length = (K+1) chunks of size N
        
        self.inner_learning_rate = float(self.train_config["inner_learning_rate"]) # step size for the inner (adaptation) gradient update
        self.use_online_adaptation = self.train_config["use_online_adaptation"] # enable/disable online adaptation during planning/eval
        
        self.outer_learning_rate = float(self.train_config["outer_learning_rate"]) 
        self.meta_batch_size = int(self.train_config["meta_batch_size"])

        # lora params 
        self.lora_rank = int(self.train_config["lora_rank"])
        self.lora_alpha = float(self.train_config["lora_alpha"])
        
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner =self._make_planner()
        self.buffer = self._make_buffer()
        
        self.episode_transitions = []  # all transitions collected so far in the current episode
        self.online_adapted_parameters = None  # current cumulatively adapted param dict for this episode
        self.online_num_updates = 0  # how many online inner updates have been applied this episode
        self.last_obs = None  # previous observation, used to form the next transition
        self.last_action = None  # previous action, used to form the next transition
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
    
    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None: 
            raise AttributeError("Missing dynamics_model config in YAML")
    
        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        seed = self.train_seed
        return DynamicsModel(self.observation_dim, self.action_dim, hidden_sizes, self.outer_learning_rate, self.lora_rank, self.lora_alpha, seed)
    
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        base_env = getattr(self.env, "unwrapped", self.env)
        
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        dynamics_fn = self.dynamics_model.predict_next_state_with_parameters

        return make_planner(planner_config, dynamics_fn, reward_fn, self.env.action_space, self.device, self.train_seed)
     
    def _compute_meta_loss(self, observations, actions, next_observations, inner_learning_rate):        
        meta_loss_terms = []  # collect one scalar sequence meta-loss per batch item

        # iterate over each sampled sequence window in the batch
        for batch_index in range(observations.shape[0]):
            
            # unpack support and query window
            obs_task = observations[batch_index]  # full observation sequence for this batch item
            act_task = actions[batch_index]  # full action sequence for this batch item
            next_obs_task = next_observations[batch_index]  # full next-observation sequence for this batch item
            
            
            current_parameters = self.dynamics_model.get_parameter_dict()  # start this task from the current meta-learned prior
            chunk_losses = []  # collect chunk 0 .. chunk K losses for this one sequence

            chunk0_obs = obs_task[:self.inner_update_interval]  # first chunk observations uses the prior (no adaptation yet)
            chunk0_act = act_task[:self.inner_update_interval]  # first chunk actions
            chunk0_next_obs = next_obs_task[:self.inner_update_interval]  # first chunk next observations
            chunk0_delta = chunk0_next_obs - chunk0_obs  # convert first chunk targets into delta targets

            chunk0_loss = self.dynamics_model.compute_loss_with_parameters(chunk0_obs, chunk0_act, chunk0_delta, current_parameters)  # prior loss on chunk 0 before any adaptation
            chunk_losses.append(chunk0_loss)  # include prior term in the meta-loss
            
            for update_index in range(1, self.inner_updates_k + 1):  # chunks 1..K use cumulative adaptation
                support_end = update_index * self.inner_update_interval  # prefix length = j * N

                support_obs = obs_task[:support_end]  # all observations up to chunk j-1
                support_act = act_task[:support_end]  # all actions up to chunk j-1
                support_next_obs = next_obs_task[:support_end]  # all next observations up to chunk j-1

                chunk_start = support_end  # chunk j starts right after the support prefix
                chunk_end = chunk_start + self.inner_update_interval  # chunk j has length N

                query_obs = obs_task[chunk_start:chunk_end]  # current evaluation chunk observations
                query_act = act_task[chunk_start:chunk_end]  # current evaluation chunk actions
                query_next_obs = next_obs_task[chunk_start:chunk_end]  # current evaluation chunk next observations
                query_delta = query_next_obs - query_obs  # convert current chunk targets into delta targets

                current_parameters = self.dynamics_model.compute_adapted_parameters_step(current_parameters, support_obs, support_act, support_next_obs, inner_learning_rate, create_graph=True)  # take one cumulative inner step from the current parameter dict

                query_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, current_parameters, )  # evaluate updated params on chunk j
                chunk_losses.append(query_loss)  # include this adapted chunk loss in the sequence meta-loss
                
            task_meta_loss = torch.stack(chunk_losses).mean()  # average chunk 0 .. chunk K for this sequence
            meta_loss_terms.append(task_meta_loss)  # store one scalar sequence meta-loss for this batch item
            
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
        observations, actions, next_observations = eval_batch  # sequence sampler returns one full window per batch item
        eval_meta_loss = self._compute_meta_loss(observations, actions, next_observations, inner_learning_rate)  # evaluate the sequence-based meta-loss on the fixed eval batch
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
                train_meta_batch = sampler.sample_sequence_meta_batch(self.buffer, "train", self.meta_batch_size, self.meta_window_size, self.device)  # sample full sequence windows for Phase 3A meta-training
                # sample one train meta batch of support/query windows
                observations, actions, next_observations = train_meta_batch  # unpack full sequence windows instead of support/query splits                
                # compute the scalar meta-loss for this train meta-batch
                meta_loss = self._compute_meta_loss(observations, actions, next_observations, self.inner_learning_rate)  # compute sequence-based meta-loss for this train batch                
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

    def _log_adaptation_diagnostics(self):
        eval_obs, eval_act, eval_next_obs = self.buffer.get_trajectories("eval")
        if len(eval_obs) == 0:
            print("adapt_diag: skipped (no eval trajectories yet)")
            return

        max_eval_episodes = 5
        start_index = max(0, len(eval_obs) - max_eval_episodes)
        episode_indices = range(start_index, len(eval_obs))

        n = self.inner_update_interval
        k = self.inner_updates_k

        prior_losses = []
        pre_losses_by_step = [[] for _ in range(k)]
        post_losses_by_step = [[] for _ in range(k)]
        improvements_by_step = [[] for _ in range(k)]
        update_norms_by_step = [[] for _ in range(k)]
        freeze_rest_losses = []
        continue_rest_losses = []
        used_episodes = 0
        skipped_short = 0

        def lora_update_norm(params_before, params_after):
            deltas = []
            for name, param in params_before.items():
                if "A.weight" in name or "B.weight" in name:
                    deltas.append((params_after[name] - param).detach().reshape(-1))
            if not deltas:
                return 0.0
            return torch.norm(torch.cat(deltas)).item()

        was_training = self.dynamics_model.training
        self.dynamics_model.eval()

        for idx in episode_indices:
            obs_np = eval_obs[idx]
            act_np = eval_act[idx]
            next_obs_np = eval_next_obs[idx]

            episode_len = len(obs_np)
            if episode_len < n:
                skipped_short += 1
                continue

            used_episodes += 1
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)

            num_full_chunks = episode_len // n
            if num_full_chunks == 0:
                skipped_short += 1
                continue

            current_params = self.dynamics_model.get_parameter_dict()

            with torch.no_grad():
                chunk0_obs = obs[:n]
                chunk0_act = act[:n]
                chunk0_next_obs = next_obs[:n]
                chunk0_delta = chunk0_next_obs - chunk0_obs
                chunk0_loss = self.dynamics_model.compute_loss_with_parameters(chunk0_obs, chunk0_act, chunk0_delta, current_params)
                prior_losses.append(chunk0_loss.item())

            max_updates = min(k, num_full_chunks - 1)

            for update_index in range(1, max_updates + 1):
                support_end = update_index * n
                support_obs = obs[:support_end]
                support_act = act[:support_end]
                support_next_obs = next_obs[:support_end]

                chunk_start = support_end
                chunk_end = chunk_start + n
                query_obs = obs[chunk_start:chunk_end]
                query_act = act[chunk_start:chunk_end]
                query_next_obs = next_obs[chunk_start:chunk_end]
                query_delta = query_next_obs - query_obs

                with torch.no_grad():
                    pre_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, current_params)

                updated_params = self.dynamics_model.compute_adapted_parameters_step(
                    current_params,
                    support_obs,
                    support_act,
                    support_next_obs,
                    self.inner_learning_rate,
                    create_graph=False,
                )

                with torch.no_grad():
                    post_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, updated_params)

                step_index = update_index - 1
                pre_losses_by_step[step_index].append(pre_loss.item())
                post_losses_by_step[step_index].append(post_loss.item())
                improvements_by_step[step_index].append(pre_loss.item() - post_loss.item())
                update_norms_by_step[step_index].append(lora_update_norm(current_params, updated_params))

                current_params = updated_params

            rest_start = (max_updates + 1) * n
            if rest_start < episode_len:
                rest_obs = obs[rest_start:]
                rest_act = act[rest_start:]
                rest_next_obs = next_obs[rest_start:]
                rest_delta = rest_next_obs - rest_obs

                with torch.no_grad():
                    freeze_loss = self.dynamics_model.compute_loss_with_parameters(rest_obs, rest_act, rest_delta, current_params)
                freeze_rest_losses.append(freeze_loss.item())

                continue_loss_total = 0.0
                continue_count = 0
                params_cont = current_params

                for chunk_index in range(max_updates + 1, num_full_chunks):
                    support_end = chunk_index * n
                    support_obs = obs[:support_end]
                    support_act = act[:support_end]
                    support_next_obs = next_obs[:support_end]

                    params_cont = self.dynamics_model.compute_adapted_parameters_step(
                        params_cont,
                        support_obs,
                        support_act,
                        support_next_obs,
                        self.inner_learning_rate,
                        create_graph=False,
                    )

                    chunk_start = support_end
                    chunk_end = chunk_start + n
                    query_obs = obs[chunk_start:chunk_end]
                    query_act = act[chunk_start:chunk_end]
                    query_next_obs = next_obs[chunk_start:chunk_end]
                    query_delta = query_next_obs - query_obs

                    with torch.no_grad():
                        chunk_loss = self.dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, params_cont)

                    continue_loss_total += chunk_loss.item() * n
                    continue_count += n

                tail_start = num_full_chunks * n
                if tail_start < episode_len:
                    tail_obs = obs[tail_start:]
                    tail_act = act[tail_start:]
                    tail_next_obs = next_obs[tail_start:]
                    tail_delta = tail_next_obs - tail_obs

                    with torch.no_grad():
                        tail_loss = self.dynamics_model.compute_loss_with_parameters(tail_obs, tail_act, tail_delta, params_cont)

                    tail_len = episode_len - tail_start
                    continue_loss_total += tail_loss.item() * tail_len
                    continue_count += tail_len

                if continue_count > 0:
                    continue_rest_losses.append(continue_loss_total / continue_count)

        if was_training:
            self.dynamics_model.train()

        print(
            f"adapt_diag: episodes_used={used_episodes} "
            f"(skipped_short={skipped_short}, total_eval={len(eval_obs)}, last_n={len(list(episode_indices))})"
        )

        if prior_losses:
            print(f"adapt_diag: chunk0_loss_mean={float(np.mean(prior_losses)):.6f}")

        for step_idx in range(k):
            if not pre_losses_by_step[step_idx]:
                continue
            pre_mean = float(np.mean(pre_losses_by_step[step_idx]))
            post_mean = float(np.mean(post_losses_by_step[step_idx]))
            delta_mean = float(np.mean(improvements_by_step[step_idx]))
            norm_mean = float(np.mean(update_norms_by_step[step_idx])) if update_norms_by_step[step_idx] else 0.0
            print(
                f"adapt_diag: update_{step_idx + 1}: "
                f"pre_loss={pre_mean:.6f} post_loss={post_mean:.6f} "
                f"delta={delta_mean:.6f} update_norm={norm_mean:.6f}"
            )

        if freeze_rest_losses:
            print(f"adapt_diag: freeze_rest_loss_mean={float(np.mean(freeze_rest_losses)):.6f}")
        if continue_rest_losses:
            print(f"adapt_diag: continue_rest_loss_mean={float(np.mean(continue_rest_losses)):.6f}")

    def train(self):
        print("Starting meta-lora training")
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
            steps_per_epoch = max(1, math.ceil(num_train_transitions / (self.meta_batch_size * self.meta_window_size)))  # one meta-update consumes sequence windows of length (K+1)*N            
            # build a fixed eval batch to use during training
            eval_batch = sampler.sample_sequence_meta_batch(self.buffer, "eval", self.meta_batch_size, self.meta_window_size, self.device)  # fixed eval batch uses full sequence windows too            
            # run GrBAL meta-training for this iteration
            self._train_dynamics_for_iteration(train_epochs, steps_per_epoch, eval_batch)
            # offline diagnostics on eval trajectories (no new data collection)
            self._log_adaptation_diagnostics()

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
        
    def predict(self, obs):
        if self.last_obs is not None and self.last_action is not None:
            self.episode_transitions.append((self.last_obs, self.last_action, obs))  # store the newest transition in this episode

        params_for_planning = self.online_adapted_parameters if self.online_adapted_parameters is not None else self.dynamics_model.get_parameter_dict()  # use the latest adapted params if this episode has already updated

        if self.use_online_adaptation and self.online_num_updates < self.inner_updates_k and len(self.episode_transitions) >= (self.online_num_updates + 1) * self.inner_update_interval:  # trigger exactly one new cumulative update each time we have another full N-step chunk
            episode_obs, episode_act, episode_next_obs = zip(*self.episode_transitions)  # use all transitions collected so far in this episode as cumulative support

            support_obs_np = np.stack(episode_obs, axis=0)  # stack episode observations into one cumulative support array
            support_act_np = np.stack(episode_act, axis=0)  # stack episode actions into one cumulative support array
            support_next_obs_np = np.stack(episode_next_obs, axis=0)  # stack episode next observations into one cumulative support array

            support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
            support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
            support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)

            current_parameters = self.online_adapted_parameters if self.online_adapted_parameters is not None else self.dynamics_model.get_parameter_dict()  # continue from the latest episode-adapted params, or start from the prior
            self.online_adapted_parameters = self.dynamics_model.compute_adapted_parameters_step(current_parameters, support_obs, support_act, support_next_obs, self.inner_learning_rate, create_graph=False)  # take exactly one cumulative online inner step
            self.online_num_updates += 1  # record that this episode has used one more online update
            params_for_planning = self.online_adapted_parameters  # plan with the freshly updated parameters
            
            
        action = self.planner.plan(obs, parameters=params_for_planning)
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

    def _reset_episode_state(self):
        self.episode_transitions = []  # clear all collected transitions for the finished episode
        self.online_adapted_parameters = None  # forget the episode-specific adapted parameter dict
        self.online_num_updates = 0  # reset the count of online updates for the new episode
        self.last_obs = None  # clear previous observation
        self.last_action = None  # clear previous action
