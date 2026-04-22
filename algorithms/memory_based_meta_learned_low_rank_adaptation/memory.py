import os
import uuid
import numpy as np
import torch


class LoRAMemory:
    def __init__(self, path):
        self.path = path
        self.entries = []
        self._load()

    def insert(self, params, created_task):
        entry = {
            "id": uuid.uuid4().hex[:8],
            "params": self._detach_to_cpu(dict(params)),
            "created_task": created_task,
            "retrieval_count": 0,
            "update_count": 0,
        }
        self.entries.append(entry)

    def update(self, component_id, params):
        entry = self._get_entry(component_id)
        if entry is None:
            raise KeyError(f"Unknown memory component id: {component_id}")
        entry["params"] = self._detach_to_cpu(dict(params))
        entry["update_count"] += 1

    def retrieve_with_temporary_adaptation(self, recent_transitions, dynamics_model,  support_window_size, inner_learning_rate):
        if not self.entries:
            print("no entries")
            return None
        
        if len(recent_transitions) < support_window_size:
            return None
        
        steps = list(recent_transitions)
        device = next(dynamics_model.parameters()).device
        episode_obs, episode_act, episode_next_obs = zip(*steps)
        obs_all = torch.as_tensor(np.stack(episode_obs, axis=0), dtype=torch.float32, device=device)
        act_all = torch.as_tensor(np.stack(episode_act, axis=0), dtype=torch.float32, device=device)
        next_obs_all = torch.as_tensor(np.stack(episode_next_obs, axis=0), dtype=torch.float32, device=device)
        theta_parameters = dynamics_model.get_parameter_dict()

        prior_loss = self._score_candidate(theta_parameters, dynamics_model, obs_all, act_all, next_obs_all, support_window_size, inner_learning_rate, device, len(steps))
        best_entry = None
        best_loss = prior_loss
        
        for entry in self.entries:
            candidate_parameters = self._merge(theta_parameters, entry["params"])
            candidate_loss = self._score_candidate(candidate_parameters, dynamics_model, obs_all, act_all, next_obs_all, support_window_size, inner_learning_rate, device, len(steps))
                
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_entry = entry
            
        if best_entry is None:
            return None
        
        best_entry["retrieval_count"] += 1
        return best_entry["id"], best_entry["params"]


    def _get_entry(self, component_id):
        for entry in self.entries:
            if entry.get("id") == component_id:
                return entry
        return None

    def _load(self):
        memory_path = os.path.join(self.path, "memory.pt")
        if not os.path.exists(memory_path):
            return
        self.entries = torch.load(memory_path, map_location="cpu")

    def save(self):
        memory_path = os.path.join(self.path, "memory.pt")
        torch.save(self.entries, memory_path)

    def _merge(self, base_parameters, lora_params):
        merged = dict(base_parameters)
        for name, param in lora_params.items():
            base_param = merged.get(name)
            if torch.is_tensor(param) and torch.is_tensor(base_param) and param.device != base_param.device:
                param = param.to(base_param.device)
            merged[name] = param
        return merged

    def _materialize_params(self, params, device):
        materialized = {}
        for name, param in params.items():
            if torch.is_tensor(param):
                materialized[name] = param.detach().to(device).clone().requires_grad_(True)
            else:
                materialized[name] = param
        return materialized

    def _score_candidate(self, initial_params, dynamics_model, obs_all, act_all, next_obs_all, support_window_size, inner_learning_rate, device, num_steps):
        query_losses = []
        for t in range(support_window_size, num_steps):
            support_start = t - support_window_size
            support_obs = obs_all[support_start:t]
            support_act = act_all[support_start:t]
            support_next_obs = next_obs_all[support_start:t]

            query_obs = obs_all[t:t + 1]
            query_act = act_all[t:t + 1]
            query_next_obs = next_obs_all[t:t + 1]
            query_delta = query_next_obs - query_obs

            candidate_params = self._materialize_params(initial_params, device)
            adapted_params = dynamics_model.compute_adapted_parameters_step(candidate_params, support_obs, support_act, support_next_obs, inner_learning_rate, create_graph=False)
            with torch.no_grad():
                query_loss = dynamics_model.compute_loss_with_parameters(query_obs, query_act, query_delta, adapted_params)
            query_losses.append(float(query_loss.item()))

        return float(np.mean(query_losses))

    def _detach_to_cpu(self, params):
        detached = {}
        for name, param in params.items():
            if torch.is_tensor(param):
                detached[name] = param.detach().cpu().clone()
            else:
                detached[name] = param
        return detached
