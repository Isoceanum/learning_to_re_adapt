import os
import uuid
import numpy as np
import torch


class FullParamMemory:
    def __init__(self, path, abs_improvement_threshold):
        self.path = path
        self.abs_improvement_threshold = float(abs_improvement_threshold)
        self.entries = []
        self._load()

    def insert(self, params):
        entry = {"id": uuid.uuid4().hex, "params": self._detach_to_cpu(dict(params))}
        self.entries.append(entry)
        return entry["id"]

    def update(self, component_id, params):
        entry = self._get_entry(component_id)
        if entry is None:
            raise KeyError(f"Unknown memory component id: {component_id}")
        entry["params"] = self._detach_to_cpu(dict(params))

    def get(self, component_id):
        entry = self._get_entry(component_id)
        if entry is None:
            return None
        return entry["params"]

    def retrieve(self, steps, dynamics_model):
        if not self.entries:
            return None
        if not steps:
            raise RuntimeError("Cannot retrieve parameters with no steps")

        episode_obs, episode_act, episode_next_obs = zip(*steps)
        obs_np = np.stack(episode_obs, axis=0)
        act_np = np.stack(episode_act, axis=0)
        next_obs_np = np.stack(episode_next_obs, axis=0)

        device = next(dynamics_model.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        act = torch.as_tensor(act_np, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        delta = next_obs - obs

        theta_parameters = dynamics_model.get_parameter_dict()
        with torch.no_grad():
            theta_loss = dynamics_model.compute_loss_with_parameters(obs, act, delta, theta_parameters)
        theta_loss = float(theta_loss.item())

        best_entry = None
        best_loss = None

        with torch.no_grad():
            for entry in self.entries:
                candidate_parameters = self._merge(theta_parameters, entry["params"])
                candidate_loss = dynamics_model.compute_loss_with_parameters(obs, act, delta, candidate_parameters)
                candidate_loss = float(candidate_loss.item())
                if best_loss is None or candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_entry = entry

        if best_entry is None or best_loss is None:
            return None

        improvement = theta_loss - best_loss
        if improvement <= self.abs_improvement_threshold:
            return None

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

    def _merge(self, base_parameters, stored_params):
        merged = dict(base_parameters)
        for name, param in stored_params.items():
            base_param = merged.get(name)
            if torch.is_tensor(param) and torch.is_tensor(base_param) and param.device != base_param.device:
                param = param.to(base_param.device)
            merged[name] = param
        return merged

    def _detach_to_cpu(self, params):
        detached = {}
        for name, param in params.items():
            if torch.is_tensor(param):
                detached[name] = param.detach().cpu().clone()
            else:
                detached[name] = param
        return detached

