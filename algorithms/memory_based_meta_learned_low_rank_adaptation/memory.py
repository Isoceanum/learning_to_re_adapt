import os
import uuid
import numpy as np
import torch


class LoRAMemory:
    def __init__(self, path, relative_improvement, absolute_improvement):
    
        self.path = path
        self.relative_improvement = relative_improvement
        self.absolute_improvement = absolute_improvement
        self.entries = []
        self._load()
     

    def insert(self, lora_params):
        entry = {"id": uuid.uuid4().hex, "lora_state": dict(lora_params)}  # store LoRA params wiht unique id
        self.entries.append(entry)

    def retrieve(self, steps, dynamics_model):
        if not self.entries:
            return None

        if not steps:
            raise RuntimeError("Cannot retrieve lora parameters with no steps")

        episode_obs, episode_act, episode_next_obs = zip(*steps)
        obs_np = np.stack(episode_obs, axis=0)  # (T, obs_dim) observations
        act_np = np.stack(episode_act, axis=0)  # (T, act_dim) actions
        next_obs_np = np.stack(episode_next_obs, axis=0)  # (T, obs_dim) next observations

        device = next(dynamics_model.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)  # move observations to model device
        act = torch.as_tensor(act_np, dtype=torch.float32, device=device)  # move actions to model device
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)  # move next obs to model device
        delta = next_obs - obs  # delta targets

        base_parameters = dynamics_model.get_parameter_dict()
        with torch.no_grad():
            base_loss = dynamics_model.compute_loss_with_parameters(obs, act, delta, base_parameters)  # base prior loss
        base_loss = float(base_loss.item())

        best_entry = None
        best_loss = None

        parameters_list = [self._merge(base_parameters, entry["lora_state"]) for entry in self.entries]  # full params per entry
        with torch.no_grad():
            losses = dynamics_model.compute_loss_with_parameters_batch(obs, act, delta, parameters_list)  # vector of losses
        for entry, candidate_loss in zip(self.entries, losses):
            candidate_loss = float(candidate_loss)  # scalar loss value
            if best_loss is None or candidate_loss < best_loss:
                best_loss = candidate_loss
                best_entry = entry

        if best_entry is None or best_loss is None:
            return None

        improvement = base_loss - best_loss
        threshold = max(self.relative_improvement * abs(base_loss), self.absolute_improvement)  # accept only if strong enough
        if improvement <= threshold:
            return None

        return best_entry["lora_state"]

    def _load(self):
        memory_path = os.path.join(self.path, "memory.pt")
        if not os.path.exists(memory_path):
            return
        self.entries = torch.load(memory_path, map_location="cpu")

    def save(self):
        memory_path = os.path.join(self.path, "memory.pt")
        torch.save(self.entries, memory_path)

    def _score(self, steps, dynamics_model, parameters):
        if not steps:
            raise RuntimeError("Cannot score lora parameters with no steps")

        episode_obs, episode_act, episode_next_obs = zip(*steps)
        obs_np = np.stack(episode_obs, axis=0)
        act_np = np.stack(episode_act, axis=0)
        next_obs_np = np.stack(episode_next_obs, axis=0)

        device = next(dynamics_model.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        act = torch.as_tensor(act_np, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        delta = next_obs - obs

        with torch.no_grad():
            loss = dynamics_model.compute_loss_with_parameters(obs, act, delta, parameters)
        return float(loss.item())

    def _merge(self, base_parameters, lora_params):
        merged = dict(base_parameters)
        for name, param in lora_params.items():
            base_param = merged.get(name)
            if torch.is_tensor(param) and torch.is_tensor(base_param) and param.device != base_param.device:
                param = param.to(base_param.device)
            merged[name] = param
        return merged
