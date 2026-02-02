import torch

class Normalizer:
    def __init__(self, mean_obs, std_obs, mean_act, std_act, mean_residual, std_residual):
        self.mean_obs = mean_obs
        self.std_obs = std_obs
        self.mean_act = mean_act
        self.std_act = std_act
        self.mean_residual = mean_residual
        self.std_residual = std_residual
        self.eps = 1e-8
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def update_residual_stats_from_raw(self, residual_raw):
        if self.frozen:
            return
        if not torch.is_tensor(residual_raw):
            residual_raw = torch.as_tensor(residual_raw, dtype=torch.float32)

        device = self.mean_residual.device
        residual_raw = residual_raw.to(device=device, dtype=torch.float32)

        self.mean_residual = residual_raw.mean(dim=0)
        self.std_residual = residual_raw.std(dim=0, unbiased=False).clamp_min(self.eps)

    def normalize_observations(self, obs):
        return (obs - self.mean_obs) / (self.std_obs + self.eps)

    def normalize_actions(self, act):
        return (act - self.mean_act) / (self.std_act + self.eps)

    def normalize_residual(self, residual):
        return (residual - self.mean_residual) / (self.std_residual + self.eps)

    def denormalize_residual(self, residual_norm):
        return residual_norm * (self.std_residual + self.eps) + self.mean_residual
