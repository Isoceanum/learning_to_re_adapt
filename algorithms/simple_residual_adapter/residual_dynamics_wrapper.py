import torch
import torch.nn.functional as F


class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter=None, k_step=1, loss_mode="mse"):
        """
        Parameters:
        - base_dynamics_model: the frozen f_theta model
        - residual_adapter: the r_phi model (optional)
        - k_step: how many steps to unroll (1 = standard 1-step)
        - loss_mode: one of ["mse", "huber", "weighted", "balanced"]
        """
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter
        self.k_step = k_step
        self.loss_mode = loss_mode

        # optional: define weights for "weighted" loss_mode
        self.per_dim_weights = torch.ones(self.base.state_dim)  # override externally if needed

    def set_residual_adapter(self, residual_adapter):
        self.residual_adapter = residual_adapter

    def predict_next_state(self, observation, action):
        if self.residual_adapter is None:
            return self.base.predict_next_state(observation, action)

        base_delta_norm = self.base.predict_state_delta_norm(observation, action)
        residual_delta_norm = self.residual_adapter(observation, action)

        delta_norm = base_delta_norm + residual_delta_norm
        delta = delta_norm * self.base.std_delta + self.base.mean_delta
        return observation + delta

    def loss(self, obs, act, next_obs):
        """
        obs, act, next_obs: tensors of shape [B, state_dim], [B, action_dim], [B, state_dim]
        Loss is computed in normalized delta space.
        """
        if self.residual_adapter is None:
            raise RuntimeError("No residual adapter found.")

        total_loss = 0
        batch_size = obs.shape[0]
        state_dim = obs.shape[1]

        if self.k_step == 1:
            base_delta = self.base.predict_state_delta_norm(obs, act)
            residual_delta = self.residual_adapter(obs, act)
            pred_delta = base_delta + residual_delta

            true_delta = (next_obs - obs - self.base.mean_delta) / self.base.std_delta
            error = pred_delta - true_delta  # [B, D]

            if self.loss_mode == "mse":
                total_loss = (error ** 2).mean()

            elif self.loss_mode == "huber":
                total_loss = F.smooth_l1_loss(pred_delta, true_delta, reduction="mean")

            elif self.loss_mode == "weighted":
                weighted = (error ** 2) * self.per_dim_weights
                total_loss = weighted.mean()

            elif self.loss_mode == "balanced":
                per_dim = (error ** 2).mean(dim=0)  # [D]
                total_loss = per_dim.sum() + per_dim.std()

            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        else:
            # k-step rollout
            all_errors = []
            obs_roll = obs.clone()
            for step in range(self.k_step):
                base_delta = self.base.predict_state_delta_norm(obs_roll, act)
                residual_delta = self.residual_adapter(obs_roll, act)
                pred_delta = base_delta + residual_delta
                pred_delta_real = pred_delta * self.base.std_delta + self.base.mean_delta

                obs_roll = obs_roll + pred_delta_real  # advance state prediction
                true_delta = (next_obs - obs - self.base.mean_delta) / self.base.std_delta
                error = pred_delta - true_delta  # per-step error in normalized space
                all_errors.append(error)

            all_errors = torch.stack(all_errors)  # [k, B, D]
            error_mean = all_errors.mean(dim=0)  # average over steps → [B, D]

            if self.loss_mode == "mse":
                total_loss = (error_mean ** 2).mean()

            elif self.loss_mode == "huber":
                total_loss = F.smooth_l1_loss(error_mean, torch.zeros_like(error_mean), reduction="mean")

            elif self.loss_mode == "weighted":
                weighted = (error_mean ** 2) * self.per_dim_weights
                total_loss = weighted.mean()

            elif self.loss_mode == "balanced":
                per_dim = (error_mean ** 2).mean(dim=0)
                total_loss = per_dim.sum() + per_dim.std()

            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        return total_loss
