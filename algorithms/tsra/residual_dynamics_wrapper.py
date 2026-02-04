import torch

class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter=None):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter

    def predict_next_state(self, observation, action):
        if self.residual_adapter is None:
            return self.base.predict_next_state(observation, action)

        self.base._assert_normalization_stats()

        with torch.no_grad():
            base_pred_next = self.base.predict_next_state(observation, action)

        obs_norm = (observation - self.base.mean_obs) / self.base.std_obs
        act_norm = (action - self.base.mean_act) / self.base.std_act
        base_pred_next_norm = (base_pred_next - self.base.mean_obs) / self.base.std_obs

        correction_norm = self.residual_adapter(obs_norm, act_norm, base_pred_next_norm)
        pred_next_norm = base_pred_next_norm + correction_norm
        pred_next = pred_next_norm * self.base.std_obs + self.base.mean_obs
        return pred_next

    def predict_next_state_norm(self, observation, action):
        if self.residual_adapter is None:
            with torch.no_grad():
                base_pred_next = self.base.predict_next_state(observation, action)  # raw
            return (base_pred_next - self.base.mean_obs) / self.base.std_obs

        self.base._assert_normalization_stats()

        with torch.no_grad():
            base_pred_next = self.base.predict_next_state(observation, action)  # raw

        obs_norm = (observation - self.base.mean_obs) / self.base.std_obs
        act_norm = (action - self.base.mean_act) / self.base.std_act
        base_pred_next_norm = (base_pred_next - self.base.mean_obs) / self.base.std_obs

        correction_norm = self.residual_adapter(obs_norm, act_norm, base_pred_next_norm)
        pred_next_norm = base_pred_next_norm + correction_norm
        return pred_next_norm
