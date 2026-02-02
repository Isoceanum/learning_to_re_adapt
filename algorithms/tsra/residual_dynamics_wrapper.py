import torch

class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter, normalizer):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter
        self.normalizer = normalizer

    def predict_next_state(self, observation, action):
        base_model_prediction = self.base.predict_next_state(observation, action)

        if self.residual_adapter is None:
            return base_model_prediction

        base_delta = base_model_prediction - observation
        
        obs_norm = self.normalizer.normalize_observations(observation)
        act_norm = self.normalizer.normalize_actions(action)
        
        delta_correction_norm = self.residual_adapter(obs_norm, act_norm)
        delta_correction_raw = self.normalizer.denormalize_residual(delta_correction_norm)
        
        pred_delta = base_delta + delta_correction_raw
        next_state_pred = observation + pred_delta
        
        return next_state_pred
