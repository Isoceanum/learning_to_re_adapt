class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter=None):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter

    def set_residual_adapter(self, residual_adapter):
        self.residual_adapter = residual_adapter

    def predict_next_state(self, observation, action):
        base_model_prediction = self.base.predict_next_state(observation, action)

        if self.residual_adapter is None:
            return base_model_prediction

        base_delta = base_model_prediction - observation

        delta_correction_norm = self.residual_adapter(observation, action, base_delta)
        delta_correction_raw = (delta_correction_norm * self.residual_adapter.residual_std + self.residual_adapter.residual_mean)

        pred_delta = base_delta + delta_correction_raw
        next_state_pred = observation + pred_delta
        return next_state_pred
