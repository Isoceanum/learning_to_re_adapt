class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter=None):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter

    def set_residual_adapter(self, residual_adapter):
        self.residual_adapter = residual_adapter

    def predict_next_state(self, observation, action, none):
        base_model_prediction = self.base.predict_next_state(observation, action)

        if self.residual_adapter is None:
            return base_model_prediction

        base_delta = base_model_prediction - observation

        delta_correction_norm = self.residual_adapter(observation, action, base_delta)
        delta_correction_raw = (delta_correction_norm * self.residual_adapter.residual_std + self.residual_adapter.residual_mean)
        
        
        # print(
        #     f"[residual_dbg] |baseΔ|={base_delta.abs().mean():.3g} "
        #     f"|corr|={delta_correction_raw.abs().mean():.3g} "
        #     f"ratio={ (delta_correction_raw.abs().mean() / (base_delta.abs().mean()+1e-8)):.3g} "
        #     f"top_base={base_delta.abs().mean(0).topk(5).indices.tolist()} "
        #     f"top_corr={delta_correction_raw.abs().mean(0).topk(5).indices.tolist()}"
        # )

        pred_delta = base_delta + delta_correction_raw
        next_state_pred = observation + pred_delta
        return next_state_pred
