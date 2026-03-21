import torch
from collections import OrderedDict
from torch.func import functional_call

class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, inner_steps, inner_lr, residual_adapter=None):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

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


    def compute_adapted_params(self, support_obs, support_act, support_next_obs, track_higher_grads):
        # track_higher_grads=True means we need higher-order grads for meta-updates,
        # so we must preserve the inner graph and build a graph on the grads.
        create_graph = track_higher_grads
        retain_graph = track_higher_grads
        
        self.base._assert_normalization_stats()
        if track_higher_grads:
            # Use fresh clones so inner updates don't share version counters with live module params
            parameters = OrderedDict(
                (name, p.clone().requires_grad_(True))
                for name, p in self.residual_adapter.named_parameters()
            )
        else:
            parameters = OrderedDict(
                (name, p.detach().clone().requires_grad_(True))
                for name, p in self.residual_adapter.named_parameters()
            )
                
        device = self.base.mean_obs.device
        support_obs = torch.as_tensor(support_obs, dtype=torch.float32, device=device)
        support_act = torch.as_tensor(support_act, dtype=torch.float32, device=device)
        support_next_obs = torch.as_tensor(support_next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            base_pred_next = self.base.predict_next_state(support_obs, support_act) 
            
        obs_norm = (support_obs - self.base.mean_obs) / self.base.std_obs
        act_norm = (support_act - self.base.mean_act) / self.base.std_act

        next_obs_norm = (support_next_obs - self.base.mean_obs) / self.base.std_obs
        base_pred_next_norm = (base_pred_next - self.base.mean_obs) / self.base.std_obs
        
        for _ in range(self.inner_steps):
            correction_norm = functional_call(self.residual_adapter, parameters, (obs_norm, act_norm, base_pred_next_norm))
            pred_next_norm = base_pred_next_norm + correction_norm
            loss = torch.mean((pred_next_norm - next_obs_norm) ** 2)
            parameter_tensors = tuple(parameters.values())
            
            grads = torch.autograd.grad(
                loss,
                parameter_tensors,
                create_graph=create_graph,
                retain_graph=retain_graph,
                allow_unused=False,
            )
            
            updated = OrderedDict()
            for (name, param), grad in zip(parameters.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(param)
                new_param = param - self.inner_lr * grad
                updated[name] = new_param if track_higher_grads else new_param.detach().requires_grad_(True)

            parameters = updated
                                        
        return parameters
    
    def predict_next_state_with_parameters(self, observation, action, parameters=None):
        if self.residual_adapter is None:
            return self.base.predict_next_state(observation, action)

        self.base._assert_normalization_stats()
        with torch.no_grad():
            base_pred_next = self.base.predict_next_state(observation, action)
            
        obs_norm = (observation - self.base.mean_obs) / self.base.std_obs
        act_norm = (action - self.base.mean_act) / self.base.std_act
        base_pred_next_norm = (base_pred_next - self.base.mean_obs) / self.base.std_obs
        
        if parameters is None:
            correction_norm = self.residual_adapter(obs_norm, act_norm, base_pred_next_norm)
        else:
            correction_norm = functional_call(self.residual_adapter, parameters,(obs_norm, act_norm, base_pred_next_norm))
            
            
        pred_next_norm = base_pred_next_norm + correction_norm
        pred_next = pred_next_norm * self.base.std_obs + self.base.mean_obs
        return pred_next

            
        
