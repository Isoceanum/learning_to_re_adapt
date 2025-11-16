import torch
from collections import OrderedDict

class InnerUpdater:
    def __init__(self, model, inner_lr, inner_steps=1):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def _convert_to_tensor(self, value):
        if torch.is_tensor(value):
            return value
        device = next(self.model.parameters()).device
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    
    def compute_adapted_params(self, support_batch):
        observations = self._convert_to_tensor(support_batch["obs"])
        actions = self._convert_to_tensor(support_batch["act"])
        next_observations = self._convert_to_tensor(support_batch["next_obs"])
        
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        next_observations = next_observations.reshape(-1, next_observations.shape[-1])
        
        parameters = OrderedDict((n, p) for n, p in self.model.model.named_parameters())
        
        for _ in range(self.inner_steps):
            inner_loss = self.model.compute_normalized_delta_loss_with_parameters(
                observations, actions, next_observations, parameters
            )

            parameter_tensors = tuple(parameters.values())
            grads = torch.autograd.grad(
                inner_loss,
                parameter_tensors,
                create_graph=True,
                allow_unused=True,
            )

            updated = OrderedDict()
            for (name, param), grad in zip(parameters.items(), grads):
                clean_grad = torch.zeros_like(param) if grad is None else grad
                updated[name] = param - self.inner_lr * clean_grad

            parameters = updated

        return parameters