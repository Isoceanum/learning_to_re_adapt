from perturbations.action_scaling_perturbation import ActionScalingPerturbation
from perturbations.action_swap_perturbation import ActionSwapPerturbation
from perturbations.action_inversion_perturbation import ActionInversionPerturbation
from perturbations.cripple_perturbation import CripplePerturbation


def get_perturbation_class(perturbation_type):
    perturbation_type = perturbation_type.lower()
    
    if perturbation_type == "action_scaling":
        return ActionScalingPerturbation
    
    if perturbation_type == "action_swap":
        return ActionSwapPerturbation
    
    if perturbation_type == "action_inversion":
        return ActionInversionPerturbation
    
    if perturbation_type == "cripple":
        return CripplePerturbation
    
    else:
        raise ValueError(f"Unknown perturbation type '{perturbation_type}'")
    

def resolve_perturbation_env(env, config, seed):
    perturbation_config = config.get("perturbation", {})

    if not perturbation_config:
        return env
    
    perturbation_type = perturbation_config.get("type")
    perturbation_class = get_perturbation_class(perturbation_type)
    
    return perturbation_class(env, perturbation_config, seed)