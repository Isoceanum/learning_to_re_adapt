from perturbations.action_scaling_perturbation import ActionScalingPerturbation


def get_perturbation_class(perturbation_type):
    perturbation_type = perturbation_type.lower()
    
    if perturbation_type == "action_scaling":
        return ActionScalingPerturbation
    
    else:
        raise ValueError(f"Unknown perturbation type '{perturbation_type}'")
    

def resolve_perturbation_env(env, config, seed):
    perturbation_config = config.get("perturbation", {})

    if not perturbation_config:
        return env
    
    perturbation_type = perturbation_config.get("type")
    perturbation_class = get_perturbation_class(perturbation_type)
    
    return perturbation_class(env, perturbation_config, seed)