from algorithms.base_trainer import BaseTrainer
import os
from algorithms.mb_mpc.trainer import DynamicsTrainer

class MBMPCTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_env()
        self.model = self._build_model()

    def _make_env(self):
        """Create a single env with x-position included in observations.

        MB-MPC uses a planner that computes forward progress from the
        x-position; therefore we enforce
        exclude_current_positions_from_observation=False and avoid
        any vectorized env wrappers.
        """
        import envs  # ensure custom envs are registered
        import gymnasium as gym

        env_id = self.config.get("env")
        return gym.make(env_id, exclude_current_positions_from_observation=False)

    def _make_eval_env(self):
        """Create a single, non-vectorized env for evaluation with x-position."""
        import envs  # ensure custom envs are registered
        import gymnasium as gym

        env_id = self.config.get("env")
        return gym.make(env_id, exclude_current_positions_from_observation=False)

        
    def _build_model(self):
        """Create and configure the MB-MPC components from config.

        Returns a DynamicsTrainer that owns the learned dynamics model,
        replay buffer, and CEM planner. Hyperparameters are read from
        `train:` in the YAML with sensible defaults.
        """
        train_cfg = self.train_config
        env = self.env

        # Infer dimensions from the environment
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Prefer ctrl cost weight from the env when available
        ctrl_cost_weight = float(
            getattr(getattr(env, "unwrapped", env), "_ctrl_cost_weight", train_cfg.get("ctrl_cost_weight", 0.1))
        )

        # Read hyperparameters with fallbacks
        hidden_sizes = train_cfg.get("hidden_sizes", [256, 256])
        lr = float(train_cfg.get("lr", train_cfg.get("learning_rate", 1e-3)))
        batch_size = int(train_cfg.get("batch_size", 256))
        val_ratio = float(train_cfg.get("val_ratio", 0.1))
        horizon = int(train_cfg.get("horizon", 20))
        num_candidates = int(train_cfg.get("num_candidates", 1000))
        device = train_cfg.get("device", "cpu")
        ensemble_size = int(train_cfg.get("ensemble_size", 1))

        # Try to get an env-provided reward function for model-based planning
        reward_fn = None
        get_r = getattr(getattr(env, "unwrapped", env), "get_model_reward_fn", None)
        if callable(get_r):
            reward_fn = get_r()

        return DynamicsTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=env.action_space,
            hidden_sizes=hidden_sizes,
            lr=lr,
            batch_size=batch_size,
            val_ratio=val_ratio,
            horizon=horizon,
            num_candidates=num_candidates,
            device=device,
            ctrl_cost_weight=ctrl_cost_weight,
            reward_fn=reward_fn,
            ensemble_size=ensemble_size,
        )
        

    def train(self):
        """Run MB-MPC training via the dynamics trainer loop."""
        cfg = self.train_config
        n_iterations = int(cfg.get("total_iterations", cfg.get("iterations", 5)))
        init_random_steps = int(cfg.get("init_random_steps", 1000))
        rollout_steps = int(cfg.get("rollout_steps", 1000))
        epochs = int(cfg.get("epochs", 50))

        save_path = os.path.join(self.output_dir, "model.pt")

        print(
            f"🚀 Starting MB-MPC training: iterations={n_iterations}, "
            f"init_random_steps={init_random_steps}, rollout_steps={rollout_steps}, epochs={epochs}"
        )
        self.model.run_training_loop(
            self.env,
            n_iterations=n_iterations,
            init_random_steps=init_random_steps,
            rollout_steps=rollout_steps,
            epochs=epochs,
            save_path=save_path,
        )
        print("✅ Training finished.")
        
        
    def _predict(self, obs, deterministic: bool):
        """Select action using the CEM planner on the learned dynamics."""
        return self.model.planner.plan(obs)

    def save(self):
        """Save learned dynamics ensemble weights and normalization stats."""
        import torch

        save_path = os.path.join(self.output_dir, "model.pt")
        models = getattr(self.model, "models", None)
        if models is None:
            # Backward compatibility: single model path
            dyn_model = getattr(self.model, "model", None)
            if dyn_model is None:
                raise RuntimeError("No dynamics model(s) found to save")
            ckpt = {
                "ensemble_size": 1,
                "state_dicts": [dyn_model.state_dict()],
                "normalization": [{
                    "state_mean": dyn_model.state_mean,
                    "state_std": dyn_model.state_std,
                    "action_mean": dyn_model.action_mean,
                    "action_std": dyn_model.action_std,
                    "delta_mean": dyn_model.delta_mean,
                    "delta_std": dyn_model.delta_std,
                }],
            }
        else:
            ckpt = {
                "ensemble_size": len(models),
                "state_dicts": [m.state_dict() for m in models],
                "normalization": [{
                    "state_mean": m.state_mean,
                    "state_std": m.state_std,
                    "action_mean": m.action_mean,
                    "action_std": m.action_std,
                    "delta_mean": m.delta_mean,
                    "delta_std": m.delta_std,
                } for m in models],
            }
        torch.save(ckpt, save_path)

    def load(self, path: str):
        """Load dynamics model weights and normalization stats."""
        import torch

        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")

        ckpt = torch.load(model_path, map_location="cpu")
        ens_size = int(ckpt.get("ensemble_size", 1))

        # Ensure we have the expected number of models
        models = getattr(self.model, "models", None)
        if models is None:
            # Backward compatibility: single model
            dyn_model = getattr(self.model, "model", None)
            if dyn_model is None:
                raise RuntimeError("No dynamics model(s) available for loading")
            state_dicts = ckpt.get("state_dicts") or [ckpt.get("state_dict")]
            normalization = ckpt.get("normalization")
            if isinstance(normalization, dict):
                normalization = [normalization]
            dyn_model.load_state_dict(state_dicts[0])
            if normalization:
                norm = normalization[0]
                dyn_model.state_mean = norm.get("state_mean", None)
                dyn_model.state_std = norm.get("state_std", None)
                dyn_model.action_mean = norm.get("action_mean", None)
                dyn_model.action_std = norm.get("action_std", None)
                dyn_model.delta_mean = norm.get("delta_mean", None)
                dyn_model.delta_std = norm.get("delta_std", None)
            return self

        if ens_size != len(models):
            raise ValueError(f"Checkpoint ensemble_size={ens_size} does not match current models={len(models)}")

        state_dicts = ckpt["state_dicts"]
        norms = ckpt.get("normalization", [{}] * ens_size)
        for m, sd, norm in zip(models, state_dicts, norms):
            m.load_state_dict(sd)
            m.state_mean = norm.get("state_mean", None)
            m.state_std = norm.get("state_std", None)
            m.action_mean = norm.get("action_mean", None)
            m.action_std = norm.get("action_std", None)
            m.delta_mean = norm.get("delta_mean", None)
            m.delta_std = norm.get("delta_std", None)

        return self
