from algorithms.base_trainer import BaseTrainer
from stable_baselines3 import PPO
import envs
import os
import time


class PPOTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        self.model = self._build_model()

    def _build_model(self):
        """Create a PPO model with parameters from config."""
        train_cfg = self.train_config

        return PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            n_steps=train_cfg.get("n_steps", 2048),
            batch_size=train_cfg.get("batch_size", 64),
            n_epochs=train_cfg.get("n_epochs", 10),
            gamma=train_cfg.get("gamma", 0.99),
            clip_range=train_cfg.get("clip_range", 0.2),
            verbose=1,
            seed=self.train_seed,
            tensorboard_log=os.path.join(self.output_dir, "tb"),
        )

    def train(self):
        """Run PPO training using and single environment."""
        total_timesteps = int(self.train_config["total_env_steps"])

        print(f"Starting PPO:  total_timesteps={total_timesteps}")

        t0 = time.time()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        elapsed = time.time() - t0
        print(f"Training finished. Elapsed: {int(elapsed)//3600:02d}:{(int(elapsed)%3600)//60:02d}:{int(elapsed)%60:02d}")

    def predict(self, obs):
        """Deterministic policy used during evaluation/render."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self):
        """Save PPO model weights."""
        save_path = os.path.join(self.output_dir, "model.zip")
        self.model.save(save_path)
        print(f"PPO model saved to {save_path}")

    def load(self, path: str):
        """Load PPO model weights."""
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model")
        if not os.path.exists(model_path) and os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"

        self.model = PPO.load(model_path, env=self.env)
        print(f"PPO model loaded from {model_path}")
        return self

