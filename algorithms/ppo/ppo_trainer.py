from algorithms.base_trainer import BaseTrainer
from stable_baselines3 import PPO
import envs
import os
import time

class PPOTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_env()
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
            verbose=1,
            tensorboard_log=os.path.join(self.output_dir, "tb") 
        )
        
    def train(self):
        """Run PPO training."""
        train_cfg = self.train_config
        total_timesteps = int(train_cfg.get("total_timesteps", 1e6))

        print(f"🚀 Starting PPO training for {total_timesteps} timesteps...")
        t0 = time.time()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        elapsed = time.time() - t0
        print(f"✅ Training finished. Elapsed: {elapsed:.2f}s")
        
        
    def _predict(self, obs, deterministic: bool):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def load(self, path: str):
        from stable_baselines3 import PPO
        model_path = path
        # Allow passing a directory (containing "model[.zip]")
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model")
        # SB3 appends .zip; accept both
        if not os.path.exists(model_path) and os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"

        self.model = PPO.load(model_path, env=self.env)
        return self
