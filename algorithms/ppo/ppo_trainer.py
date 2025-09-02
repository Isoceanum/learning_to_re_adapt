from algorithms.base_trainer import BaseTrainer
from stable_baselines3 import PPO
import envs 
import os 

class PPOTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.env = self._make_env()
        self.model = self._build_model()
        
    def _make_env(self):
        """Create a single environment instance."""
        import gymnasium
        env_id = self.config.get("env")
        return gymnasium.make(env_id)
        
    def _build_model(self):
        """Create a PPO model with parameters from config."""
        train_cfg = self.config.get("train", {})
        
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
        train_cfg = self.config.get("train", {})
        total_timesteps = int(train_cfg.get("total_timesteps", 1e6))

        print(f"🚀 Starting PPO training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
        self.model.save(os.path.join(self.output_dir, "model") )
        print(f"✅ Training finished. Model saved to {self.output_dir}")
        
        
    def evaluate(self, episodes=10):
        """Run a simple evaluation loop for a trained PPO model."""
        all_rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

            all_rewards.append(total_reward)
            print(f"Episode {ep+1}/{episodes}: reward = {total_reward:.2f}")

        mean_reward = sum(all_rewards) / len(all_rewards)
        print(f"\n✅ Mean reward over {episodes} episodes: {mean_reward:.2f}")
        return mean_reward