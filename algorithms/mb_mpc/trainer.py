import torch
import torch.optim as optim
import numpy as np

from mb_mpc.dynamics import DynamicsModel
from mb_mpc.buffer import ReplayBuffer
from mb_mpc.planner import RandomShootingPlanner


class DynamicsTrainer:
    def __init__(self, state_dim, action_dim, action_space,
                 hidden_sizes=[256, 256],
                 lr=1e-3, batch_size=256, val_ratio=0.1,
                 horizon=20, num_candidates=1000,
                 device="cpu"):
        self.device = torch.device(device)
        self.model = DynamicsModel(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.val_ratio = val_ratio

        self.buffer = ReplayBuffer(state_dim, action_dim)

        # Planner (initialized after first training)
        self.planner = RandomShootingPlanner(
            self.model, action_space,
            horizon=horizon, num_candidates=num_candidates,
            device=self.device
        )

    def collect_rollouts(self, env, num_steps=1000, use_planner=False):
        """
        Collect rollouts into the buffer.
        - If use_planner=False: random actions
        - If use_planner=True: use MPC planner
        """
        state, _ = env.reset()
        for _ in range(num_steps):
            if use_planner:
                action = self.planner.plan(state)
            else:
                action = env.action_space.sample()

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, next_state)
            state = next_state

            if done:
                state, _ = env.reset()

    def train_dynamics(self, epochs=50):
        """Train dynamics model on buffer until val loss stabilizes."""
        states, actions, next_states = self.buffer.get_all()
        train, val = self.buffer.train_val_split(self.val_ratio)

        # fit normalization stats
        self.model.fit_normalization(states, actions, next_states)

        train_states, train_actions, train_next_states = train
        val_states, val_actions, val_next_states = val

        train_states = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        train_actions = torch.tensor(train_actions, dtype=torch.float32, device=self.device)
        train_next_states = torch.tensor(train_next_states, dtype=torch.float32, device=self.device)

        val_states = torch.tensor(val_states, dtype=torch.float32, device=self.device)
        val_actions = torch.tensor(val_actions, dtype=torch.float32, device=self.device)
        val_next_states = torch.tensor(val_next_states, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            # sample minibatches
            idxs = np.random.permutation(len(train_states))
            for start in range(0, len(train_states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                s = train_states[batch_idx]
                a = train_actions[batch_idx]
                ns = train_next_states[batch_idx]

                loss = self.model.train_step(self.optimizer, s, a, ns)

            # compute val loss
            with torch.no_grad():
                val_loss = self.model.loss_fn(val_states, val_actions, val_next_states).item()

            print(f"Epoch {epoch+1}/{epochs} | train_loss={loss:.4f} | val_loss={val_loss:.4f}")

    def run_training_loop(self, env, n_iterations=5,
                          init_random_steps=1000, rollout_steps=1000, epochs=50):
        """
        Full MB-MPC training loop:
          - Iter 0: collect random rollouts
          - Train dynamics model
          - Iter >0: use planner for rollouts
          - Retrain dynamics model each iteration
        """
        for itr in range(n_iterations):
            print(f"\n=== Iteration {itr+1}/{n_iterations} ===")

            if itr == 0:
                print("Collecting initial random rollouts...")
                self.collect_rollouts(env, num_steps=init_random_steps, use_planner=False)
            else:
                print("Collecting planner-based rollouts...")
                self.collect_rollouts(env, num_steps=rollout_steps, use_planner=True)

            print("Training dynamics model...")
            self.train_dynamics(epochs=epochs)
