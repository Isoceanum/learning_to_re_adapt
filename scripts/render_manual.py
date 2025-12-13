import argparse
import os
import time
import yaml

# Ensure custom envs are registered
import envs  # noqa: F401
import gymnasium as gym

from scripts.run_experiment import _build_trainer

# Rendering knobs: keep these constants to avoid CLI complexity.
# Set TARGET_VIEW_FPS=0 to disable extra sleeping (fastest possible rendering).
TARGET_VIEW_FPS = 0   # set >0 to pace playback, 0 for no artificial delay
RENDER_STRIDE = 5     # render every N physics steps; 5 looked natural at 100 Hz sim
NUM_EPISODES = 3      # number of episodes to show


def compute_render_stride(env_dt: float, target_view_fps: float) -> tuple[int, float]:
    """Return (stride, sim_fps) so rendering stays roughly target_view_fps."""
    sim_fps = 1.0 / env_dt if env_dt > 0 else 0.0
    stride = max(1, int(round(sim_fps / target_view_fps))) if target_view_fps > 0 else 1
    return stride, sim_fps


def main():
    parser = argparse.ArgumentParser(description="Render a trained policy with manual frame skipping.")
    parser.add_argument("run_dir", type=str, help="Run directory containing config.yaml and checkpoint(s)")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Build trainer bound to this run directory
    trainer = _build_trainer(config, run_dir)

    # Recreate env with manual rendering control (no auto-render in step)
    env_id = config.get("env")
    env = gym.make(env_id, render_mode=None, exclude_current_positions_from_observation=False)
    trainer.env = env

    # Load trained weights
    trainer.load(run_dir)

    # Compute stride based on current env dt; allow overriding with RENDER_STRIDE
    sim_fps = 1.0 / trainer.env.unwrapped.dt if trainer.env.unwrapped.dt > 0 else 0.0
    if RENDER_STRIDE and RENDER_STRIDE > 0:
        render_stride = RENDER_STRIDE
    else:
        render_stride, _ = compute_render_stride(trainer.env.unwrapped.dt, TARGET_VIEW_FPS)
    effective_view_fps = sim_fps / render_stride if render_stride > 0 else sim_fps
    print(
        f"Sim dt={trainer.env.unwrapped.dt:.4f}s ({sim_fps:.1f} fps). "
        f"Rendering every {render_stride} step(s) ≈ {effective_view_fps:.1f} fps."
    )

    try:
        for ep in range(NUM_EPISODES):
            obs, _ = trainer.env.reset()
            done = False
            ep_return = 0.0
            step = 0
            while not done:
                action = trainer.predict(obs)
                obs, reward, terminated, truncated, _ = trainer.env.step(action)
                if step % render_stride == 0:
                    trainer.env.unwrapped.mujoco_renderer.render("human")
                step += 1
                done = terminated or truncated
                ep_return += float(reward)
            print(f"Episode {ep + 1}/{NUM_EPISODES} return: {ep_return:.2f}")
    finally:
        try:
            trainer.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
