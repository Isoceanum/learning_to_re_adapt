from gymnasium.envs.registration import register

# Register HalfCheetah
register(
    id="HalfCheetahCustom-v0",
    entry_point="envs.half_cheetah_env:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

# Register Ant
register(
    id="AntCustom-v0",
    entry_point="envs.ant_env:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register Hopper
register(
    id="HopperCustom-v0",
    entry_point="envs.hopper:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="ReacherCustom-v0",
    entry_point="envs.reacher:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)