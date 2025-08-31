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