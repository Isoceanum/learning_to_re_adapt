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
    entry_point="envs.hopper_env:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

# Register HalfCheetah
register(
    id="HalfCheetahNagabandi-v0",
    entry_point="envs.half_cheetah_env_nagabandi:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)



# Register HalfCheetah
register(
    id="GymHalfCheetah-v0",
    entry_point="envs.gymnasium_half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)



