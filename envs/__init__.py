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
    id="GymAnt-v5",
    entry_point="envs.ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register Ant
register(
    id="GymAnt-v0",
    entry_point="envs.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register Gymnasium Ant
register(
    id="GymnasiumAnt-v0",
    entry_point="envs.gymnasium_ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register Gymnasium HalfCheetah
register(
    id="GymnasiumHalfCheetah-v0",
    entry_point="envs.gymnasium_half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="AntNagabandi-v0",
    entry_point="envs.ant_nagabandi:AntNagabandiEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register Faithful Nagabandi Ant
register(
    id="FaithfulAnt-v0",
    entry_point="envs.faithful_ant:FaithfulAntEnv",
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


register(
    id="GymPusher-v0",
    entry_point="envs.pusher:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)
