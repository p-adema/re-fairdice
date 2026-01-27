from gym.envs.registration import register

register(
    id = 'MO-Ant-v2',
    entry_point = 'environments.ant:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'environments.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = 'environments.hopper_v3:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = 'environments.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = 'environments.walker2d:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = 'environments.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='MO-FourRooms-v0',
    entry_point='environments.four_rooms:MOFourRoomsEnv',
    max_episode_steps=50,
)

register(
    id='MO-FourRooms-v1',
    entry_point='environments.four_rooms:MOFourRoomsEnv',
    max_episode_steps=50,
)

register(
    id='MO-RandomMOMDP-v0',
    entry_point='environments.momdp:MOMDPEnv',
    max_episode_steps=50,
)

register(
    id='MO-Minecart-v0',
    entry_point='environments.minecart_wrapper:make_minecart',
    max_episode_steps=1000,
)

register(
    id='MO-Minecart-Deterministic-v0',
    entry_point='environments.minecart_wrapper:make_minecart_deterministic',
    max_episode_steps=1000,
)

register(
    id='MO-Minecart-RGB-v0',
    entry_point='environments.minecart_wrapper:make_minecart_rgb',
    max_episode_steps=1000,
)