from warnings import simplefilter as _simplefilter

from gymnasium import register as _register

from .constants import objective_counts, state_norm_params

_simplefilter("ignore", DeprecationWarning, 512)  # Hopper-v2 / v3 warning

_register(
    id="MO-Ant-v2",
    entry_point="environments.ant:AntEnv",
    max_episode_steps=500,
)

_register(
    id="MO-Hopper-v2",
    entry_point="environments.hopper:HopperEnv",
    max_episode_steps=500,
)

_register(
    id="MO-Hopper-v3",
    entry_point="environments.hopper_v3:HopperEnv",
    max_episode_steps=500,
)

_register(
    id="MO-HalfCheetah-v2",
    entry_point="environments.half_cheetah:HalfCheetahEnv",
    max_episode_steps=500,
)

_register(
    id="MO-Walker2d-v2",
    entry_point="environments.walker2d:Walker2dEnv",
    max_episode_steps=500,
)

_register(
    id="MO-Swimmer-v2",
    entry_point="environments.swimmer:SwimmerEnv",
    max_episode_steps=500,
)

__all__ = ["objective_counts", "state_norm_params"]
