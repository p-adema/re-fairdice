import gym
import numpy as np

import mo_gymnasium as mo_gym



class MinecartRewardWrapper(gym.Wrapper):
    """
    Reward structure:
    - Index 0: Ore type 1 collected (sparse, [0, 1.5])
    - Index 1: Ore type 2 collected (sparse, [0, 1.5])  
    - Index 2: Fuel consumed (dense, [-1, 0] -> shifted to [0, 1])
    """
    FUEL_SHIFT = 1.0 
    
    def __init__(self, env):
        super().__init__(env)
        self.obj_dim = 3
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shifted_reward = np.array(reward, dtype=np.float32)
        shifted_reward[2] += self.FUEL_SHIFT  # Fuel: [-1,0] -> [0,1]
        
        info['raw_reward'] = reward
        info['obj'] = shifted_reward
        
        scalar_reward = float(np.sum(shifted_reward))
        
        # gym vs gymnasium API differences
        if isinstance(terminated, bool):
            done = terminated or truncated
            return obs, scalar_reward, done, info
        else:
            return obs, scalar_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else {}
        return result


class MinecartGymAPIWrapper(gym.Wrapper):
    """
    Wrapper to convert gymnasium API to gym API for compatibility.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.obj_dim = 3
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Shift fuel consumption reward (originally ranges from -1 to 0)
        shifted_reward = np.array(reward, dtype=np.float32)
        shifted_reward[2] += MinecartRewardWrapper.FUEL_SHIFT
        
        info['raw_reward'] = reward
        info['obj'] = shifted_reward
        
        scalar_reward = float(np.sum(shifted_reward))
        return obs, scalar_reward, done, info
    
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            result = self.env.reset(seed=seed, **kwargs)
        else:
            result = self.env.reset(**kwargs)
        
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        return self.env.close()


def make_minecart(deterministic=False):
    env_name = "minecart-deterministic-v0" if deterministic else "minecart-v0"
    env = mo_gym.make(env_name)
    return MinecartGymAPIWrapper(env)


def make_minecart_deterministic():
    return make_minecart(deterministic=True)


def make_minecart_rgb():
    """    
    Returns 480x480x3 images
    """

    env = mo_gym.make("minecart-rgb-v0")
    return MinecartGymAPIWrapper(env)
