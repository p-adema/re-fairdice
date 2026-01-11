# HalfCheetah-v2 env
# two objectives
# running speed, energy efficiency

from pathlib import Path

import gymnasium.spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


class HalfCheetahEnv(MujocoEnv, EzPickle):
    def __init__(self):
        self.obj_dim = 2
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "assets" / "half_cheetah.xml"),
            frame_skip=5,
            observation_space=gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            ),
        )
        EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        xposafter, ang = self.data.qpos[0], self.data.qpos[2]
        ob = self._get_obs()
        alive_bonus = 1.0

        reward_run = (xposafter - xposbefore) / self.dt
        reward_run = min(4.0, reward_run) + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(action).sum() + alive_bonus

        done = not (abs(ang) < np.deg2rad(50))
        return ob, 0.0, done, False, {"obj": np.array([reward_run, reward_energy])}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat[1:], self.data.qvel.flat])

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + c * self.np_random.standard_normal(self.model.nv),
        )
        return self._get_obs()


if __name__ == "__main__":
    print(f"{HalfCheetahEnv().dt =}")
