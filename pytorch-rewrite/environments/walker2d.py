# Walker2d-v2 env
# two objectives
# running speed, energy efficiency

from pathlib import Path

import gymnasium.spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


class Walker2dEnv(MujocoEnv, EzPickle):
    def __init__(self):
        self.obj_dim = 2
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "assets" / "walker2d.xml"),
            frame_skip=4,
            observation_space=gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            ),
        )
        EzPickle.__init__(self)

    def step(self, action):
        posbefore = self.data.qpos[0]
        a = np.clip(action, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[:3]
        alive_bonus = 1.0
        reward_speed = (posafter - posbefore) / self.dt + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(a).sum() + alive_bonus
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        ob = self._get_obs()

        return ob, 0.0, done, False, {"obj": np.array([reward_speed, reward_energy])}

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return self._get_obs()


if __name__ == "__main__":
    print(f"{Walker2dEnv().dt =}")
