# Hopper-v2 env
# two objectives
# running speed, jumping height

from pathlib import Path

import gymnasium.spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


class HopperEnv(MujocoEnv, EzPickle):
    def __init__(self):
        self.obj_dim = 2
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "assets" / "hopper.xml"),
            frame_skip=5,
            observation_space=gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
            ),
        )
        EzPickle.__init__(self)

    def step(self, action):
        posbefore = self.data.qpos[0]
        a = np.clip(action, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(a, self.frame_skip)
        posafter, height, _ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward_others = alive_bonus - 2e-4 * np.square(a).sum()
        reward_run = 1.5 * (posafter - posbefore) / self.dt + reward_others
        reward_jump = 12.0 * (height - self.init_qpos[1]) + reward_others
        s = self.state_vector()
        done = not (
            (s[1] > 0.4)
            and abs(s[2]) < np.deg2rad(90)
            and abs(s[3]) < np.deg2rad(90)
            and abs(s[4]) < np.deg2rad(90)
            and abs(s[5]) < np.deg2rad(90)
        )

        ob = self._get_obs()
        return ob, 0.0, done, False, {"obj": np.array([reward_run, reward_jump])}

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos.flat[1:], np.clip(self.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        c = 1e-3
        new_qpos = self.init_qpos + self.np_random.uniform(
            low=-c, high=c, size=self.model.nq
        )
        new_qpos[1] = self.init_qpos[1]
        new_qvel = self.init_qvel + self.np_random.uniform(
            low=-c, high=c, size=self.model.nv
        )
        self.set_state(new_qpos, new_qvel)
        return self._get_obs()


if __name__ == "__main__":
    print(f"{HopperEnv().dt =}")
