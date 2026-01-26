# Swimmer-v2 env
# two objectives
# forward speed, energy efficiency

from pathlib import Path

import gymnasium.spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


class SwimmerEnv(MujocoEnv, EzPickle):
    def __init__(self):
        self.obj_dim = 2
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "assets" / "swimmer.xml"),
            frame_skip=4,
            observation_space=gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            ),
        )
        EzPickle.__init__(self)

    def step(self, action):
        ctrl_cost_coeff = 0.15
        xposbefore = self.data.qpos[0]
        a = np.clip(action, -1, 1)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = 0.3 - ctrl_cost_coeff * np.square(a).sum()
        ob = self._get_obs()
        return ob, 0.0, False, False, {"obj": np.array([reward_fwd, reward_ctrl])}

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return self._get_obs()


if __name__ == "__main__":
    print(f"{SwimmerEnv().dt =}")
