# Ant-v2 env
# two objectives
# x-axis speed, y-axis speed

from pathlib import Path

import gymnasium.spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


class AntEnv(MujocoEnv, EzPickle):
    def __init__(self):
        self.obj_dim = 2
        self.cost_weights = np.ones(self.obj_dim) / self.obj_dim
        MujocoEnv.__init__(
            self,
            model_path=str(Path(__file__).parent / "assets" / "ant.xml"),
            frame_skip=5,
            observation_space=gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
            ),
        )
        EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        a = np.clip(action, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        ctrl_cost = 0.5 * np.square(a).sum()
        survive_reward = 1.0
        other_reward = -ctrl_cost + survive_reward

        vx_reward = (xposafter - xposbefore) / self.dt + other_reward
        vy_reward = (yposafter - yposbefore) / self.dt + other_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all()
        done = not notdone
        ob = self._get_obs()
        return ob, 0, done, False, {"obj": np.array([vx_reward, vy_reward])}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat[2:], self.data.qvel.flat])

    def reset_model(self):
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + c * self.np_random.standard_normal(self.model.nv),
        )
        return self._get_obs()


if __name__ == "__main__":
    print(f"{AntEnv().dt =}")
