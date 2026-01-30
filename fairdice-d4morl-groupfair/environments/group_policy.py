from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import gym
import tqdm
from gym import spaces
import pickle


class GroupPolicyEnv(gym.Env):
    def __init__(self, seed=42):
        super().__init__()

        self.num_groups = 5
        self.num_policies = 7
        self.groups_per_individual = 3
        self.num_individuals = 100
        self.gamma = 0.95
        self.max_steps = 500

        self.action_space = spaces.Discrete(self.num_policies)

        # For each group, indicate the expected welfare for each policy
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_policies * self.num_groups,),
            dtype=np.float32,
        )

        self.obj_dim = self.num_individuals

        self.rng = np.random.RandomState(seed)

        self.memberships = self.rng.choice(
            self.num_groups, size=(self.num_individuals, self.groups_per_individual)
        )
        self.current_advantage = np.zeros(self.num_groups, dtype=np.float32)
        self.current_options = self._generate_policies()
        self.steps = 0

    def _generate_policies(self):
        # The options are randomly sampled from a Dirichlet distribution, where groups
        # which are currently ahead are more likely to be favoured by new policies.
        base_alphas = np.ones(self.num_groups, dtype=np.float32) + 1e-4
        advantage = np.tanh((self.current_advantage - self.current_advantage.mean()))
        scores = self.rng.dirichlet(base_alphas + advantage, size=self.num_policies)
        return scores + 1e-8

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_options = self._generate_policies()
        self.current_advantage[:] = 0
        self.steps = 0
        return self.current_options.reshape(-1)

    def step(self, action):
        if hasattr(action, "item"):
            action = action.item()
        action = int(action)

        policy = self.current_options[action]
        group_rewards = policy
        indiv_rewards = group_rewards[self.memberships].sum(1)
        # Groups which were rewarded gain an advantage in how likely policies are to favour them.
        self.current_advantage += group_rewards / 10
        self.current_options = self._generate_policies()

        self.steps += 1

        done = self.steps >= self.max_steps

        return (
            self.current_options.reshape(-1),
            indiv_rewards.sum(),
            done,
            {"obj": indiv_rewards},
        )

    def get_goals(self):
        return self.goals


def generate_group_policy_dataset(
    num_trajectories: int = 100,
    seed: int = 42,
    save_dir: str = "./data/MO-GroupPolicy-v1",
    quality: str = "amateur",
):
    env = GroupPolicyEnv(seed=seed)

    dataset = []

    for _ in tqdm.trange(num_trajectories, desc="Generating", unit="traj"):
        obs = env.reset()
        done = False

        traj_obs, traj_next_obs, traj_actions, traj_rewards = [], [], [], []
        traj_terminals, traj_timeouts = [], []
        while not done:
            if quality == "amateur":
                action = env.action_space.sample()
            elif quality == "expert":
                # Take action that locally is best for group 0
                policies = obs.reshape(env.num_policies, env.num_groups)
                action = policies[:, 0].argmax()
            else:
                raise ValueError(f"Invalid {quality=}")

            next_obs, reward, done, info = env.step(action)

            timeout = env.steps >= env.max_steps
            terminal = done and not timeout

            traj_obs.append(obs)
            traj_actions.append(action)
            traj_rewards.append(info["obj"])
            traj_next_obs.append(next_obs)
            traj_terminals.append(terminal)
            traj_timeouts.append(timeout)

            obs = next_obs

        raw_rewards = np.array(traj_rewards, dtype=np.float32)
        dataset.append(
            {
                "observations": np.array(traj_obs, dtype=np.float32),
                "actions": np.array(traj_actions, dtype=np.float32).reshape(-1, 1),
                "next_observations": np.array(traj_next_obs, dtype=np.float32),
                "raw_rewards": raw_rewards,
                "terminals": np.array(traj_terminals, dtype=bool),
                "preference": np.full_like(raw_rewards, 1 / raw_rewards.shape[1]),
            }
        )
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    with (save_path / f"MO-GroupPolicy-v1_50000_{quality}_uniform.pkl").open("wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    # Nash-focussed: 5.67, Random: 5.55, Single-focussed: 4.9
    e = GroupPolicyEnv()
    print("Memberships:", np.bincount(e.memberships.reshape(-1)))
    ns = []
    trg = np.bincount(e.memberships.reshape(-1)).argmax()
    # trg = 0
    for seed in range(100):
        do = False
        o = e.reset(seed)
        rew = 0
        prew = np.zeros(e.num_individuals, dtype=np.float32)

        while not do:
            # print(o.reshape(e.num_policies, e.num_groups).round(3))
            # print("NSW:", np.log(o.reshape(e.num_policies, e.num_groups)).sum(1).round(2))
            # print("ADV:", np.tanh((e.current_advantage - e.current_advantage.mean())).round(3))
            # a = int(input("> "))
            sco = o.reshape(e.num_policies, e.num_groups).round(3)[:, 0].copy()
            sco.sort()
            # print("CHO:", sco)
            # a = np.log(o.reshape(e.num_policies, e.num_groups)).sum(1).argmax()
            # a = e.action_space.sample()
            # a = o.reshape(e.num_policies, e.num_groups)[:, trg].argmax()
            # a = 0
            # a = np.log(o.reshape(e.num_policies, e.num_groups)).sum(1).argmin()
            # time.sleep(0.1)
            o, r, do, inf = e.step(a)
            prew += inf["obj"]
            rew += r
        # prew -= 30
        # prew /= 10
        # print(f"{seed}:", np.tanh((e.current_advantage - e.current_advantage.mean())).round(3), e.current_advantage.max().round(3), round(rew), np.log(prew).mean(), prew.max(), prew.min())
        print(f"{seed}:", np.log(prew).sum(), prew.max(), prew.min())
        ns.append(np.log(prew).sum())
        # print(inf)
    print("MEAN", np.mean(ns).round(3))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_trajectories", type=int, default=10_000)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--save_dir", type=str, default="./data/MO-GroupPolicy-v1")
    # parser.add_argument("--quality", type=str, default="amateur")
    # args = parser.parse_args()
    # generate_group_policy_dataset(
    #     args.num_trajectories, args.seed, args.save_dir, args.quality
    # )
