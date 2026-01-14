import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

import torch
from pyarrow.parquet import read_table as read_parquet


class Batch(NamedTuple):
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    is_valids: torch.Tensor
    init_states: torch.Tensor


class Buffer:
    def __init__(
        self,
        base_dir: str | Path,
        env_id: str,
        quality: str,
        preference_dist: str = "uniform",
    ):
        path = Path(base_dir, env_id, f"{env_id}_50000_{quality}_{preference_dist}")
        assert path.exists(), f"Missing dataset at {path =} (maybe run convert_data?)"
        self.path = path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading", path)

        def load_array(name: str, dtype: torch.dtype):
            return torch.asarray(
                np.asarray(read_parquet(f"{path}/{name}.pq")),
                dtype=dtype,
                device=device,
            )

        self.states = load_array("observations", torch.float32)
        self.next_states = load_array("next_observations", torch.float32)
        self.actions = load_array("actions", torch.float32)
        self.rewards = load_array("raw_rewards", torch.float32)
        self.is_valids = load_array("terminals", torch.bool).logical_not_().view(-1, 1)
        init_idxs = load_array("init_idxs", torch.long).view(len(self.states))
        self.init_states = self.states[init_idxs]

        self.size = len(self.states)
        self.position = 0
        self.shuffle()

    @staticmethod
    def _convert_or_compute(
        mean_std: tuple[npt.NDArray, npt.NDArray] | None, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mean_std is not None:
            mean = torch.asarray(mean_std[0]).to(reference)
            std = torch.asarray(mean_std[1]).to(reference)
        else:
            mean, std = reference.mean(0), reference.std(0)
        return mean, std

    def normalise(
        self,
        normalise_rewards: bool,
        state_mean_std: tuple[npt.NDArray, npt.NDArray] | None,
        action_mean_std: tuple[npt.NDArray, npt.NDArray] | None,
    ):
        state_mean, state_std = self._convert_or_compute(state_mean_std, self.states)
        self.states.sub_(state_mean).div_(state_std)
        self.next_states.sub_(state_mean).div_(state_std)
        self.init_states.sub_(state_mean).div_(state_std)

        act_mean, act_std = self._convert_or_compute(action_mean_std, self.actions)
        self.actions.sub_(act_mean).div_(act_std)

        reward_min, reward_max = self.rewards.amin(0), self.rewards.amax(0)
        if normalise_rewards:
            self.rewards.sub_(reward_min).div_(reward_max - reward_min)

        return reward_min, reward_max

    def shuffle(self):
        perm = torch.randperm(self.size)
        self.states = self.states[perm]
        self.next_states = self.next_states[perm]
        self.actions = self.actions[perm]
        self.rewards = self.rewards[perm]
        self.is_valids = self.is_valids[perm]
        self.init_states = self.init_states[perm]
        self.position = 0

    def sample(self, batch_size):
        assert batch_size < self.size
        end = self.position + batch_size
        if end >= self.size:
            self.shuffle()
            end = batch_size

        batch = Batch(
            self.states[self.position : end],
            self.next_states[self.position : end],
            self.actions[self.position : end],
            self.rewards[self.position : end],
            self.is_valids[self.position : end],
            self.init_states[self.position : end],
        )
        self.position = end
        return batch

    def __repr__(self):
        return f"Buffer(states={self.states}, ...)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset base directory"
    )
    parser.add_argument(
        "--env_name", type=str, default="MO-Hopper-v2", help="Environment name"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["expert", "amateur"],
        default="expert",
        help="Dataset quality",
    )
    parser.add_argument(
        "--preference_dist",
        type=str,
        choices=["uniform", "wide", "narrow"],
        default="uniform",
        help="Preference distribution",
    )
    args = parser.parse_args()
    print(f"Trying to load dataset for {args=}")
    buffer = Buffer(args.base_dir, args.env_name, args.quality, args.preference_dist)
    print(buffer)
    print("Load successful!")
    print(f"Sample: {buffer.sample(5)}")
    print("Shuffle:", buffer.shuffle())
    print(f"Sample 2: {buffer.sample(5)}")
