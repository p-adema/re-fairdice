import argparse
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", type=str, default="limodice", help="Learner type")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=0.001, help="beta hyperparameter")
    parser.add_argument(
        "--divergence",
        type=str,
        default="SOFT_CHI",
        help="Divergence type (SOFT_CHI/CHI/KL)",
    )
    parser.add_argument(
        "--gradient_penalty_coeff",
        type=float,
        default=1e-4,
        help="Gradient penalty coefficient",
    )
    parser.add_argument(
        "--tanh_squash_distribution",
        type=bool,
        default=False,
        help="Use tanh-squash distribution for actions if set",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the network"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for the policy"
    )
    parser.add_argument(
        "--layer_norm", type=bool, default=True, help="Use layer normalization if set"
    )
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument(
        "--policy_lr", type=float, default=3e-4, help="Policy learning rate"
    )
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
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
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=500,
        help="Max sequence length in trajectories",
    )
    parser.add_argument(
        "--normalize_reward",
        type=bool,
        default=False,
        help="Whether to normalize reward",
    )
    parser.add_argument(
        "--env_name", type=str, default="MO-Hopper-v2", help="Environment name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Running mode: 'train' or 'eval'",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint (for eval mode).",
    )
    parser.add_argument(
        "--total_train_steps", type=int, default=100_000, help="Total training steps"
    )
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval")
    parser.add_argument(
        "--eval_episodes", type=int, default=10, help="Evaluation episodes"
    )
    parser.add_argument(
        "--wandb", type=bool, default=False, help="Use wandb for logging"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save the model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")

    args = parser.parse_args()
    config = SimpleNamespace(**vars(args))
    data_path = f"./data/{config.env_name}/{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl"
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
        print("Loaded trajectories from", data_path)

    env = gym.make(config.env_name)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.reward_dim = env.obj_dim
    config.state_mean = state_norm_params[config.env_name]["mean"]
    config.state_std = np.sqrt(state_norm_params[config.env_name]["var"])
    config.ACTION_HIGH = env.action_space.high
    config.ACTION_LOW = env.action_space.low
    config.ACTION_SCALE = (config.ACTION_HIGH - config.ACTION_LOW) / 2.0
    config.ACTION_BIAS = (config.ACTION_HIGH + config.ACTION_LOW) / 2.0

    reward_min, reward_max = None, None
    for traj in trajs:
        r = traj["raw_rewards"]

        r_min = r.min(axis=0)
        r_max = r.max(axis=0)

        if reward_min is None:
            reward_min, reward_max = r_min, r_max
        else:
            reward_min = np.minimum(reward_min, r_min)
            reward_max = np.maximum(reward_max, r_max)
    config.reward_min = reward_min
    config.reward_max = reward_max

    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(
                traj["raw_rewards"], reward_min, reward_max
            )
        else:
            traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(
            traj["observations"], config.state_mean, config.state_std
        )
        traj["next_states"] = normalization(
            traj["next_observations"], config.state_mean, config.state_std
        )
        traj["actions"] = (traj["actions"] - config.ACTION_BIAS) / config.ACTION_SCALE
        traj["init_observations"] = np.tile(
            traj["observations"][0], (traj["observations"].shape[0], 1)
        )
        traj["init_states"] = np.tile(traj["states"][0], (traj["states"].shape[0], 1))

    tmp = defaultdict(list)

    for traj in trajs:
        for key, value in traj.items():
            tmp[key].append(value)

    batch = defaultdict(list)

    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0)

    for key, value in batch.items():
        print(key, value.shape)

    config.hidden_dims = [config.hidden_dim] * config.num_layers

    time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    run_name = f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
