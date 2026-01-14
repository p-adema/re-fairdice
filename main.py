import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch.cuda
import tqdm
from dataset import Buffer
from environments import objective_counts, state_norm_params
from evaluation import evaluate_policy
from fairdice import FairDICE


def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner",
        type=str,
        choices=["FairDICE"],
        default="FairDICE",
        help="Learner type",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=1.0, help="beta hyperparameter")
    parser.add_argument(
        "--divergence",
        type=str,
        choices=["KL", "CHI", "SOFT_CHI", "DUAL_DICE"],
        default="SOFT_CHI",
        help="Divergence type",
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
        "--hidden_dim", type=int, default=768, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of layers in the network"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for the policy"
    )
    parser.add_argument(
        "--layer_norm", type=bool, default=True, help="Use layer normalization if set"
    )
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument(
        "--policy_lr", type=float, default=3e-4, help="Policy learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset base directory"
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
        default=True,
        help="Whether to normalize reward",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        choices=[
            "MO-Hopper-v2",
            "MO-Hopper-v3",
            "MO-Ant-v2",
            "MO-HalfCheetah-v2",
            "MO-Swimmer-v2",
            "MO-Walker2d-v2",
        ],
        default="MO-Hopper-v2",
        help="Environment name",
    )
    parser.add_argument(
        "--total_train_steps", type=int, default=100_000, help="Total training steps"
    )
    parser.add_argument("--log_interval", type=int, default=10_000, help="Log interval")
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
        help="Path to save results and the model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    config = SimpleNamespace(**vars(args))
    env = gymnasium.make(config.env_name)
    example_obs, _ = env.reset()

    config.HIDDEN_DIMS = [config.hidden_dim] * config.num_layers
    config.STATE_DIM = env.observation_space.shape[0]
    config.ACTION_DIM = env.action_space.shape[0]
    config.REWARD_DIM = objective_counts[config.env_name]
    config.STATE_MEAN = state_norm_params[config.env_name]["mean"]
    config.STATE_STD = np.sqrt(state_norm_params[config.env_name]["var"])
    config.ACTION_BIAS = (env.action_space.high + env.action_space.low) / 2.0
    config.ACTION_SCALE = (env.action_space.high - env.action_space.low) / 2.0

    buffer = Buffer(
        args.data_dir,
        args.env_name,
        args.quality,
        args.preference_dist,
    )
    print("Initialising...")
    config.REWARD_MIN, config.REWARD_MAX = buffer.normalise(
        args.normalize_reward,
        (config.STATE_MEAN, config.STATE_STD),
        (config.ACTION_BIAS, config.ACTION_SCALE),
    )

    model_cls = {"FairDICE": FairDICE}[config.learner]

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_"
        f"{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
    )
    out_dir = Path(config.save_path, run_name)
    print("Saving to", out_dir)
    (out_dir / "logs").mkdir(parents=True)
    csv = (out_dir / "logs" / "stats.csv").open("w")
    csv.write("iteration,steps,nash,utilitarian\n")


    model = model_cls(config)
    model.to("cuda" if torch.cuda.is_available() else "cpu").train()
    torch.autograd.set_detect_anomaly(True)

    print("Compiling...")
    model.step(buffer.sample(config.batch_size))

    bar = tqdm.tqdm(
        iterable=range(1, config.total_train_steps),
        desc="Training",
        unit="batches",
        smoothing=0,
        initial=1,
        total=config.total_train_steps,
    )
    for it in bar:
        if config.log_interval and it % config.log_interval == 0:
            steps, nash, utilitarian = evaluate_policy(
                config=config,
                policy=model,
                env=env,
                save_dir=out_dir / "logs",
                num_episodes=config.eval_episodes,
                max_steps=config.max_seq_len,
                t_env=it,
                env_seed=config.seed,
            )
            model.train()
            csv.write(f"{it},{steps},{nash},{utilitarian}\n")
            csv.flush()
            bar.set_postfix_str(f"nsw={nash:.2f}, usw={utilitarian:.2f}")

        batch = buffer.sample(config.batch_size)
        model.step(batch)



    steps, nash, utilitarian = evaluate_policy(
        config=config,
        policy=model,
        env=env,
        save_dir=out_dir / "eval",
        num_episodes=config.eval_episodes,
        max_steps=config.max_seq_len,
        t_env=config.total_train_steps,
        env_seed=config.seed,
    )
    csv.write(f"{config.total_train_steps},{steps},{nash},{utilitarian}\n")
    csv.close()
    model.save(out_dir / "model.pt")
    end_time = datetime.now()
    print(f"Run complete: took {end_time - start_time}")
    if torch.cuda.is_available():
        print(f"Max CUDA VRAM use: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
