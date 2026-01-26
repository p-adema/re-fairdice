import argparse
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium
import torch.cuda
import tqdm
from dataset import Buffer
from environments import objective_counts, state_norm_params
from evaluation import evaluate_policy
from fairdice import FairDICE

# These models are so small that full precision doesn't even cause slowdown.
warnings.filterwarnings("ignore", "TensorFloat32", UserWarning)

def main():
    start_time = datetime.now()
    config = parse_args()
    env = gymnasium.make_vec(config.env_name, config.eval_episodes)
    example_obs, _ = env.reset()

    config.HIDDEN_DIMS = [config.hidden_dim] * config.num_layers
    config.STATE_DIM = env.single_observation_space.shape[0]
    config.ACTION_DIM = env.single_action_space.shape[0]
    config.REWARD_DIM = objective_counts[config.env_name]
    config.STATE_MEAN = torch.asarray(state_norm_params[config.env_name]["mean"])
    config.STATE_STD = torch.asarray(state_norm_params[config.env_name]["var"]).sqrt()
    act_high, act_low = env.single_action_space.high, env.single_action_space.low
    config.ACTION_BIAS = torch.asarray(act_high + act_low) / 2.0
    config.ACTION_SCALE = torch.asarray(act_high - act_low) / 2.0

    torch.manual_seed(config.seed)
    buffer = Buffer(
        config.data_dir,
        config.env_name,
        config.quality,
        config.preference_dist,
    )
    config.REWARD_MIN, config.REWARD_MAX = buffer.normalise(
        config.normalize_reward,
        (config.STATE_MEAN, config.STATE_STD),
        (config.ACTION_BIAS, config.ACTION_SCALE),
    )

    print(f"Loaded: {len(buffer) // config.batch_size} batches of {config.batch_size}")
    print("Compiling...")

    model_cls = {"FairDICE": FairDICE}[config.learner]

    model = model_cls(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    model.step(buffer.sample(config.batch_size))
    with torch.no_grad():
        model(torch.asarray(example_obs, device=device, dtype=torch.float32))

    time_stamp = start_time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_"
        f"{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
    )
    out_dir = Path(config.save_path, run_name)
    print("Saving to", out_dir)
    (out_dir / "logs").mkdir(parents=True)
    csv = (out_dir / "logs" / "stats.csv").open("w")
    csv.write("iteration,steps,nash,utilitarian\n")
    bar = tqdm.tqdm(
        iterable=range(1, config.total_train_steps),
        desc="Training",
        unit="batches",
        smoothing=0,
        initial=1,
        total=config.total_train_steps,
    )

    for it in bar:
        batch = buffer.sample(config.batch_size)
        model.step(batch)

        if config.log_interval and it % config.log_interval == 0:
            steps, nash, utilitarian = evaluate_policy(
                config=config,
                policy=model,
                env=env,
                save_dir=out_dir / "logs",
                max_steps=config.max_seq_len,
                t_env=it,
                env_seed=config.seed,
            )
            model.train()
            csv.write(f"{it},{steps},{nash},{utilitarian}\n")
            csv.flush()
            bar.set_postfix_str(f"nsw={nash:.2f}, usw={utilitarian:.2f}")

    # Ensure the model is loadable
    model.save(out_dir / "model.pt")
    del model
    model = model_cls.load(out_dir / "model.pt").requires_grad_(False)

    steps, nash, utilitarian = evaluate_policy(
        config=config,
        policy=model,
        env=env,
        save_dir=out_dir / "eval",
        max_steps=config.max_seq_len,
        t_env=config.total_train_steps,
        env_seed=config.seed,
    )
    csv.write(f"{config.total_train_steps},{steps},{nash},{utilitarian}\n")
    csv.close()
    end_time = datetime.now()
    print(f"Run complete (nsw {nash:.2f}), took {end_time - start_time}")
    if torch.cuda.is_available():
        print(f"Max CUDA VRAM use: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def parse_args() -> SimpleNamespace:
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
        default=0,  # As described in the paper, without any penalty!
        help="Gradient penalty coefficient",
    )
    parser.add_argument(
        "--tanh_squash_distribution",
        type=bool,
        default=False,  # This doesn't work becuase actions aren't fixed to [-1, 1]
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
        "--eval_episodes", type=int, default=10, help="Final evaluation episodes"
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
    return config


if __name__ == "__main__":
    main()
