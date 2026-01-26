import argparse
import os
import pathlib
import sys
import warnings
from types import SimpleNamespace

import gym
import jax
import numpy as np
import tqdm
from environments.norm import state_norm_params
from utils import min_max_normalization, normalization


def evaluate_policy(
    config,
    policy,
    env,
    save_dir,
    num_episodes=3,
    max_steps=500,
    t_env=None,
):
    policy.eval()
    raw_returns = []
    normalized_returns = []

    @jax.jit
    def select_action(observation):
        dist = policy(observation)
        action = dist.mean()  # deterministic action
        return action.flatten()

    for iter in range(num_episodes):
        seed = config.seed * 1000 + iter
        env.seed(seed)
        state = env.reset()
        if num_episodes == 1:
            print(seed, state)
        done = False
        steps = 0
        raw_rewards_list = []
        normalized_rewards_list = []

        while not done and steps < max_steps:
            s_t = normalization(state, config.state_mean, config.state_std)
            action = (
                select_action(s_t) * config.ACTION_SCALE + config.ACTION_BIAS
            ).astype(np.float32)
            state, _, done, info = env.step(action)

            raw_rewards = info["obj"]
            raw_rewards_list.append(raw_rewards)
            if config.normalize_reward:
                normalized_rewards = min_max_normalization(
                    raw_rewards, config.reward_min, config.reward_max
                )
            else:
                normalized_rewards = raw_rewards
            normalized_rewards_list.append(normalized_rewards)

            steps += 1

        raw_returns.append(np.sum(raw_rewards_list, axis=0))
        normalized_returns.append(np.sum(normalized_rewards_list, axis=0))

    if num_episodes != 1:
        np.save(os.path.join(save_dir, f"raw_returns_step_{t_env}.npy"), raw_returns)
        np.save(
            os.path.join(save_dir, f"normalized_returns_step_{t_env}.npy"),
            normalized_returns,
        )


def single_re_eval(config: SimpleNamespace):
    # Required to load the model
    config.policy_lr = 0.1
    config.nu_lr = 0.1
    config.mu_lr = 0.1
    config.total_train_steps = 1
    config.normalize_reward = True

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

    reward_norms = {
        ("MO-Hopper-v2", "amateur"): (
            [-6.607636451721191, -9.858153343200684],
            [12.189576148986816, 20.59270668029785],
        ),
        ("MO-Hopper-v2", "expert"): (
            [0.4883827269077301, -5.628507614135742],
            [11.69115161895752, 20.523406982421875],
        ),
        ("MO-Walker2d-v2", "amateur"): (
            [-0.40262269973754883, -1.0],
            [7.001187801361084, 4.999591827392578],
        ),
        ("MO-Walker2d-v2", "expert"): (
            [-0.3674027919769287, -0.5800488591194153],
            [7.1373162269592285, 4.996114253997803],
        ),
        ("MO-Swimmer-v2", "amateur"): (
            [-4.515071868896484, 0.0],
            [3.9241249561309814, 0.30000001192092896],
        ),
        ("MO-Swimmer-v2", "expert"): (
            [-1.216691493988037, 0.0],
            [1.7002224922180176, 0.30000001192092896],
        ),
        ("MO-HalfCheetah-v2", "amateur"): (
            [-0.6353333592414856, -1.0],
            [5.0, 4.999977111816406],
        ),
        ("MO-HalfCheetah-v2", "expert"): (
            [0.11494450271129608, -0.41102534532546997],
            [5.0, 4.999918460845947],
        ),
        ("MO-Ant-v2", "amateur"): (
            [-3.2283928394317627, -2.9306795597076416],
            [7.729767799377441, 7.630065441131592],
        ),
        ("MO-Ant-v2", "expert"): (
            [-2.248424768447876, -2.5920214653015137],
            [7.633843421936035, 7.382970809936523],
        ),
        ("MO-Hopper-v3", "amateur"): (
            [-3.879704713821411, -10.074402809143066, -19.0],
            [13.506857872009277, 20.417827606201172, 4.999999523162842],
        ),
        ("MO-Hopper-v3", "expert"): (
            [-0.27443230152130127, -6.7318315505981445, -19.0],
            [11.846762657165527, 20.688718795776367, 4.999999046325684],
        ),
    }

    config.reward_min = np.asarray(reward_norms[config.env_name, config.quality][0])
    config.reward_max = np.asarray(reward_norms[config.env_name, config.quality][1])

    config.hidden_dims = [config.hidden_dim] * config.num_layers

    if config.learner == "FairDICE":
        from FairDICE import get_model, load_model
    else:
        raise ValueError("Invalid learner type.")

    path = pathlib.Path(config.load_path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {config.load_path =}")

    save_dir = path.parent / f"re-eval-{config.eval_episodes}"
    save_dir.mkdir(exist_ok=True)
    if (save_dir / f"raw_returns_step_{config.seed}.npy").exists():
        print(f"Skipping {save_dir} / seed {config.seed} as it already exists.")
    else:
        model = load_model(str(path.resolve()), config)
        policy = get_model(model.policy_state)[0]
        evaluate_policy(
            config,
            policy,
            env,
            save_dir,
            num_episodes=config.eval_episodes,
            max_steps=config.max_seq_len,
            t_env=config.seed,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--learner", type=str, default="FairDICE")
        parser.add_argument(
            "--tanh_squash_distribution",
            type=bool,
            default=False,
        )
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--layer_norm", type=bool, default=True)
        parser.add_argument(
            "--quality",
            type=str,
            choices=["expert", "amateur"],
            default="expert",
        )
        parser.add_argument(
            "--preference_dist",
            type=str,
            choices=["uniform"],
            default="uniform",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=500,
        )
        parser.add_argument("--env_name", type=str, default="MO-Hopper-v2")
        parser.add_argument(
            "--load_path",
            type=str,
            required=True,
        )
        parser.add_argument("--eval_episodes", type=int, default=10)
        parser.add_argument("--seed", type=int, default=0, help="Random seed")

        args, unknown = parser.parse_known_args()
        configuration = SimpleNamespace(**vars(args))
        single_re_eval(configuration)
    else:
        # Orbax checkpoint loader warning, Hopper-v2 out of date warning
        warnings.simplefilter("ignore", UserWarning, 1175)
        warnings.simplefilter("ignore", UserWarning, 505)

        manifests = pathlib.Path.cwd().glob("*/*_FairDICE_*/model/manifest.ocdbt")
        runs = [p.parent.parent for p in manifests]
        for run in tqdm.tqdm(runs, desc="Re-evaluating models..."):
            run: pathlib.Path
            _, _, lrn, env_n, quality, _, _, _, _, seedname = str(run.name).split("_")
            assert lrn in ("FairDICE",), f"Strange {lrn=}"
            assert env_n.startswith("MO-"), f"Strange {env_n=}"
            assert quality in ("expert", "amateur"), f"Strange {quality=}"
            assert seedname.startswith("seed"), f"Strange {seedname=}"
            seed = int(seedname.removeprefix("seed"))
            configuration = SimpleNamespace(
                learner=lrn,
                tanh_squash_distribution=False,
                hidden_dim=512 if env_n == "MO-Ant-v2" else 768,
                num_layers=4 if env_n == "MO-Hopper-v3" else 3,
                temperature=1.0,
                layer_norm=True,
                quality=quality,
                preference_dist="uniform",
                max_seq_len=500,
                env_name=env_n,
                load_path=str(run / "model"),
                eval_episodes=100,
                seed=seed,
            )
            single_re_eval(configuration)
