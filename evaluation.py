from pathlib import Path
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
from torch import nn


@torch.no_grad()
def evaluate_policy(
    config: SimpleNamespace,
    policy: nn.Module,
    env: gymnasium.Env,
    save_dir: str | Path,
    num_episodes: int = 10,
    max_steps: int = 500,
    t_env: int = -1,
    env_seed: int = 42,
) -> tuple[float, float, float]:
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    steps_list = []
    raw_returns = []
    normalized_returns = []
    rew_min = config.REWARD_MIN.numpy(force=True)
    rew_max = config.REWARD_MAX.numpy(force=True)

    for seed in np.random.SeedSequence(entropy=env_seed).generate_state(num_episodes):
        state, _ = env.reset(seed=seed.item())
        done = False
        steps = 0
        raw_rewards_list = []
        normalized_rewards_list = []

        while not done and steps < max_steps:
            # print(f"{state.shape=}, {config.STATE_MEAN=} {config.STATE_STD=}")
            s_t = (state - config.STATE_MEAN) / config.STATE_STD
            action_dist = policy(torch.asarray(s_t, device=device, dtype=torch.float32))
            action = (
                action_dist.mean.view(-1).numpy(force=True) * config.ACTION_SCALE
                + config.ACTION_BIAS
            )
            state, _, term, trunc, info = env.step(action)
            done = term or trunc

            raw_rewards = info["obj"]
            raw_rewards_list.append(raw_rewards)
            normalized_rewards = (raw_rewards - rew_min) / (rew_max - rew_min)
            normalized_rewards_list.append(normalized_rewards)

            steps += 1

        steps_list.append(steps)
        raw_returns.append(np.sum(raw_rewards_list, axis=0))
        normalized_returns.append(np.sum(normalized_rewards_list, axis=0))

    avg_steps = np.mean(steps_list).item()
    avg_normalized_nsw_score = np.log(normalized_returns).sum(1).mean().item()
    avg_normalized_usw_score = np.sum(normalized_returns, axis=1).mean()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"raw_returns_step_{t_env}.npy", raw_returns)
    np.save(save_dir / f"normalized_returns_step_{t_env}.npy", normalized_returns)
    np.save(save_dir / f"steps_step_{t_env}.npy", steps_list)

    return avg_steps, avg_normalized_nsw_score, avg_normalized_usw_score
