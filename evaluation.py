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
    env: gymnasium.vector.VectorEnv,
    save_dir: str | Path,
    max_steps: int = 500,
    t_env: int = -1,
    env_seed: int = 42,
) -> tuple[float, float, float]:
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    dones = np.zeros((env.num_envs, max_steps), dtype=np.bool)
    rew_min = config.REWARD_MIN.numpy(force=True).reshape(-1)
    rew_max = config.REWARD_MAX.numpy(force=True).reshape(-1)

    state, _ = env.reset(seed=env_seed)
    raw_rewards = np.full((env.num_envs, max_steps, len(rew_max)), np.nan)
    has_completed = np.zeros(env.num_envs, dtype=np.bool)

    for step in range(max_steps):
        # print(f"{state.shape=}, {config.STATE_MEAN=} {config.STATE_STD=}")
        s_t = (state - config.STATE_MEAN) / config.STATE_STD
        action_dist = policy(torch.asarray(s_t, device=device, dtype=torch.float32))
        action = (
            action_dist.mean.numpy(force=True) * config.ACTION_SCALE
            + config.ACTION_BIAS
        )
        state, _rew, term, trunc, info = env.step(action)
        dones[:, step] = term | trunc
        has_completed |= term | trunc
        raw_rewards[:, step] = info["obj"]
        if has_completed.all():
            break

    steps = dones.argmax(1)
    valid = np.arange(max_steps).reshape(1, -1) <= steps.reshape(-1, 1)
    raw_rewards[~valid] = 0
    raw_returns = raw_rewards.sum(1)

    normalized_rewards = (raw_rewards - rew_min) / (rew_max - rew_min)
    normalized_rewards[~valid] = 0
    normalized_returns = normalized_rewards.sum(1)

    avg_steps = steps.mean().item()
    avg_normalized_nsw_score = np.log(normalized_returns).sum(1).mean().item()
    avg_normalized_usw_score = np.sum(normalized_returns, axis=1).mean().item()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / f"raw_returns_step_{t_env}.npy", raw_returns)
    np.save(save_dir / f"normalized_returns_step_{t_env}.npy", normalized_returns)
    np.save(save_dir / f"steps_step_{t_env}.npy", steps)

    return avg_steps, avg_normalized_nsw_score, avg_normalized_usw_score
