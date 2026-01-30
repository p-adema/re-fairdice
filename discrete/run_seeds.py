#!/usr/bin/env python
"""
Multi-seed training runner with aggregated heatmap visualization.

Runs the training pipeline directly (no subprocesses) across N seeds,
then aggregates all trained models into a single combined visitation heatmap.

Optimized for running 100-1000+ seeds efficiently.

Usage:
    python run_seeds.py --num_seeds 100 --beta 0.1 1.0 --total_train_steps 50000
"""

import os
import gc
import json
import glob
import gym
import environments
from environments.norm import state_norm_params
from environments.four_rooms import aggregate_policy_heatmaps, MOFourRoomsEnv, obs_to_onehot

import pickle
import numpy as np
from collections import defaultdict
from utils import normalization, min_max_normalization, normalize_rewards, social_welfare
from buffer import Buffer
import argparse
from types import SimpleNamespace
from evaluation import evaluate_policy
import random
from tqdm import tqdm
import jax
from datetime import datetime
import gym.spaces

import gymnasium.spaces as gymnasium_spaces

from FairDICE import init_train_state, train_step, get_model, save_model, load_model


def is_discrete_action_space(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return True
    if isinstance(action_space, gymnasium_spaces.Discrete):
        return True
    return False

def is_minecart_env(env_name):
    minecart_names = ['minecart', 'MO-Minecart']
    return any(name.lower() in env_name.lower() for name in minecart_names)


def is_rgb_env(env_name):
    return 'rgb' in env_name.lower()


def is_fourrooms_env(env_name):
    return 'fourrooms' in env_name.lower() or 'four-rooms' in env_name.lower()


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.h, self.w = 13, 13 
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.h * self.w,), dtype=np.float32
        )

    def observation(self, obs):
        y, x = int(obs[0]), int(obs[1])
        one_hot = np.zeros(self.h * self.w, dtype=np.float32)
        idx = y * self.w + x
        if 0 <= idx < len(one_hot):
            one_hot[idx] = 1.0
        return one_hot


def preprocess_dataset_to_onehot(trajs, h=13, w=13):
    new_dim = h * w
    for traj in trajs:
        N = traj['observations'].shape[0]
        new_obs = np.zeros((N, new_dim), dtype=np.float32)
        for i, obs in enumerate(traj['observations']):
            y, x = int(obs[0]), int(obs[1])
            idx = y * w + x
            if 0 <= idx < new_dim:
                new_obs[i, idx] = 1.0
        traj['observations'] = new_obs
        
        new_next_obs = np.zeros((N, new_dim), dtype=np.float32)
        for i, obs in enumerate(traj['next_observations']):
            y, x = int(obs[0]), int(obs[1])
            idx = y * w + x
            if 0 <= idx < new_dim:
                new_next_obs[i, idx] = 1.0
        traj['next_observations'] = new_next_obs
        traj["init_observations"] = np.tile(traj['observations'][0], (N, 1))
        
    return trajs, new_dim


def setup_config(args, beta):
    config = SimpleNamespace(
        alpha=args.alpha,
        learner=args.learner,
        lr=args.lr,
        gamma=args.gamma,
        beta=beta,
        divergence=args.divergence,
        gradient_penalty_coeff=args.gradient_penalty_coeff,
        tanh_squash_distribution=args.tanh_squash_distribution,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        temperature=args.temperature,
        layer_norm=args.layer_norm,
        nu_lr=args.nu_lr,
        policy_lr=args.policy_lr,
        mu_lr=args.mu_lr,
        batch_size=args.batch_size,
        quality=args.quality,
        preference_dist=args.preference_dist,
        max_seq_len=args.max_seq_len,
        normalize_reward=args.normalize_reward,
        reward_norm=args.reward_norm,
        ignore_fuel=args.ignore_fuel,
        env_name=args.env_name,
        total_train_steps=args.total_train_steps,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        wandb=False,
        save_path=args.save_path,
        seed=0,  # Will be overwritten later
        tag=args.tag,
    )
    config.hidden_dims = [config.hidden_dim] * config.num_layers
    return config


def compute_reward_stats(trajs):
    """Compute reward statistics needed for normalization."""
    reward_min, reward_max = None, None
    reward_sum = None
    reward_sq_sum = None
    reward_count = 0
    returns = []

    for traj in trajs:
        r = traj["raw_rewards"]
        r_min, r_max = r.min(axis=0), r.max(axis=0)
        if reward_min is None:
            reward_min, reward_max = r_min, r_max
        else:
            reward_min = np.minimum(reward_min, r_min)
            reward_max = np.maximum(reward_max, r_max)

        returns.append(r.sum(axis=0))
        reward_count += r.shape[0]
        reward_sum = r.sum(axis=0) if reward_sum is None else reward_sum + r.sum(axis=0)
        reward_sq_sum = (r ** 2).sum(axis=0) if reward_sq_sum is None else reward_sq_sum + (r ** 2).sum(axis=0)

    reward_mean = reward_sum / max(reward_count, 1)
    reward_var = reward_sq_sum / max(reward_count, 1) - reward_mean ** 2
    reward_std = np.sqrt(np.maximum(reward_var, 0.0))

    returns = np.array(returns) if returns else np.zeros((0, 0))
    return_mean = returns.mean(axis=0) if returns.size else np.zeros(0)
    reward_return_scale = 1.0 / np.maximum(return_mean, 1e-8)
    
    # Compute max-based scale (so max reward = 1 for each objective)
    reward_max_scale = 1.0 / np.maximum(reward_max, 1e-8)
    
    # Compute balanced scale: scale so total contributions are equal
    # Target: all objectives contribute equally to total reward
    max_return = np.max(return_mean) if return_mean.size else 1.0
    reward_balanced_scale = max_return / np.maximum(return_mean, 1e-8)

    return reward_min, reward_max, reward_mean, reward_std, reward_return_scale, reward_max_scale, reward_balanced_scale


def load_and_prepare_data(config):
    """Load dataset and prepare buffer (done once for all seeds)."""
    data_path = f"./data/{config.env_name}/{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl"
    if not os.path.exists(data_path):
        alt_path = f"./data/{config.env_name}/{config.env_name}_{config.quality}.pkl" 
        if os.path.exists(alt_path):
            data_path = alt_path
    
    if not os.path.exists(data_path):
        pattern = f"./data/{config.env_name}/{config.env_name}_*_{config.quality}_{config.preference_dist}.pkl"
        matches = glob.glob(pattern)
        if matches:
            data_path = matches[0]

    print(f"Loading trajectories from {data_path}")
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
    
    import copy
    trajs = copy.deepcopy(trajs)
    
    if is_minecart_env(config.env_name):
        from environments.minecart_wrapper import make_minecart, make_minecart_deterministic, make_minecart_rgb
        
        if is_rgb_env(config.env_name):
            env = make_minecart_rgb()
            config.use_cnn = True
            config.cnn_feature_dim = 256
            print("Preprocessing RGB observations (normalizing to [0, 1])...")
            for traj in trajs:
                traj['observations'] = traj['observations'].astype(np.float32) / 255.0
                traj['next_observations'] = traj['next_observations'].astype(np.float32) / 255.0
            obs_dim = trajs[0]['observations'].shape[1:]
            config.state_dim = obs_dim
        elif 'deterministic' in config.env_name.lower():
            env = make_minecart_deterministic()
            config.use_cnn = False
            config.state_dim = 7  # 7D vector observation
        else:
            env = make_minecart(deterministic=False)
            config.use_cnn = False
            config.state_dim = 7
    elif is_fourrooms_env(config.env_name):
        trajs, one_hot_dim = preprocess_dataset_to_onehot(trajs)
        env = gym.make(config.env_name)
        env = OneHotWrapper(env)
        config.state_dim = one_hot_dim
        config.use_cnn = False
    else:
        env = gym.make(config.env_name)
        config.state_dim = env.observation_space.shape[0]
        config.use_cnn = False
    
    if is_discrete_action_space(env.action_space):
        config.is_discrete = True
        config.action_dim = env.action_space.n
        config.ACTION_HIGH = None
        config.ACTION_LOW = None
        config.ACTION_SCALE = None
        config.ACTION_BIAS = None
    else:
        config.is_discrete = False
        config.action_dim = env.action_space.shape[0]
        config.ACTION_HIGH = env.action_space.high
        config.ACTION_LOW = env.action_space.low
        config.ACTION_SCALE = (config.ACTION_HIGH - config.ACTION_LOW) / 2.0
        config.ACTION_BIAS = (config.ACTION_HIGH + config.ACTION_LOW) / 2.0

    # Handle ignore_fuel: slice rewards to only use ore1 and ore2
    if config.ignore_fuel and is_minecart_env(config.env_name):
        print("\n*** IGNORING FUEL: Using only ore1 and ore2 (2D rewards) ***\n")
        config.reward_dim = 2
        # Slice the rewards in the data to only keep first 2 dimensions
        for traj in trajs:
            traj["raw_rewards"] = traj["raw_rewards"][:, :2]
    else:
        config.reward_dim = env.obj_dim

    if config.env_name in state_norm_params:
        config.state_mean = state_norm_params[config.env_name]["mean"]
        config.state_std = np.sqrt(state_norm_params[config.env_name]["var"])
    elif is_minecart_env(config.env_name):
        if is_rgb_env(config.env_name):
            config.state_mean = 0.0
            config.state_std = 1.0
        else:
            config.state_mean = np.zeros(7)
            config.state_std = np.ones(7)
    elif is_fourrooms_env(config.env_name):
        config.state_mean = np.zeros(13 * 13)
        config.state_std = np.ones(13 * 13)
    else:
        config.state_mean = np.zeros(config.state_dim)
        config.state_std = np.ones(config.state_dim)
    
    # Compute simple reward statistics
    reward_min, reward_max = None, None
    for traj in trajs:
        r = traj["raw_rewards"]
        r_min, r_max = r.min(axis=0), r.max(axis=0)
        if reward_min is None:
            reward_min, reward_max = r_min, r_max
        else:
            reward_min = np.minimum(reward_min, r_min)
            reward_max = np.maximum(reward_max, r_max)
    config.reward_min = reward_min
    config.reward_max = reward_max

    # Debug: print reward statistics
    print(f"\n{'='*60}")
    print("Dataset Reward Statistics:")
    print(f"  reward_dim:           {config.reward_dim}")
    print(f"  reward_min:           {reward_min}")
    print(f"  reward_max:           {reward_max}")
    print(f"  ignore_fuel:          {config.ignore_fuel}")
    
    # Count trajectories with ore
    n_with_ore = sum(1 for t in trajs if t['raw_rewards'][:, 0].sum() > 0 or t['raw_rewards'][:, 1].sum() > 0)
    print(f"  Trajectories with ore: {n_with_ore}/{len(trajs)} ({100*n_with_ore/len(trajs):.1f}%)")
    
    # Debug: action distribution in training data
    all_actions = np.concatenate([t['actions'].flatten() for t in trajs])
    action_counts = {}
    for a in all_actions:
        a_int = int(a)
        action_counts[a_int] = action_counts.get(a_int, 0) + 1
    print(f"  Action distribution in data:")
    total = len(all_actions)
    for a in sorted(action_counts.keys()):
        pct = 100 * action_counts[a] / total
        print(f"    action {a}: {action_counts[a]} ({pct:.1f}%)")
    
    # Check: what actions lead to ore rewards?
    ore_actions = []
    for t in trajs:
        for i, r in enumerate(t['raw_rewards']):
            if r[0] > 0 or r[1] > 0:
                ore_actions.append(int(t['actions'][i]))
    if ore_actions:
        ore_action_counts = {}
        for a in ore_actions:
            ore_action_counts[a] = ore_action_counts.get(a, 0) + 1
        print(f"  Actions that led to ore collection:")
        for a in sorted(ore_action_counts.keys()):
            pct = 100 * ore_action_counts[a] / len(ore_actions)
            print(f"    action {a}: {ore_action_counts[a]} ({pct:.1f}%)")
    else:
        print(f"  WARNING: No actions led to ore collection!")
    
    print(f"{'='*60}\n")

    use_cnn = getattr(config, 'use_cnn', False)
    
    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], config.reward_min, config.reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        
        if use_cnn:
            traj["states"] = traj['observations']
            traj['next_states'] = traj['next_observations']
        else:
            traj["states"] = normalization(traj['observations'], config.state_mean, config.state_std)
            traj['next_states'] = normalization(traj['next_observations'], config.state_mean, config.state_std)
        
        if not config.is_discrete:
            traj['actions'] = (traj['actions'] - config.ACTION_BIAS) / config.ACTION_SCALE
        else:
            traj['actions'] = traj['actions'].astype(np.float32)

        N = traj['observations'].shape[0]
        if use_cnn:
            traj["init_observations"] = np.tile(traj['observations'][0:1], (N, 1, 1, 1))
            traj["init_states"] = traj["init_observations"]
        else:
            traj["init_observations"] = np.tile(traj['observations'][0], (N, 1))
            traj["init_states"] = np.tile(traj['states'][0], (N, 1))

    # Flatten trajectories into batch
    tmp = defaultdict(list)
    for traj in trajs:
        for key, value in traj.items():
            tmp[key].append(value)

    batch = {}
    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0)

    return config, env, batch


def evaluate_policy_detailed(policy_fn, config, num_episodes=100):
    env = MOFourRoomsEnv()
    
    total_episodes = 0
    successful_episodes = 0
    goal_counts = [0, 0, 0]
    steps_list = []
    returns_per_goal = [[], [], []]
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = np.zeros(3)
        
        while not done:
            obs_onehot = obs_to_onehot(obs)
            action = policy_fn(obs_onehot)
            obs, reward, done, info = env.step(action)
            episode_reward += info['obj']
        
        total_episodes += 1
        steps_list.append(env.steps)
        
        if reward > 0:
            successful_episodes += 1
            for i in range(3):
                if info['obj'][i] > 0:
                    goal_counts[i] += 1
                    returns_per_goal[i].append(1.0)
    
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    goal_distribution = [g / successful_episodes if successful_episodes > 0 else 0 for g in goal_counts]
    
    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
        "goal_counts": goal_counts,
        "goal_distribution": goal_distribution,
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
    }


def train_single_seed(config, env, batch, seed, save_dir):
    """Train a single seed and return the best model path and statistics."""
    config.seed = seed
    
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    
    train_state = init_train_state(config)
    train_carry = (train_state, key)
    buffer = Buffer(batch)
    
    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step(config, train_state, data, subkey)
        return (train_state, key), update_info
    
    train_iterations = config.total_train_steps // config.log_interval
    best_welfare = -float('inf')
    best_model_path = os.path.abspath(save_dir + "/model_best")
    
    training_history = []
    best_returns = None
    
    for iter in range(train_iterations):
        step = (iter + 1) * config.log_interval
        train_carry, update_info = jax.lax.scan(train_body, train_carry, length=config.log_interval)
        
        policy = get_model(train_carry[0].policy_state)[0]
        avg_returns, _, avg_steps, raw_returns = evaluate_policy(
            config, policy, env,
            save_dir + "/eval",
            num_episodes=config.eval_episodes,
            max_steps=config.max_seq_len,
            t_env=step,
            return_episode_returns=True
        )
        
        pop_welfare = social_welfare(avg_returns, config.alpha)
        returns_array = np.array(raw_returns)
        returns_mean = returns_array.mean(axis=0)
        returns_std = returns_array.std(axis=0)
        returns_min = returns_array.min(axis=0)
        returns_max = returns_array.max(axis=0)

        # Get mu values from update_info (last step of the scan)
        mu_values = np.array(update_info["mu"][-1]) if "mu" in update_info else None
        w_mean = float(update_info["w_mean"][-1]) if "w_mean" in update_info else 0
        w_std = float(update_info["w_std"][-1]) if "w_std" in update_info else 0
        policy_loss = float(update_info["policy_loss"][-1]) if "policy_loss" in update_info else 0
        
        print(
            "Step {}: Returns={}  mu={}  Welfare={:.4f}  policy_loss={:.4f}  w_mean={:.4f}  w_std={:.4f}".format(
                step,
                returns_mean.tolist(),
                mu_values.tolist() if mu_values is not None else "N/A",
                float(pop_welfare),
                policy_loss,
                w_mean,
                w_std,
            )
        )
        
        training_history.append({
            "step": step,
            "avg_returns": avg_returns.tolist(),
            "returns_std": returns_std.tolist(),
            "returns_min": returns_min.tolist(),
            "returns_max": returns_max.tolist(),
            "welfare": float(pop_welfare),
            "alpha": config.alpha,
            "avg_steps": float(avg_steps),
        })
        
        if pop_welfare > best_welfare:
            best_welfare = pop_welfare
            best_returns = avg_returns.tolist()
            save_model(train_carry[0], best_model_path)
    
    # save_model(train_carry[0], os.path.abspath(save_dir + "/model"))
    
    with open(os.path.join(save_dir, "eval", "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    jax.clear_caches()
    gc.collect()
    
    return best_model_path, {
        "seed": seed,
        "best_welfare": float(best_welfare),
        "alpha": config.alpha,
        "best_returns": best_returns,
        "final_step": config.total_train_steps,
    }


def create_policy_fn(model_path, config):
    """Load a model and create a policy function."""
    train_state = load_model(model_path, config)
    policy_network = get_model(train_state.policy_state)[0]
    
    def policy_fn(obs_onehot):
        state = normalization(obs_onehot, config.state_mean, config.state_std)
        state = state.reshape(1, -1)
        dist = policy_network(state)
        action = dist.sample(seed=jax.random.PRNGKey(np.random.randint(0, 10000)))
        return int(action[0])
    
    return policy_fn


def aggregate_with_statistics(policy_fns, config, save_path, title, num_episodes_per_policy=100):
    """Aggregate heatmaps and collect detailed statistics."""
    env = MOFourRoomsEnv()
    layout = env.get_layout()
    visitation_counts = np.zeros(layout.shape)
    
    total_episodes = 0
    successful_episodes = 0
    goal_counts = [0, 0, 0]
    steps_all = []
    
    for policy_idx, policy_fn in enumerate(policy_fns):
        for _ in range(num_episodes_per_policy):
            obs = env.reset()
            done = False
            trajectory = []
            
            while not done:
                y, x = int(obs[0]), int(obs[1])
                trajectory.append((y, x))
                
                obs_onehot = obs_to_onehot(obs)
                action = policy_fn(obs_onehot)
                obs, reward, done, info = env.step(action)
            
            y, x = int(obs[0]), int(obs[1])
            trajectory.append((y, x))
            
            total_episodes += 1
            steps_all.append(env.steps)
            
            if reward > 0:
                successful_episodes += 1
                for i in range(3):
                    if info['obj'][i] > 0:
                        goal_counts[i] += 1
                
                for y, x in trajectory:
                    visitation_counts[y, x] += 1
        
        if (policy_idx + 1) % 10 == 0:
            print(f"  Processed {policy_idx + 1}/{len(policy_fns)} policies")
    
    # Compute statistics
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    goal_distribution = [g / successful_episodes if successful_episodes > 0 else 0 for g in goal_counts]
    
    stats = {
        "num_policies": len(policy_fns),
        "episodes_per_policy": num_episodes_per_policy,
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
        "goal_counts": goal_counts,
        "goal_distribution": goal_distribution,
        "avg_steps": float(np.mean(steps_all)),
        "std_steps": float(np.std(steps_all)),
        "total_visitations": int(np.sum(visitation_counts)),
    }
    
    print(f"Success rate: {success_rate:.2%}")
    print(f"Goal distribution: {goal_distribution}")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 8))
    
    mask = layout
    sns.heatmap(visitation_counts, mask=mask, cmap="Blues", cbar=True, square=True,
                linewidths=0.5, linecolor='gray')

    plt.imshow(layout, cmap="binary", alpha=0.3)
    
    colors = ['red', 'green', 'blue']
    for i, goal in enumerate(env.goals):
        pct = goal_distribution[i] * 100
        plt.text(goal[1]+0.5, goal[0]+0.5, f"G{i+1}\n{pct:.1f}%", color=colors[i], 
                 ha='center', va='center', weight='bold', fontsize=10)
    
    plt.text(env.start_pos[1]+0.5, env.start_pos[0]+0.5, "S", color='black', 
             ha='center', va='center', weight='bold', fontsize=12)
        
    plt.title(f"{title}\n({len(policy_fns)} policies, {successful_episodes}/{total_episodes} successful, SR={success_rate:.1%})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Aggregated heatmap saved to {save_path}")
    
    return stats, visitation_counts


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed training with aggregated heatmap")
    
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed number")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for fairness (1.0 = NSW, 0.0 = Utilitarian)")
    parser.add_argument("--learner", type=str, default="FairDICE", help="Learner type")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.001], help="beta hyperparameter(s) - can pass multiple values")
    parser.add_argument("--divergence", type=str, default="SOFT_CHI", help="Divergence type (SOFT_CHI/CHI/KL)")
    parser.add_argument("--gradient_penalty_coeff", type=float, default=1e-2, help="Gradient penalty coefficient")
    parser.add_argument("--tanh_squash_distribution", type=bool, default=False, help="Use tanh-squash distribution for actions if set")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the network")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the policy")
    parser.add_argument("--layer_norm", type=bool, default=True, help="Use layer normalization if set")
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--quality", type=str, choices=["expert", "amateur"], default="expert", help="Dataset quality")
    parser.add_argument("--preference_dist", type=str, choices=["uniform", "wide", "narrow"], default="uniform", help="Preference distribution")
    parser.add_argument("--max_seq_len", type=int, default=500, help="Max sequence length in trajectories")
    parser.add_argument("--normalize_reward", type=bool, default=False, help="Whether to normalize reward")
    parser.add_argument("--reward_norm", type=str, default="minmax",
                        choices=["minmax", "none"],
                        help="Reward normalization mode (simplified)")
    parser.add_argument("--ignore_fuel", action="store_true",
                        help="Ignore fuel objective entirely - use only ore1 and ore2 (2D rewards)")
    parser.add_argument("--env_name", type=str, default="MO-FourRooms-v0", help="Environment name")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval") 
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes during training")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save the model checkpoint")
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    
    parser.add_argument("--episodes_per_model", type=int, default=100, 
                        help="Episodes to run per model for aggregated heatmap")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only aggregate existing models")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Existing experiment directory to use with --skip_training")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    beta_values = args.beta
    
    print(f"\n{'#'*60}")
    print(f"# Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")
    print(f"# Learner: {args.learner}")
    print(f"# Environment: {args.env_name}")
    print(f"# Beta values: {beta_values}")
    print(f"# Gamma: {args.gamma}")
    print(f"# Total steps per seed: {args.total_train_steps}")
    print(f"{'#'*60}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # shared for all betas
    
    all_experiment_dirs = []
    
    for beta_idx, beta in enumerate(beta_values):
        print(f"# Processing beta={beta}")
        
        config = setup_config(args, beta)
        config, env, batch = load_and_prepare_data(config)
        
        if args.skip_training and args.experiment_dir:
            if len(beta_values) == 1:
                experiment_dir = args.experiment_dir
            else:
                base_dir = os.path.dirname(args.experiment_dir)
                experiment_dir = None
                for d in os.listdir(base_dir):
                    if f"_beta{beta}_" in d:
                        experiment_dir = os.path.join(base_dir, d)
                        break
                if experiment_dir is None:
                    print(f"Warning: Could not find experiment directory for beta={beta}, skipping...")
                    continue
        else:
            experiment_name = f"{timestamp}_{args.learner}_{args.env_name}_beta{beta}_alpha{args.alpha}_{args.num_seeds}seeds"
            experiment_dir = os.path.join(args.save_path, experiment_name)
        
        os.makedirs(experiment_dir, exist_ok=True)
        all_experiment_dirs.append(experiment_dir)
        
        config_dict = vars(args).copy()
        config_dict["beta"] = beta
        config_dict["experiment_dir"] = experiment_dir
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        best_model_paths = []
        seed_results = []
        
        if not args.skip_training:
            for seed in tqdm(seeds, desc=f"Seeds (beta={beta})"):
                seed_dir = os.path.join(experiment_dir, f"seed_{seed}")
                os.makedirs(seed_dir + "/eval", exist_ok=True)
                
                model_path, seed_stats = train_single_seed(
                    config, env, batch, seed, seed_dir
                )
                
                with open(os.path.join(seed_dir, "eval", "summary.json"), "w") as f:
                    json.dump(seed_stats, f, indent=2)
                
                best_model_paths.append(model_path)
                seed_results.append(seed_stats)
                
                tqdm.write(f"  Seed {seed}: Best Welfare(alpha={config.alpha}) = {seed_stats['best_welfare']:.4f}")
            
            print(f"\nbeta={beta} done")
        else:
            for seed in seeds:
                seed_dir = os.path.join(experiment_dir, f"seed_{seed}")
                model_path = os.path.abspath(os.path.join(seed_dir, "model_best"))
                if os.path.exists(model_path):
                    best_model_paths.append(model_path)
                    
                    summary_path = os.path.join(seed_dir, "eval", "summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as f:
                            seed_results.append(json.load(f))
                    else:
                        seed_results.append({"seed": seed})
                            
        aggregation_stats = None
        visitation_counts = None
        if best_model_paths and is_fourrooms_env(config.env_name):
            print(f"Aggregating {len(best_model_paths)} models for beta={beta}...")
            
            policy_fns = []
            for model_path in tqdm(best_model_paths, desc="Loading models"):
                try:
                    policy_fn = create_policy_fn(model_path, config)
                    policy_fns.append(policy_fn)
                except Exception as e:
                    tqdm.write(f"  Error loading {model_path}: {e}")
            
            if policy_fns:
                heatmap_path = os.path.join(experiment_dir, "aggregated_heatmap.png")
                
                aggregation_stats, visitation_counts = aggregate_with_statistics(
                    policy_fns=policy_fns,
                    config=config,
                    save_path=heatmap_path,
                    title=f"Aggregated Policy Heatmap\n{args.learner} | beta={beta}",
                    num_episodes_per_policy=args.episodes_per_model,
                )
                np.save(os.path.join(experiment_dir, "visitation_counts.npy"), visitation_counts)
                
            else:
                print("No models loaded")
        elif best_model_paths:
            print(f"\nSkipping heatmap aggregation (not supported for {config.env_name})")
        
        results = {
            "experiment_config": config_dict,
            "num_seeds_trained": len(seed_results),
            "num_seeds_aggregated": len(best_model_paths),
            "seed_results": seed_results,
            "aggregation_stats": aggregation_stats,
            "summary": {
                "beta": beta,
                "alpha": config.alpha,
                "mean_best_welfare": float(np.mean([r["best_welfare"] for r in seed_results if "best_welfare" in r])) if seed_results else None,
                "std_best_welfare": float(np.std([r["best_welfare"] for r in seed_results if "best_welfare" in r])) if seed_results else None,
            }
        }
        
        with open(os.path.join(experiment_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        if aggregation_stats:
            print(f"Success rate: {aggregation_stats['success_rate']:.2%}")
            print(f"Goal distribution: {aggregation_stats['goal_distribution']}")
        elif seed_results:
            mean_welfare = np.mean([r["best_welfare"] for r in seed_results if "best_welfare" in r])
            std_welfare = np.std([r["best_welfare"] for r in seed_results if "best_welfare" in r])
            print(f"# Mean best welfare: {mean_welfare:.4f} +/- {std_welfare:.4f}")


if __name__ == "__main__":
    main()
