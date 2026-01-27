import os
import gym
import gymnasium
import environments
from environments.norm import state_norm_params
from environments.four_rooms import visualize_policy_heatmap

import pickle
import numpy as np
from collections import defaultdict
from utils import normalization, min_max_normalization, social_welfare
from buffer import Buffer
import argparse
from types import SimpleNamespace
from evaluation import evaluate_policy
import random
from tqdm import tqdm
import wandb
import pandas as pd
import jax
from datetime import datetime
import gym.spaces
import gymnasium.spaces


# =============================================================================
# Environment Detection and Configuration
# =============================================================================

def is_minecart_env(env_name):
    """Check if the environment is a minecart variant."""
    minecart_names = ['minecart', 'MO-Minecart']
    return any(name.lower() in env_name.lower() for name in minecart_names)


def is_rgb_env(env_name):
    """Check if the environment uses RGB image observations."""
    return 'rgb' in env_name.lower()


def is_fourrooms_env(env_name):
    """Check if the environment is FourRooms."""
    return 'fourrooms' in env_name.lower() or 'four-rooms' in env_name.lower()


# =============================================================================
# Environment Wrappers
# =============================================================================

class OneHotWrapper(gym.ObservationWrapper):
    """Wrapper for FourRooms that converts (y, x) coordinates to one-hot vector."""
    def __init__(self, env):
        super().__init__(env)
        # Assuming standard FourRooms 13x13
        self.h, self.w = 13, 13 
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.h * self.w,), dtype=np.float32
        )

    def observation(self, obs):
        # Convert (y, x) -> One-Hot Vector
        y, x = int(obs[0]), int(obs[1])
        one_hot = np.zeros(self.h * self.w, dtype=np.float32)
        idx = y * self.w + x
        if 0 <= idx < len(one_hot):
            one_hot[idx] = 1.0
        return one_hot


class MinecartNormWrapper(gym.ObservationWrapper):
    """Wrapper for minecart that normalizes observations."""
    def __init__(self, env, is_rgb=False):
        super().__init__(env)
        self.is_rgb = is_rgb
        
    def observation(self, obs):
        if self.is_rgb:
            # Normalize RGB images to [0, 1]
            return obs.astype(np.float32) / 255.0
        else:
            # Vector observations are already roughly normalized
            return obs.astype(np.float32)


# =============================================================================
# Dataset Preprocessing Functions
# =============================================================================

def preprocess_dataset_to_onehot(trajs, h=13, w=13):
    """Convert FourRooms coordinates to one-hot representation."""
    print("Converting dataset coordinates to One-Hot representation...")
    new_dim = h * w
    for traj in trajs:
        # Transform 'observations'
        N = traj['observations'].shape[0]
        new_obs = np.zeros((N, new_dim), dtype=np.float32)
        for i, obs in enumerate(traj['observations']):
            y, x = int(obs[0]), int(obs[1])
            idx = y * w + x
            if 0 <= idx < new_dim:
                new_obs[i, idx] = 1.0
        traj['observations'] = new_obs
        
        # Transform 'next_observations'
        new_next_obs = np.zeros((N, new_dim), dtype=np.float32)
        for i, obs in enumerate(traj['next_observations']):
            y, x = int(obs[0]), int(obs[1])
            idx = y * w + x
            if 0 <= idx < new_dim:
                new_next_obs[i, idx] = 1.0
        traj['next_observations'] = new_next_obs
        
        # Update init_observations/states clones
        traj["init_observations"] = np.tile(traj['observations'][0], (N, 1))
        
    return trajs, new_dim


def preprocess_minecart_dataset(trajs, is_rgb=False):
    """
    Preprocess minecart dataset.
    
    For vector observations (7D): keeps as-is, observations are already normalized.
    For RGB observations: normalizes pixel values to [0, 1].
    """
    if is_rgb:
        print("Preprocessing RGB observations (normalizing to [0, 1])...")
        for traj in trajs:
            traj['observations'] = traj['observations'].astype(np.float32) / 255.0
            traj['next_observations'] = traj['next_observations'].astype(np.float32) / 255.0
            N = traj['observations'].shape[0]
            traj["init_observations"] = np.tile(traj['observations'][0], (N, 1, 1, 1))
    else:
        print("Processing vector observations for minecart...")
        for traj in trajs:
            traj['observations'] = traj['observations'].astype(np.float32)
            traj['next_observations'] = traj['next_observations'].astype(np.float32)
            N = traj['observations'].shape[0]
            traj["init_observations"] = np.tile(traj['observations'][0], (N, 1))
    
    # Return observation dimension
    obs_shape = trajs[0]['observations'].shape[1:]
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = obs_shape  # For images, keep as tuple
    
    return trajs, obs_dim


def setup_environment(config):
    """
    Set up environment and configure observation/action spaces.
    
    Returns:
        env: Wrapped environment
        config: Updated config with environment parameters
    """
    env_name = config.env_name
    
    if is_minecart_env(env_name):
        # Use the wrapper from environments.minecart_wrapper
        from environments.minecart_wrapper import make_minecart, make_minecart_deterministic, make_minecart_rgb
        
        if is_rgb_env(env_name):
            env = make_minecart_rgb()
            config.use_cnn = True
            config.cnn_feature_dim = 256
        elif 'deterministic' in env_name.lower():
            env = make_minecart_deterministic()
            config.use_cnn = False
        else:
            env = make_minecart(deterministic=False)
            config.use_cnn = False
    else:
        # Standard gym.make for other environments
        env = gym.make(env_name)
        config.use_cnn = False
        
        # Apply OneHotWrapper for FourRooms
        if is_fourrooms_env(env_name):
            env = OneHotWrapper(env)
    
    return env, config

def main():
    parser = argparse.ArgumentParser()
    # In main.py arguments
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for fairness (1.0 = NSW, 0.0 = Utilitarian)")
    parser.add_argument("--learner", type=str, default="limodice", help="Learner type")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=0.001, help="beta hyperparameter")
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
    parser.add_argument("--normalize_reward", type=bool, default=False, help="Whether to normalize reward") # Default changed to False for discrete usually
    parser.add_argument("--env_name", type=str, default="MO-Hopper-v2", help="Environment name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Running mode: 'train' or 'eval'")
    parser.add_argument("--load_path", type=str, default=None, help="Path to a saved model checkpoint (for eval mode).")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval") 
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save the model checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")    
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    
    args, unknown = parser.parse_known_args()
    config = SimpleNamespace(**vars(args))
    
    # Construct data path (ensure directory exists or file exists)
    data_path = f"./data/{config.env_name}/{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl"
    # Fallback for the simpler naming in our discrete script if needed
    if not os.path.exists(data_path):
        alt_path = f"./data/{config.env_name}/{config.env_name}_{config.quality}.pkl" 
        if os.path.exists(alt_path):
            data_path = alt_path
    
    # Try minecart-style naming (with total transitions instead of 50000)
    if not os.path.exists(data_path):
        import glob
        pattern = f"./data/{config.env_name}/{config.env_name}_*_{config.quality}_{config.preference_dist}.pkl"
        matches = glob.glob(pattern)
        if matches:
            data_path = matches[0]

    print(f"Loading trajectories from {data_path}")
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
    
    # Setup environment first (to determine env type)
    env, config = setup_environment(config)
    
    # Preprocess dataset based on environment type
    if is_minecart_env(config.env_name):
        is_rgb = is_rgb_env(config.env_name)
        trajs, obs_dim = preprocess_minecart_dataset(trajs, is_rgb=is_rgb)
        
        if is_rgb:
            # For RGB, state_dim is the image shape
            config.state_dim = obs_dim  # Will be (480, 480, 3)
        else:
            # For vector observations
            config.state_dim = obs_dim
    elif is_fourrooms_env(config.env_name):
        trajs, one_hot_dim = preprocess_dataset_to_onehot(trajs)
        config.state_dim = one_hot_dim
    else:
        # Default: use observation space from env
        config.state_dim = env.observation_space.shape[0]
    
    # Configure action space (check both gym and gymnasium Discrete spaces)
    if isinstance(env.action_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
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
        config.ACTION_LOW  = env.action_space.low
        config.ACTION_SCALE = (config.ACTION_HIGH - config.ACTION_LOW) / 2.0  
        config.ACTION_BIAS  = (config.ACTION_HIGH + config.ACTION_LOW) / 2.0 

    config.reward_dim = env.obj_dim

    # Set up state normalization
    if config.env_name in state_norm_params:
        config.state_mean = state_norm_params[config.env_name]["mean"]
        config.state_std = np.sqrt(state_norm_params[config.env_name]["var"])
    elif is_minecart_env(config.env_name):
        # Minecart: observations are roughly in [-1, 1], use identity normalization
        if is_rgb_env(config.env_name):
            # RGB: already normalized to [0, 1] in preprocessing
            config.state_mean = 0.0
            config.state_std = 1.0
        else:
            config.state_mean = np.zeros(7)
            config.state_std = np.ones(7)
    else:
        print(f"Warning: {config.env_name} not found in norm.py. Using identity normalization.")
        if is_fourrooms_env(config.env_name):
            config.state_mean = np.zeros(13 * 13)
            config.state_std = np.ones(13 * 13)
        else:
            all_obs = np.concatenate([t['observations'] for t in trajs], axis=0)
            config.state_mean = np.mean(all_obs, axis=0)
            config.state_std = np.std(all_obs, axis=0) + 1e-8
    
    # Calculate Reward Statistics
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

    # Normalize Data
    use_cnn = getattr(config, 'use_cnn', False)
    
    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], reward_min, reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        
        # Handle state normalization based on observation type
        if use_cnn:
            # For CNN (RGB images): observations are already normalized to [0, 1]
            # states are the same as observations for CNN
            traj["states"] = traj['observations']
            traj['next_states'] = traj['next_observations']
        else:
            # For vector observations: apply standard normalization
            traj["states"] = normalization(traj['observations'], config.state_mean, config.state_std)
            traj['next_states'] = normalization(traj['next_observations'], config.state_mean, config.state_std)
        
        # Apply Action Normalization only for Continuous
        if not config.is_discrete:
            traj['actions'] = (traj['actions'] - config.ACTION_BIAS) / config.ACTION_SCALE
        else:
            # For discrete, ensure it is float32 for the buffer/network
            traj['actions'] = traj['actions'].astype(np.float32)

        # Handle init_observations/states based on observation shape
        N = traj['observations'].shape[0]
        if use_cnn:
            # For images: tile with appropriate shape
            traj["init_observations"] = np.tile(traj['observations'][0:1], (N, 1, 1, 1))
            traj["init_states"] = traj["init_observations"]
        else:
            traj["init_observations"] = np.tile(traj['observations'][0], (N, 1))
            traj["init_states"] = np.tile(traj['states'][0], (N, 1))

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
    
    if config.learner == "FairDICE":
        from FairDICE import init_train_state, train_step, get_model, save_model, load_model
    else:
        raise ValueError("Invalid learner type.")
    
    
    save_dir = f"{config.save_path}/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + "/eval")


    random.seed(config.seed); np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    train_state = init_train_state(config)
    train_carry = (train_state, key)
    buffer = Buffer(batch)
            
    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step(config, train_state, data, subkey)
        return (train_state, key), (update_info)
    
    if config.wandb:
        wandb.init(
            project=f"exp_{config.tag}",
            name=run_name,
            config=config
            )   
    train_iterations = config.total_train_steps // config.log_interval

    best_welfare = -float('inf')

    for iter in tqdm(range(train_iterations), desc="Training ..."):
        step = (iter + 1) * config.log_interval  
        train_carry, update_info = jax.lax.scan(train_body, train_carry, length=config.log_interval)
        policy = get_model(train_carry[0].policy_state)[0]
        avg_returns, _, avg_steps = evaluate_policy(config, 
                    policy, 
                    env,
                    save_dir + "/eval",
                    num_episodes=config.eval_episodes, 
                    max_steps=config.max_seq_len,
                    t_env=step)

        pop_welfare = social_welfare(avg_returns, config.alpha)
        
        print(f"Step {step}: Returns={avg_returns}, Welfare(alpha={config.alpha})={pop_welfare:.4f} Steps={avg_steps}")
        if pop_welfare > best_welfare:
            print(f" -> New Best Model! (Previous: {best_welfare:.4f})")
            best_welfare = pop_welfare
            # Save to a specific 'best' folder
            if config.save_path:
                best_save_path = os.path.abspath(save_dir + "/model_best")
                save_model(train_carry[0], best_save_path)
            
        if config.wandb:
            for key, value in update_info.items():
                if "loss" in key or "grad" in key or "debug"   in key:
                    wandb.log({f"{key}": value[-1]}, step=step)
                else:
                    for i in range(config.reward_dim):
                        wandb.log({f"{key}_{i}": value[-1][i]}, step=step)

    if config.wandb:
        wandb.finish()
    if config.save_path:
        save_model(train_carry[0], os.path.abspath(save_dir + "/model"))

    # Load best model and visualize heatmap
    best_model_path = os.path.abspath(save_dir + "/model_best")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for visualization...")
        best_train_state = load_model(best_model_path, config)
        best_policy = get_model(best_train_state.policy_state)[0]
        
        # Create policy function that works with the wrapped environment
        def policy_fn(obs):
            # obs is already one-hot from the OneHotWrapper
            state = normalization(obs, config.state_mean, config.state_std)
            state = state.reshape(1, -1)  # Add batch dimension
            dist = best_policy(state)
            action = dist.sample(seed=jax.random.PRNGKey(np.random.randint(0, 10000)))
            return int(action[0])
        
        # Visualize and save heatmap
        heatmap_path = os.path.join(save_dir, "policy_heatmap.png")
        visualize_policy_heatmap(
            policy_fn=policy_fn,
            save_path=heatmap_path,
            title=f"Best Policy Heatmap - {run_name}",
            num_episodes=1000,
            success_only=True
        )
        print(f"Policy heatmap saved to {heatmap_path}")
    else:
        print("No best model found, skipping heatmap visualization.")

if __name__ == "__main__":
    main()