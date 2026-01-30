import numpy as np
import gym
from gym import spaces
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
 

class MOFourRoomsEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.layout = """
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(line) for line in self.layout.strip().split('\n')]) == 'w'
        
        self.goals = [
            (10, 2),  
            (2, 10),  
            (10, 10)
        ]
        
        self.start_pos = (2, 2)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=12, shape=(2,), dtype=np.int32)
        self.obj_dim = 3
        self.max_steps = 50
        
        self.deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.noise_prob = 0.1
 
    def reset(self):
        self.agent_pos = self.start_pos
        self.steps = 0
        return np.array(self.agent_pos, dtype=np.float32)
 
    def step(self, action):
        action = int(action)
        self.steps += 1
        
        # randomly (p=0.1 as in paper) go in the wrong direction
        if np.random.random() < self.noise_prob:
            action = np.random.randint(0, 4)
            
        dy, dx = self.deltas[action]
        ny, nx = self.agent_pos[0] + dy, self.agent_pos[1] + dx
        
        # wall collision check
        if not self.occupancy[ny, nx]:
            self.agent_pos = (ny, nx)
            
        # Check goals
        reward_vec = np.zeros(3)
        done = False
        
        for i, goal in enumerate(self.goals):
            if self.agent_pos == goal:
                reward_vec[i] = 1.0
                done = True
                
        if self.steps >= self.max_steps:
            done = True
            # for i, goal in enumerate(self.goals):
            #     dist = (goal[0] - self.agent_pos[0]) ** 2 + (goal[1] - self.agent_pos[1]) ** 2
            #     reward_vec[i] = 1 / (1 + dist)
        
        return np.array(self.agent_pos, dtype=np.float32), np.sum(reward_vec), done, {'obj': reward_vec}
 
    def get_layout(self):
        return self.occupancy
 
 
def obs_to_onehot(obs, h=13, w=13):
    """Convert (y, x) observation to one-hot encoding."""
    y, x = int(obs[0]), int(obs[1])
    one_hot = np.zeros(h * w, dtype=np.float32)
    idx = y * w + x
    if 0 <= idx < len(one_hot):
        one_hot[idx] = 1.0
    return one_hot

def visualize_policy_heatmap(policy_fn, save_path, title, num_episodes=50, success_only=False):
    env = MOFourRoomsEnv()
    layout = env.get_layout()
    visitation_counts = np.zeros(layout.shape)
    trajectories = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        y, x = int(obs[0]), int(obs[1])
        trajectory = [(y, x)]
        
        while not done:
            if policy_fn:
                # Convert raw (y, x) observation to one-hot for the policy
                obs_onehot = obs_to_onehot(obs)
                action = policy_fn(obs_onehot)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, _ = env.step(action)
            y, x = int(obs[0]), int(obs[1])
            trajectory.append((y, x))
        
        if not success_only or reward == 1.0:
            trajectories.append(trajectory)
 
 
    for trajectory in trajectories:
        for y, x in trajectory:
            visitation_counts[y, x] += 1
 
 
    plt.figure(figsize=(6, 6))
    
    mask = layout
    sns.heatmap(visitation_counts, mask=mask, cmap="Blues", cbar=False, square=True,
                linewidths=0.5, linecolor='gray')
 
    plt.imshow(layout, cmap="binary", alpha=0.3)
    
    colors = ['red', 'green', 'blue']
    for i, goal in enumerate(env.goals):
        plt.text(goal[1]+0.5, goal[0]+0.5, f"G{i+1}", color=colors[i], 
                 ha='center', va='center', weight='bold')
        
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def aggregate_policy_heatmaps(policy_fns, save_path, title, num_episodes_per_policy=100, success_only=True):
    """
    Args:
        policy_fns: List of policy functions that take one-hot obs and return action
        save_path: Path to save the combined heatmap
        title: Title for the heatmap
        num_episodes_per_policy: Number of episodes to run per policy
        success_only: If True, only count trajectories that reach a goal
    """
    env = MOFourRoomsEnv()
    layout = env.get_layout()
    visitation_counts = np.zeros(layout.shape)
    total_trajectories = 0
    successful_trajectories = 0
        
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
                obs, reward, done, _ = env.step(action)
            
            y, x = int(obs[0]), int(obs[1])
            trajectory.append((y, x))
            
            total_trajectories += 1
            
            if not success_only or reward == 1.0:
                successful_trajectories += 1
                for y, x in trajectory:
                    visitation_counts[y, x] += 1
            
    print(f"Total trajectories: {total_trajectories}, Successful: {successful_trajectories}")
    
    # Generate heatmap
    plt.figure(figsize=(8, 8))
    
    mask = layout
    sns.heatmap(visitation_counts, mask=mask, cmap="Blues", cbar=True, square=True,
                linewidths=0.5, linecolor='gray')

    plt.imshow(layout, cmap="binary", alpha=0.3)
    
    colors = ['red', 'green', 'blue']
    for i, goal in enumerate(env.goals):
        plt.text(goal[1]+0.5, goal[0]+0.5, f"G{i+1}", color=colors[i], 
                 ha='center', va='center', weight='bold', fontsize=12)
    
    # Add start position marker
    plt.text(env.start_pos[1]+0.5, env.start_pos[0]+0.5, "S", color='black', 
             ha='center', va='center', weight='bold', fontsize=12)
        
    plt.title(f"{title}\n({len(policy_fns)} policies, {successful_trajectories} successful trajectories)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Heatmap saved to {save_path}")
    return visitation_counts

