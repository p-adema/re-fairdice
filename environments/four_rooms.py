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
        # hardcode for now, deal with 8 objectives later
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
        
        # Wall collision check
        if not self.occupancy[ny, nx]:
            self.agent_pos = (ny, nx)
            
        # Check goals
        reward_vec = np.zeros(3)
        done = False
        
        for i, goal in enumerate(self.goals):
            if self.agent_pos == goal:
                # assert 0, (self.agent_pos, self.steps, goal)
                reward_vec[i] = 1.0
                done = True
                break
                
        if self.steps >= self.max_steps:
            done = True
        
        return np.array(self.agent_pos, dtype=np.float32), np.sum(reward_vec), done, {'obj': reward_vec}
 
    def get_layout(self):
        return self.occupancy
 
 
def generate_dataset(num_trajectories=300):
    env = MOFourRoomsEnv()
    dataset = []
    
    print(f"Generating {num_trajectories} trajectories...")
    for _ in range(num_trajectories):
        obs = env.reset()
        done = False
        traj_obs, traj_next_obs, traj_actions, traj_rewards = [], [], [], []
        traj_terminals, traj_timeouts = [], []
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            timeout = (env.steps >= env.max_steps)
            terminal = done and not timeout
            
            traj_obs.append(obs)
            traj_actions.append(action)
            traj_rewards.append(info['obj'])
            traj_next_obs.append(next_obs)
            traj_terminals.append(terminal)
            traj_timeouts.append(timeout)
            
            obs = next_obs
 
        dataset.append({
            'observations': np.array(traj_obs),
            'actions': np.array(traj_actions, dtype=np.float32).reshape(-1, 1),
            'next_observations': np.array(traj_next_obs),
            'raw_rewards': np.array(traj_rewards),
            'terminals': np.array(traj_terminals),
            'timeouts': np.array(traj_timeouts),
            'preference': np.ones((len(traj_obs), 3)) / 3.0
        })
        
    os.makedirs("data/MO-FourRooms", exist_ok=True)
    save_path = "data/MO-FourRooms/MO-FourRooms_expert_uniform.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")
    return dataset
 
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
                action = policy_fn(obs)
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
 
 
if __name__ == "__main__":
    def mock_policy(obs):
        return int(np.random.choice([0,1,2,3], 1, p=[0, 0.9, 0, 0.1]))
 
    visualize_policy_heatmap(mock_policy, "Mock Policy", "test_heatmap.png", 1000)
    visualize_policy_heatmap(mock_policy, "Mock Policy Success Only", "test_heatmap_success_only.png", 1000, success_only=True)