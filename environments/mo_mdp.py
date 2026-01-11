import numpy as np
import gym
from gym import spaces
import pickle
import os


class MOMDPEnv(gym.Env):
    def __init__(self, seed=42):
        super().__init__()
        
        self.num_states = 50
        self.num_actions = 4
        self.gamma = 0.95
        
        self.start_state = 0
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_states,), dtype=np.float32
        )
        
        self.obj_dim = 3
        self.max_steps = 50
        
        self.rng = np.random.RandomState(seed)
        
        candidates = list(range(1, self.num_states))
        self.goals = sorted(self.rng.choice(candidates, size=3, replace=False))
        
        self.goal_rewards = {
            self.goals[0]: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            self.goals[1]: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            self.goals[2]: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        
        # P(s' | s, a)
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                next_candidates = self.rng.choice(self.num_states, size=4, replace=False)
                probs = self.rng.dirichlet(alpha=[1, 1, 1, 1])
                
                self.transitions[s, a, next_candidates] = probs

        self.current_state = self.start_state
        self.steps = 0

    def _get_obs(self, state_idx):
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[state_idx] = 1.0
        return obs

    def reset(self):
        self.current_state = self.start_state
        self.steps = 0
        return self._get_obs(self.current_state)

    def step(self, action):
        if hasattr(action, 'item'): 
            action = int(action.item())
        else:
            action = int(action)

        self.steps += 1
        
        probs = self.transitions[self.current_state, action]
        next_state = np.random.choice(self.num_states, p=probs)
        
        self.current_state = next_state
        
        reward_vec = np.zeros(3, dtype=np.float32)
        done = False
        
        if self.current_state in self.goal_rewards:
            reward_vec = self.goal_rewards[self.current_state]
            done = True 
        if self.steps >= self.max_steps:
            done = True
                    
        return self._get_obs(self.current_state), np.sum(reward_vec), done, {'obj': reward_vec}

    def get_goals(self):
        return self.goals


def generate_momdp_dataset(num_trajectories=100, seed=42):
    env = MOMDPEnv(seed=seed)
    
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

    os.makedirs("data/MO-RandomMOMDP", exist_ok=True)
    save_path = "data/MO-RandomMOMDP/MO-RandomMOMDP_expert.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")