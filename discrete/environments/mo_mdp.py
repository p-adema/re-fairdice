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


def compute_optimal_policy(env):
    """
    Compute optimal policy using value iteration.
    Uses sum of objective rewards as scalarized reward (reaching any goal gives reward 1.0).
    
    Returns:
        optimal_policy: Array of shape (num_states,) with optimal action for each state
        V: Array of shape (num_states,) with optimal value for each state
    """
    num_states = env.num_states
    num_actions = env.num_actions
    gamma = env.gamma
    
    # Scalarized reward: 1.0 for reaching any goal state
    rewards = np.zeros(num_states)
    for goal in env.goals:
        rewards[goal] = 1.0
    
    # Terminal state mask (goal states are terminal)
    terminal_mask = np.zeros(num_states)
    for goal in env.goals:
        terminal_mask[goal] = 1.0
    
    # Value iteration
    V = np.zeros(num_states)
    max_iters = 1000
    tol = 1e-8
    
    for iteration in range(max_iters):
        # Q(s,a) = E[R(s') + gamma * V(s') * (1 - terminal(s'))]
        # where expectation is over s' ~ P(s'|s,a)
        
        # Expected immediate reward from next state
        expected_reward = np.einsum('sak,k->sa', env.transitions, rewards)
        
        # Expected future value (0 for terminal next states)
        V_non_terminal = V * (1 - terminal_mask)
        expected_future = np.einsum('sak,k->sa', env.transitions, V_non_terminal)
        
        Q = expected_reward + gamma * expected_future
        V_new = np.max(Q, axis=1)
        
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    
    # Extract optimal policy (greedy w.r.t. Q)
    optimal_policy = np.argmax(Q, axis=1)
    
    return optimal_policy, V


def generate_momdp_dataset(num_trajectories=100, seed=42, optimality=0.5):
    """
    Generate offline dataset for MO-MDP.
    
    Args:
        num_trajectories: Number of trajectories to collect (default: 100)
        seed: Random seed for environment generation
        optimality: Behavior policy optimality level (0.0 = random, 1.0 = optimal)
                   The behavior policy takes the optimal action with probability `optimality`
                   and a uniformly random action with probability `1 - optimality`.
    """
    env = MOMDPEnv(seed=seed)
    rng = np.random.RandomState(seed + 1000)  # Separate RNG for behavior policy
    
    # Compute optimal policy via value iteration
    optimal_policy, V = compute_optimal_policy(env)
    
    dataset = []
    print(f"Generating {num_trajectories} trajectories with optimality={optimality}...")

    
    for _ in range(num_trajectories):
        obs = env.reset()
        done = False
        
        traj_obs, traj_next_obs, traj_actions, traj_rewards = [], [], [], []
        traj_terminals, traj_timeouts = [], []
        
        while not done:
            # Behavior policy: with prob `optimality`, take optimal action; otherwise random
            if rng.random() < optimality:
                action = optimal_policy[env.current_state]
            else:
                action = rng.randint(env.num_actions)
            
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

    os.makedirs("data/MO-RandomMOMDP-v0", exist_ok=True)
    save_path = "data/MO-RandomMOMDP-v0/MO-RandomMOMDP-v0_50000_expert_uniform.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    generate_momdp_dataset(num_trajectories=100, seed=42, optimality=0.5)