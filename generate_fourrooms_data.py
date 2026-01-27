import numpy as np
import pickle
import os
import collections
import argparse
from environments.four_rooms import MOFourRoomsEnv

def compute_distance_maps(env):
    layout = env.get_layout()
    H, W = layout.shape
    distance_maps = {} # goal_idx -> np.array(H, W)
    
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i, goal in enumerate(env.goals):
        dist_map = np.full((H, W), fill_value=np.inf)
        goal_int = (int(goal[0]), int(goal[1]))
        
        queue = collections.deque([(goal_int, 0)])
        dist_map[goal_int] = 0
        visited = {goal_int}
        
        while queue:
            (cy, cx), dist = queue.popleft()
            
            for dy, dx in deltas:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and not layout[ny, nx]:
                    if (ny, nx) not in visited:
                        visited.add((ny, nx))
                        dist_map[ny, nx] = dist + 1
                        queue.append(((ny, nx), dist + 1))
        
        distance_maps[i] = dist_map
        
    return distance_maps

def get_shortest_path_action(env, current_pos, target_pos):
    start = tuple(map(int, current_pos))
    target = tuple(map(int, target_pos))
    if start == target: return env.action_space.sample()
    
    queue = collections.deque([(start, [])])
    visited = {start}
    layout = env.get_layout()
    H, W = layout.shape
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (cy, cx), path = queue.popleft()
        if (cy, cx) == target:
            return path[0] if path else env.action_space.sample()
        
        for action, (dy, dx) in enumerate(deltas):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and not layout[ny, nx]:
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [action]))
    return env.action_space.sample()

def get_valid_start_positions(env):
    layout = env.get_layout()
    H, W = layout.shape
    return [(y, x) for y in range(H) for x in range(W) if not layout[y, x]]

def generate_dense_dataset(total_trajectories=400, noise=0.05, seed=42, distribution=None, version="v0"):
    env = MOFourRoomsEnv()
    np.random.seed(seed)
    
    if distribution is None:
        distribution = [1/3, 1/3, 1/3]
    
    if len(distribution) != 3:
        raise ValueError("Distribution must have exactly 3 values (one per goal)")
    if not np.isclose(sum(distribution), 1.0):
        raise ValueError(f"Distribution must sum to 1.0, got {sum(distribution)}")
    
    dist_maps = compute_distance_maps(env)
    valid_starts = get_valid_start_positions(env)
    
    counts = [int(total_trajectories * d) for d in distribution[:2]]
    counts.append(total_trajectories - sum(counts))
    
    dataset = []
    
    print(f"Trajectory counts per goal: {counts}")
    
    for goal_idx, count in enumerate(counts):
        target_goal = env.goals[goal_idx]
        
        for _ in range(count):
            # random starts: super-crucial for convergence
            start_pos = valid_starts[np.random.randint(len(valid_starts))]  
            obs = env.reset()
            env.agent_pos = start_pos
            obs = np.array(env.agent_pos, dtype=np.float32)
            
            done = False
            traj_data = collections.defaultdict(list)
            
            while not done:
                if np.random.random() < noise:
                    action = env.action_space.sample()
                else:
                    action = get_shortest_path_action(env, env.agent_pos, target_goal)
                
                next_obs, sparse_reward, done, info = env.step(action)
                
                curr_y, curr_x = int(obs[0]), int(obs[1])
                dense_reward = np.zeros(3)
                
                for obj_i in range(3):
                    dist = dist_maps[obj_i][curr_y, curr_x]
                    if np.isinf(dist):
                        r = 0.0
                    else:
                        r = 1.0 / (dist + 1.0)
                        
                        if sparse_reward > 0 and obj_i == np.argmax(info['obj']):
                             r += 2.0 
                    
                    dense_reward[obj_i] = r

                timeout = (env.steps >= 20)
                terminal = done and not timeout
                
                traj_data['observations'].append(obs)
                traj_data['actions'].append(action)
                traj_data['next_observations'].append(next_obs)
                
                traj_data['raw_rewards'].append(dense_reward) 
                
                traj_data['terminals'].append(terminal)
                traj_data['timeouts'].append(timeout)
                
                obs = next_obs

            dataset.append({
                'observations': np.array(traj_data['observations']),
                'actions': np.array(traj_data['actions'], dtype=np.float32).reshape(-1, 1),
                'next_observations': np.array(traj_data['next_observations']),
                'raw_rewards': np.array(traj_data['raw_rewards']), # Dense!
                'terminals': np.array(traj_data['terminals']),
                'timeouts': np.array(traj_data['timeouts']),
                'preference': np.ones((len(traj_data['observations']), 3)) / 3.0
            })

    save_dir = f"data/MO-FourRooms-{version}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"MO-FourRooms-{version}_50000_expert_uniform.pkl")
    
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate dense reward dataset for MO-FourRooms environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--total-trajectories',
        type=int,
        default=2000,
        help='Total number of trajectories to generate'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.05,
        help='Probability of taking random action instead of expert action'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--distribution',
        type=float,
        nargs=3,
        default=None,
        metavar=('GOAL0', 'GOAL1', 'GOAL2'),
        help='Distribution of trajectories per goal (3 floats that sum to 1.0). '
             'Example: --distribution 0.8 0.1 0.1 for imbalanced dataset'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='v0',
        help='Version string for the dataset (e.g., v0, v1, v2)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_dense_dataset(
        total_trajectories=args.total_trajectories,
        noise=args.noise,
        seed=args.seed,
        distribution=args.distribution,
        version=args.version
    )