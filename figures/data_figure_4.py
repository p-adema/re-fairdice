import pickle

import numpy as np
import pandas as pd
import os

import glob

ENV  = "MO-Walker2d-v2"
TYPE = "amateur_uniform"

BC_FILE   = "bc_main"
MODT_FILE = "modt_main"
RVS_FILE  = "rvs_main"


BASE_GLOB_PATTERN = "/home/scur0076/PEDA/results/{FILE_NAME}/{ENV}/{TYPE}/**/seed={seed}/logs/step=100000_rollout.pkl"

def load_and_aggregate(file_name):
    all_seeds_nsw = {}
    preferences = None
    seeds = [0, 1, 2, 3, 4]
    
    print(f"Aggregating NSW for seeds: {seeds}...")

    total_negative_rewards = 0
    total_episodes = 0
    
    for seed in seeds:
        # Format the pattern for the current seed
        search_pattern = BASE_GLOB_PATTERN.format(FILE_NAME=file_name, ENV=ENV, TYPE=TYPE, seed=seed)
        
        # Find matches
        matches = glob.glob(search_pattern, recursive=True)
        
        if not matches:
            print(f"Warning: File not found for seed {seed} matching: {search_pattern}")
            all_seeds_nsw[seed] = None
            continue
            
        # Use the first match found
        rollout_file = matches[0]
        print(f"Found file for seed {seed}: {rollout_file}")
        
        if not os.path.exists(rollout_file):

            all_seeds_nsw[seed] = None
            continue
            
        try:
            with open(rollout_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading seed {seed}: {e}")
            all_seeds_nsw[seed] = None
            continue

        returns = data.get('rollout_unweighted_raw_r_all')
        prefs = data.get('target_prefs')
        
        if returns is None:
            print(f"Error: Data missing in seed {seed}")
            all_seeds_nsw[seed] = None
            continue
            
        if preferences is None:
            preferences = prefs
            
        n_prefs = returns.shape[0]
        n_episodes = returns.shape[1]
        
        seed_avg_nsw = []
        for i in range(n_prefs):
            nsw_values = []
            for ep in range(n_episodes):
                r_vec = returns[i, ep, :]
                if np.any(r_vec <= 0):
                    total_negative_rewards += 1
                total_episodes += 1
                
                r_vec = np.maximum(r_vec, 1e-5)
                nsw_values.append(np.sum(np.log(r_vec)))
            seed_avg_nsw.append(np.nanmean(nsw_values))
        all_seeds_nsw[seed] = seed_avg_nsw
        
    if total_episodes > 0:
        print(f"STATS for {file_name}: Negative/Zero Reward Episodes: {total_negative_rewards}/{total_episodes} ({(total_negative_rewards/total_episodes)*100:.2f}%)")

    if preferences is None:
        print("No valid data found.")
        return

    rows = []
    n_prefs = len(preferences)
    
    for i in range(n_prefs):
        pref_str = f"[{preferences[i][0]:.2f}, {preferences[i][1]:.2f}]"
        row_data = {"Preference": pref_str}
        row_values = []
        
        for seed in seeds:
            val = all_seeds_nsw[seed][i] if all_seeds_nsw[seed] is not None else np.nan
            row_data[f"Seed_{seed}"] = f"{val:.4f}" if not np.isnan(val) else "NaN"
            row_values.append(val)
            
        avg_over_seeds = np.nanmean(row_values)
        row_data["Avg_All_Seeds"] = f"{avg_over_seeds:.4f}"
        rows.append(row_data)

    rows.reverse()
    df = pd.DataFrame(rows)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(f"\n--- Evaluation Results (Avg NSW) for {file_name} ---")
    print(df.to_string(index=False))
    
    # Calculate and print variance stats
    avg_stds = []
    for i in range(n_prefs):
        row_values = []
        for seed in seeds:
            val = all_seeds_nsw[seed][i] if all_seeds_nsw[seed] is not None else np.nan
            row_values.append(val)
        avg_stds.append(np.nanstd(row_values))
    
    print(f"STATS for {file_name}: Average Std Dev across preferences: {np.mean(avg_stds):.4f}")
    
    output_dir = os.path.join("/home/scur0076/PEDA/figures", ENV, TYPE)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{file_name}_figure_4.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table to {csv_path}")

if __name__ == "__main__":
    ENVS = ["MO-Hopper-v2", "MO-Walker2d-v2", "MO-Ant-v2", "MO-HalfCheetah-v2", "MO-Swimmer-v2"]
    TYPES = ["expert_uniform", "amateur_uniform"]
    
    for env in ENVS:
        for dtype in TYPES:
            # Update global variables for the current iteration
            ENV = env
            TYPE = dtype
            print(f"\nProcessing {ENV} / {TYPE}...")
            
            for fname in [BC_FILE, MODT_FILE, RVS_FILE]:
                load_and_aggregate(fname)
