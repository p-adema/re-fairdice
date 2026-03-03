import pickle
import numpy as np
import pandas as pd
import os

import glob

ENV = "MO-Walker2d-v2"
TYPE = "amateur_uniform"

BC_FILE   = "bc_main"
MODT_FILE = "modt_main"
RVS_FILE  = "rvs_main"

BASE_GLOB_PATTERN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "{FILE_NAME}/{ENV}/{TYPE}/**/seed={seed}/logs/step=100000_rollout.pkl")

def load_and_aggregate(file_name):
    aggregated_data = {}
    
    seeds = [0, 1, 2, 3, 4]
    print(f"Aggregating Raw Returns for seeds: {seeds}...")

    initialized = False
    n_prefs = 0

    for seed in seeds:
        # Format the pattern for the current seed
        search_pattern = BASE_GLOB_PATTERN.format(FILE_NAME=file_name, ENV=ENV, TYPE=TYPE, seed=seed)
        
        # Find matches
        matches = glob.glob(search_pattern, recursive=True)
        
        if not matches:
            print(f"Warning: File not found for seed {seed} matching pattern: {search_pattern}")
            continue
            
        return_file = matches[0]
        print(f"Found file for seed {seed}: {return_file}")
        
        rollout_file = return_file
        
        if not os.path.exists(rollout_file):
            print(f"Warning: File path issue for seed {seed}: {rollout_file}")
            continue
            
        try:
            with open(rollout_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading seed {seed}: {e}")
            continue

        returns = data.get('rollout_original_raw_r_all') 
        prefs = data.get('target_prefs')
        
        if returns is None:
            print(f"Error: Data missing in seed {seed}")
            continue
            
        current_n_prefs = returns.shape[0]
        
        if not initialized:
            n_prefs = current_n_prefs
            for i in range(n_prefs):
                aggregated_data[i] = {'obj1': [], 'obj2': [], 'pref_vec': prefs[i]}
            initialized = True
        elif current_n_prefs != n_prefs:
            print(f"Warning: Mismatch in number of preferences for seed {seed}. Expected {n_prefs}, got {current_n_prefs}")
            continue
            
        # Process data for this seed
        for i in range(n_prefs):
            seed_mean_returns = np.mean(returns[i], axis=0)
            
            aggregated_data[i]['obj1'].append(seed_mean_returns[0])
            aggregated_data[i]['obj2'].append(seed_mean_returns[1])

    if not initialized:
        print("No valid data found.")
        return

    # Create DataFrame
    rows = []
    for i in range(n_prefs):
        pref_vec = aggregated_data[i]['pref_vec']
        obj1_vals = aggregated_data[i]['obj1']
        obj2_vals = aggregated_data[i]['obj2']
        
        if not obj1_vals: # No data for this pref
            continue
            
        row_data = {
            "Pref_Weight_0": pref_vec[0],
            "Pref_Weight_1": pref_vec[1],
            "Obj1_Mean": np.mean(obj1_vals),
            "Obj1_Std": np.std(obj1_vals),
            "Obj1_SE": np.std(obj1_vals) / np.sqrt(len(obj1_vals)),
            "Obj2_Mean": np.mean(obj2_vals),
            "Obj2_Std": np.std(obj2_vals),
            "Obj2_SE": np.std(obj2_vals) / np.sqrt(len(obj2_vals)),
            "Count": len(obj1_vals)
        }
        rows.append(row_data)

    df = pd.DataFrame(rows)
    
    df = df.sort_values(by="Pref_Weight_0")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(f"\n--- Evaluation Results (Raw Returns) for {file_name} ---")
    print(df.to_string(index=False))
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV, TYPE)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{file_name}_figure_5.csv")
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
