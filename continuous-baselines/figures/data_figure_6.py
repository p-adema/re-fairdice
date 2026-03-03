import numpy as np
import pickle
import os
import pandas as pd

ENV = "MO-Hopper-v3"
TYPES = ["expert_uniform", "amateur_uniform"]
METHODS = {
    'bc_main': "mo_rtg=False/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam",
    'modt_main': "mo_rtg=True/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=1/concat_act_pref=1/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam",
    'rvs_main': "mo_rtg=True/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam"
}

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def process_and_save(subsample_size=50):
    for dataset_type in TYPES:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV, dataset_type)
        os.makedirs(output_dir, exist_ok=True)
        
        for method_name, config_suffix in METHODS.items():
            print(f"Processing {dataset_type} / {method_name}...")
            
            all_original_returns = []
            all_nsw = []
            all_weights = []
            
            for seed in range(5):
                file_path = os.path.join(BASE_PATH, method_name, ENV, dataset_type, "K=20", config_suffix, f"seed={seed}", "logs", "step=100000_rollout.pkl")
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            points = data.get('rollout_original_raw_r')
                            raw_all = data.get('rollout_unweighted_raw_r_all')
                            weights = data.get('target_prefs')
                            
                            if points is not None and raw_all is not None and weights is not None:
                                all_original_returns.append(points)
                                
                                clipped_raw = np.maximum(raw_all, 1e-5)
                                episode_nsw = np.sum(np.log(clipped_raw), axis=2)
                                all_nsw.append(np.mean(episode_nsw, axis=1))

                                if len(all_weights) == 0:
                                    all_weights = weights
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"File not found: {file_path}")

            if all_original_returns:
                try:
                    avg_returns = np.mean(np.array(all_original_returns), axis=0)
                    avg_nsw = np.mean(np.array(all_nsw), axis=0)
                    weights = np.array(all_weights)
                    
                    n_points = len(avg_returns)
                    if n_points > subsample_size:
                        indices = np.linspace(0, n_points - 1, subsample_size, dtype=int)
                        weights = weights[indices]
                        avg_returns = avg_returns[indices]
                        avg_nsw = avg_nsw[indices]
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'W1': weights[:, 0],
                        'W2': weights[:, 1],
                        'W3': weights[:, 2],
                        'Ret1': avg_returns[:, 0],
                        'Ret2': avg_returns[:, 1],
                        'Ret3': avg_returns[:, 2],
                        'NSW': avg_nsw
                    })
                    
                    csv_path = os.path.join(output_dir, f"{method_name}_figure_6.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"Saved {csv_path}")
                    
                except ValueError as e:
                    print(f"Warning: Mismatch in data shapes for {method_name}: {e}")
            else:
                print(f"No data found for {method_name}")

if __name__ == "__main__":
    process_and_save()
