import numpy as np
import pickle
import os

def load_data(dataset_type, method, subsample_size=50):

    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    

    configs = {
        'bc': ("bc_main", "mo_rtg=False/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam"),
        'modt': ("modt_main", "mo_rtg=True/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=1/concat_act_pref=1/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam"),
        'rvs': ("rvs_main", "mo_rtg=True/rtg_scale=100/norm_rew=True/concat_state_pref=1/concat_rtg_pref=0/concat_act_pref=0/percent=1/batch=256/dim=512/layers=3/obj=-1/use_pref=False/return_loss=False/pref_loss=False/optim=adam")
    }

    if method not in configs:
        return np.array([]), np.array([]), np.array([])
    
    method_dir, config_suffix = configs[method]
    full_dataset_name = f"{dataset_type}_uniform"
    
    all_original_returns = [] 
    all_nsw = []              
    all_weights = []

    for seed in range(5):
        file_path = os.path.join(base_path, method_dir, "MO-Hopper-v3", full_dataset_name, "K=20", config_suffix, f"seed={seed}", "logs", "step=100000_rollout.pkl")
        
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

    if all_original_returns:
        try:
            avg_returns = np.mean(np.array(all_original_returns), axis=0)
            avg_nsw = np.mean(np.array(all_nsw), axis=0)
            

            n_points = len(avg_returns)
            if n_points > subsample_size:
                indices = np.linspace(0, n_points - 1, subsample_size, dtype=int)
                return np.array(all_weights)[indices], avg_returns[indices], avg_nsw[indices]
            
            return np.array(all_weights), avg_returns, avg_nsw
        except ValueError:
             print("Warning: Mismatch in data shapes.")
             
    return np.array([]), np.array([]), np.array([])
