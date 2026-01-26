import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
from scores import scores

def plot_figure_4():
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True)
    
    rows = ["Expert", "Amateur"]
    cols = ["MO-Hopper", "MO-Walker2d", "MO-Ant", "MO-HalfCheetah", "MO-Swimmer"]
    
    env_map = {
        "MO-Hopper": "MO-Hopper-v2",
        "MO-Walker2d": "MO-Walker2d-v2",
        "MO-Ant": "MO-Ant-v2",
        "MO-HalfCheetah": "MO-HalfCheetah-v2",
        "MO-Swimmer": "MO-Swimmer-v2"
    }
    type_map = {
        "Expert": "expert_uniform",
        "Amateur": "amateur_uniform"
    }

    base_path = "/home/scur0076/PEDA/figures"

    handles = []
    labels = []
    
    seen_labels = set()

    for i, row_type in enumerate(rows):
        for j, col_env in enumerate(cols):
            ax = axes[i, j]
            
            current_env = env_map[col_env]
            current_type = type_map[row_type]
            data_dir = os.path.join(base_path, current_env, current_type)
            
            bc_file = os.path.join(data_dir, "bc_main_figure_4.csv")
            rvs_file = os.path.join(data_dir, "rvs_main_figure_4.csv")
            modt_file = os.path.join(data_dir, "modt_main_figure_4.csv")

            if os.path.exists(bc_file):
                try:
                    df = pd.read_csv(bc_file)
                    df['Obj1_Weight'] = df['Preference'].apply(lambda x: float(str(x).strip('[]').split(',')[0]))
                    x = df['Obj1_Weight'].values
                    y = df['Avg_All_Seeds'].values
                    line1, = ax.plot(x, y, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#1f77b4', label='BC(P)')
                    if 'BC(P)' not in seen_labels:
                        handles.append(line1)
                        labels.append('BC(P)')
                        seen_labels.add('BC(P)')
                except Exception as e:
                    print(f"Error plotting BC for {col_env} {row_type}: {e}")
            else:
                print(f"Missing BC file: {bc_file}")

            if os.path.exists(modt_file):
                try:
                    df = pd.read_csv(modt_file)
                    df['Obj1_Weight'] = df['Preference'].apply(lambda x: float(str(x).strip('[]').split(',')[0]))
                    x = df['Obj1_Weight'].values
                    y = df['Avg_All_Seeds'].values
                    line2, = ax.plot(x, y, marker='s', linestyle='-', linewidth=2.5, markersize=8, color='#2ca02c', label='MODT(P)')
                    if 'MODT(P)' not in seen_labels:
                        handles.append(line2)
                        labels.append('MODT(P)')
                        seen_labels.add('MODT(P)')
                except Exception as e:
                    print(f"Error plotting MODT for {col_env} {row_type}: {e}")
            else:
                 print(f"Missing MODT file: {modt_file}")

            if os.path.exists(rvs_file):
                try:
                    df = pd.read_csv(rvs_file)
                    df['Obj1_Weight'] = df['Preference'].apply(lambda x: float(str(x).strip('[]').split(',')[0]))
                    x = df['Obj1_Weight'].values
                    y = df['Avg_All_Seeds'].values
                    line3, = ax.plot(x, y, marker='^', linestyle='-', linewidth=2.5, markersize=8, color='#ff7f0e', label='MORvS(P)')
                    if 'MORvS(P)' not in seen_labels:
                        handles.append(line3)
                        labels.append('MORvS(P)')
                        seen_labels.add('MORvS(P)')
                except Exception as e:
                    print(f"Error plotting RVS for {col_env} {row_type}: {e}")
            else:
                 print(f"Missing RVS file: {rvs_file}")

            ax.set_title(col_env, fontsize=25)
            
            score_env_key = current_env.replace("MO-", "")
            
            score_type_key = "expert" if row_type == "Expert" else "amateur"
            
            score_key = f"{score_env_key}/{score_type_key}/Rerun/1.0"
            baseline_val = scores.get(score_key)

            if "Hopper" in col_env:
                if "Expert" == row_type:
                    ax.set_ylim(2, 12)
                    ax.set_yticks([2, 4, 6, 8, 10, 12])
                elif "Amateur" == row_type:
                    ax.set_ylim(2, 14)
                    ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
            elif "Walker" in col_env:
                if "Expert" == row_type:
                    ax.set_ylim(7, 12)
                    ax.set_yticks([7, 8, 9, 10, 11, 12])
                elif "Amateur" == row_type:
                    ax.set_ylim(7, 13)
                    ax.set_yticks([7, 8, 9, 10, 11, 12, 13])
            elif "Ant" in col_env:
                if "Expert" == row_type:
                    ax.set_ylim(10, 12)
                    ax.set_yticks([10.0, 10.5, 11.0, 11.5, 12.0])
                elif "Amateur" == row_type:
                    ax.set_ylim(10, 12)
                    ax.set_yticks([10.0, 10.5, 11.0, 11.5, 12.0])
            elif "HalfCheetah" in col_env:
                if "Expert" == row_type:
                    ax.set_ylim(10, 13)
                    ax.set_yticks([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
                elif "Amateur" == row_type:
                    ax.set_ylim(10, 12.5)
                    ax.set_yticks([10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
            elif "Swimmer" in col_env:
                if "Expert" == row_type:
                    ax.set_ylim(9, 13)
                    ax.set_yticks([9, 10, 11, 12, 13])
                elif "Amateur" == row_type:
                    ax.set_ylim(11, 12)
                    ax.set_yticks([11.0, 11.2, 11.4, 11.6, 11.8, 12.0])

            if baseline_val is not None:
                line4 = ax.axhline(y=baseline_val, color='#d62728', linestyle='--', linewidth=2.5, label='FairDICE')
                if 'FairDICE' not in seen_labels:
                    handles.append(line4)
                    labels.append('FairDICE')
                    seen_labels.add('FairDICE')
            else:
                print(f"Warning: Baseline value not found for key: {score_key}")

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', labelsize=15)
            ax.set_xticks([1.0, 0.6666, 0.3333, 0.0])
            ax.set_xticklabels(["0", "10", "20", "30"])
            ax.set_xlim(1.05, -0.05)
            
            if j == 0:
                ax.set_ylabel(f"{row_type}", fontsize=25)

    fig.supylabel("Avg Nash Social Welfare", fontsize=25, x=0.01)
    fig.supxlabel("Preference Weight ([1.0, 0.0] -> [0.0, 1.0])", fontsize=25, y=0.10)

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=25, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0.02, 0.1, 1, 1], h_pad=3, w_pad=3)
    
    output_path = os.path.join(base_path, 'replication_figure_4.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_figure_4()
