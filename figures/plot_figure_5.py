import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
import json
def get_pareto_frontier(x, y):
    # Sort by x descending
    sorted_indices = np.argsort(x)[::-1]
    sorted_x = np.array(x)[sorted_indices]
    sorted_y = np.array(y)[sorted_indices]
    
    frontier_x = []
    frontier_y = []
    max_y = -np.inf
    
    for px, py in zip(sorted_x, sorted_y):
        if py >= max_y:
            frontier_x.append(px)
            frontier_y.append(py)
            max_y = py
            
    # Sort by x ascending for plotting lines
    frontier_x = np.array(frontier_x)
    frontier_y = np.array(frontier_y)
    
    sort_idx = np.argsort(frontier_x)
    return frontier_x[sort_idx], frontier_y[sort_idx]

def plot_figure_5():

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
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
    
    with open(os.path.join(base_path, 'scores.json'), 'r') as f:
        scores_data = json.load(f)
    

    handles = []
    labels = []
    seen_labels = set()

    for i, row_type in enumerate(rows):
        for j, col_env in enumerate(cols):
            ax = axes[i, j]
            

            xlabel = ""
            ylabel = ""
            if "Hopper" in col_env:
                x_min_limit = -500
                y_min_limit = -500
                ax.set_xlim(-500, 3800)
                ax.set_ylim(-500, 5200)
                ax.set_xticks([0, 1000, 2000, 3000])
                ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
                ylabel = "Height"
                xlabel = "Speed vs. Height"
            elif "Walker" in col_env:
                x_min_limit = -100
                y_min_limit = 0
                ax.set_xlim(-100, 2500)
                ax.set_ylim(0, 2700)
                ax.set_xticks([0, 1000, 2000])
                ax.set_yticks([500, 1000, 1500, 2000, 2500])
                ylabel = "Energy"
                xlabel = "Speed vs. Energy"
            elif "Ant" in col_env:

                x_min_limit = 0
                y_min_limit = 0
                ax.set_xlim(-100, 2800)
                ax.set_ylim(-200, 2800)
                ax.set_xticks([0, 1000, 2000])
                ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
                ylabel = "Vertical Speed" 
                xlabel = "Horizontal vs. Vertical Speed"
            elif "HalfCheetah" in col_env:

                x_min_limit = 300
                y_min_limit = 400
                ax.set_xlim(300, 2600)
                ax.set_ylim(400, 2700)
                ax.set_xticks([500, 1000, 1500, 2000, 2500])
                ax.set_yticks([500, 1000, 1500, 2000, 2500])
                ylabel = "Energy"
                xlabel = "Speed vs. Energy"
            elif "Swimmer" in col_env:

                x_min_limit = -10
                y_min_limit = 0
                ax.set_xlim(-10, 280)
                ax.set_ylim(4, 165)
                ax.set_xticks([0, 100, 200])
                ax.set_yticks([25, 50, 75, 100, 125, 150])
                ylabel = "Energy"
                xlabel = "Speed vs. Energy"
            else:
                x_min_limit = 0
                y_min_limit = 0


            score_env_key = col_env.replace("MO-", "") + "-v2"
            score_type_key = "expert" if row_type == "Expert" else "amateur"

            try:
                raw_rerun = scores_data[score_type_key][score_env_key]["Raw_Rerun"]
            except KeyError:
                raw_rerun = None

            try:
                raw_fixed = scores_data[score_type_key][score_env_key]["Raw_Fixed"]
            except KeyError:
                raw_fixed = None

            current_env = env_map[col_env]
            current_type = type_map[row_type]
            data_dir = os.path.join(base_path, current_env, current_type)
            
            bc_file = os.path.join(data_dir, "bc_main_figure_5.csv")
            rvs_file = os.path.join(data_dir, "rvs_main_figure_5.csv")
            modt_file = os.path.join(data_dir, "modt_main_figure_5.csv")


            datasets = []
            

            if os.path.exists(bc_file):
                try:
                    df = pd.read_csv(bc_file)
                    datasets.append((df['Obj1_Mean'].values, df['Obj2_Mean'].values, '#1f77b4', 'o', 'BC(P)'))
                except Exception as e:
                    print(f"Error loading BC {col_env} {row_type}: {e}")
            

            if os.path.exists(modt_file):
                try:
                    df = pd.read_csv(modt_file)
                    datasets.append((df['Obj1_Mean'].values, df['Obj2_Mean'].values, '#2ca02c', 's', 'MODT(P)'))
                except Exception as e:
                    print(f"Error loading MODT {col_env} {row_type}: {e}")


            if os.path.exists(rvs_file):
                try:
                    df = pd.read_csv(rvs_file)
                    datasets.append((df['Obj1_Mean'].values, df['Obj2_Mean'].values, '#ff7f0e', '^', 'MORvS(P)'))
                except Exception as e:
                    print(f"Error loading RVS {col_env} {row_type}: {e}")


            for data_x, data_y, color, marker, label in datasets:
                if len(data_x) > 0:
                    ax.scatter(data_x, data_y, color=color, s=60, marker=marker, alpha=0.3, linewidths=0, zorder=2)
                    
                    pf_x, pf_y = get_pareto_frontier(data_x, data_y)
                    
                    line, = ax.plot(pf_x, pf_y, color=color, linewidth=2.5, zorder=3, label=label)
                    
                    ax.scatter(pf_x, pf_y, color=color, s=80, marker=marker, zorder=4, alpha=1.0)
                    

                    if label not in seen_labels:
                        handles.append(line)
                        labels.append(label)
                        seen_labels.add(label)


                    if len(pf_x) > 0:
                        fill_x = np.concatenate(([x_min_limit], pf_x))
                        fill_y = np.concatenate(([pf_y[0]], pf_y))
                        ax.fill_between(fill_x, fill_y, y2=y_min_limit, color=color, alpha=0.1, zorder=1)


            if raw_rerun:
                ax.scatter(raw_rerun[0], raw_rerun[1], color='#d62728', s=250, marker='o', zorder=10, label='FairDICE (Rerun)', edgecolors='white', linewidths=2.0)
                if 'FairDICE (Rerun)' not in seen_labels:
                    import matplotlib.lines as mlines
                    fd_line = mlines.Line2D([], [], color='#d62728', marker='o', linestyle='None',
                                          markersize=15, label='FairDICE (Rerun)', markeredgecolor='white', markeredgewidth=2.0)
                    handles.append(fd_line)
                    labels.append('FairDICE (Rerun)')
                    seen_labels.add('FairDICE (Rerun)')

            if raw_fixed:
                ax.scatter(raw_fixed[0], raw_fixed[1], color='#9467bd', s=250, marker='o', zorder=10, label='FairDICE (Fixed)', edgecolors='white', linewidths=2.0)
                if 'FairDICE (Fixed)' not in seen_labels:
                    import matplotlib.lines as mlines
                    fd_line_fixed = mlines.Line2D([], [], color='#9467bd', marker='o', linestyle='None',
                                          markersize=15, label='FairDICE (Fixed)', markeredgecolor='white', markeredgewidth=2.0)
                    handles.append(fd_line_fixed)
                    labels.append('FairDICE (Fixed)')
                    seen_labels.add('FairDICE (Fixed)')


            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(col_env, fontsize=20)
            ax.tick_params(axis='both', labelsize=14)
            

            if j == 0:
                ax.set_ylabel(f"{row_type}", fontsize=20)
            

            if i == 1:
                ax.set_xlabel(xlabel, fontsize=18)


    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=20, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1], w_pad=3, h_pad=3)
    output_path = os.path.join(base_path, 'replication_figure_5.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_figure_5()
