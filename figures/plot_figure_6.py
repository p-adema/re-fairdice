import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from data_loader_6 import load_data
import sys
import os


import json

try:
    pass
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def plot_combined_figure_6():

    fig = plt.figure(figsize=(24, 6))
    plt.subplots_adjust(wspace=0.3)
    
    fig.suptitle("MO-Hopper-3obj", fontsize=20, y=0.95)

    methods_config = [
        ('bc', '#1f77b4', 'o', 'BC(P)'), 
        ('modt', '#2ca02c', 's', 'MODT(P)'), 
        ('rvs', '#ff7f0e', '^', 'MORvS(P)')
    ]

    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, 'scores.json'), 'r') as f:
        scores_data = json.load(f)


    def setup_returns_plot(idx, title, dataset_type, rerun_point, fixed_point, expert):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        ax.set_title(title, fontsize=18)
        
        ax.set_xlim(0, 4500)
        ax.set_ylim(0, 4000)
        ax.set_zlim(0, 3000)
        
        ax.set_xlabel("Speed", fontsize=14, labelpad=10)
        ax.set_ylabel("Height", fontsize=14, labelpad=10)
        ax.set_zlabel("Energy", fontsize=14, labelpad=10)
        
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_yticks([0, 1000, 2000, 3000, 4000])
        ax.set_zticks([0, 1000, 2000, 3000])
        
        ax.view_init(elev=20, azim=67)


        for m, color, marker, _ in methods_config:
            _, points, _ = load_data(dataset_type, m)
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=marker, s=30, alpha=0.4, depthshade=False, linewidths=0)
    
        if expert:
            if rerun_point is not None:
                ax.scatter(*rerun_point, c='#d62728', marker='o', s=60, depthshade=False, edgecolors='white', linewidths=2)
                ax.text(rerun_point[0] - 1000, rerun_point[1] - 2000, rerun_point[2] + 1900, f"({rerun_point[0]:.1f}, {rerun_point[1]:.1f}, {rerun_point[2]:.1f})", 
                        color='#d62728', fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))

            if fixed_point is not None:
                ax.scatter(*fixed_point, c='#9467bd', marker='o', s=60, depthshade=False, edgecolors='white', linewidths=2)
                ax.text(rerun_point[0] + 1000, rerun_point[1], rerun_point[2] + 2050, f"({fixed_point[0]:.1f}, {fixed_point[1]:.1f}, {fixed_point[2]:.1f})", 
                        color='#9467bd', fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='#9467bd', boxstyle='round,pad=0.3'))
        else:
            if rerun_point is not None:
                ax.scatter(*rerun_point, c='#d62728', marker='o', s=60, depthshade=False, edgecolors='white', linewidths=2)
                ax.text(rerun_point[0], rerun_point[1], rerun_point[2] + 2000, f"({rerun_point[0]:.1f}, {rerun_point[1]:.1f}, {rerun_point[2]:.1f})", 
                        color='#d62728', fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))

            if fixed_point is not None:
                ax.scatter(*fixed_point, c='#9467bd', marker='o', s=60, depthshade=False, edgecolors='white', linewidths=2)
                ax.text(rerun_point[0], rerun_point[1] - 2000, rerun_point[2] + 2000, f"({fixed_point[0]:.1f}, {fixed_point[1]:.1f}, {fixed_point[2]:.1f})", 
                        color='#9467bd', fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='#9467bd', boxstyle='round,pad=0.3'))
        

    def setup_simplex_plot(idx, title, dataset_type, z_min, rerun_score, fixed_score):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        ax.set_title(title, fontsize=18)
        ax.set_zlabel("NSW Score", fontsize=14, labelpad=10)
        ax.set_zlim(z_min, 18)
        
        v_w1, v_w2, v_w3 = np.array([-1, -0.5, z_min]), np.array([1, -0.5, z_min]), np.array([0, 1.0, z_min])
        

        for v1, v2 in [(v_w2, v_w1), (v_w1, v_w3), (v_w3, v_w2)]:
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [z_min, z_min], 'k-', lw=1, alpha=0.3)


        if rerun_score > fixed_score:
            # Fixed is lower, plot full
            ax.plot_trisurf([-1, 1, 0], [-0.5, -0.5, 1.0], [fixed_score]*3, color='#9467bd', alpha=0.3, shade=False)
            ax.text(-0.2, 1.0, fixed_score - 2.0, f"{fixed_score:.2f}", color='#9467bd', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='#9467bd', boxstyle='round,pad=0.3'))
            
            # Rerun is higher, plot cut (left 3/4)
            ax.plot_trisurf([-1, 0.5, 0], [-0.5, -0.5, 1.0], [rerun_score]*3, color='#d62728', alpha=0.3, shade=False)
            ax.text(-0.2, 1.0, rerun_score + 2.0, f"{rerun_score:.2f}", color='#d62728', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))
        else:
            # Rerun is lower (or equal), plot full
            ax.plot_trisurf([-1, 1, 0], [-0.5, -0.5, 1.0], [rerun_score]*3, color='#d62728', alpha=0.3, shade=False)
            ax.text(-0.2, 1.0, rerun_score + 2.0, f"{rerun_score:.2f}", color='#d62728', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))
            
            # Fixed is higher, plot cut (left 3/4)
            ax.plot_trisurf([-1, 0.5, 0], [-0.5, -0.5, 1.0], [fixed_score]*3, color='#9467bd', alpha=0.3, shade=False)
            ax.text(-0.2, 1.0, fixed_score - 2.0, f"{fixed_score:.2f}", color='#9467bd', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='#9467bd', boxstyle='round,pad=0.3'))


        for m, color, marker, _ in methods_config:
            weights, _, nsw_scores = load_data(dataset_type, m)
            if len(nsw_scores) > 0:
                px = weights[:, 1] - weights[:, 0]
                py = 1.5 * weights[:, 2] - 0.5
                ax.scatter(px, py, nsw_scores, c=color, marker=marker, s=30, alpha=0.5, depthshade=False, linewidths=0)


        ax.text(v_w1[0], v_w1[1], z_min, "w1 = 1", fontsize=10, ha='right')
        ax.text(v_w2[0], v_w2[1], z_min, "w2 = 1", fontsize=10, ha='left')
        ax.text(v_w3[0], v_w3[1], z_min, "w3 = 1", fontsize=10, ha='center', va='bottom')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text2D(0.5, 0.05, "Preference Weight Simplex", transform=ax.transAxes, ha='center', fontsize=14)
        ax.view_init(elev=20, azim=45)

    setup_returns_plot(1, "Expert", "expert", 
                       np.array(scores_data['expert']['Hopper-v3']['Raw_Rerun']),
                       np.array(scores_data['expert']['Hopper-v3']['Raw_Fixed']),
                       expert=True)
    
    setup_returns_plot(2, "Amateur", "amateur", 
                       np.array(scores_data['amateur']['Hopper-v3']['Raw_Rerun']),
                       np.array(scores_data['amateur']['Hopper-v3']['Raw_Fixed']),
                       expert=False)
    
    setup_simplex_plot(3, "Expert", "expert", z_min=8, 
                       rerun_score=scores_data['expert']['Hopper-v3']['Rerun'],
                       fixed_score=scores_data['expert']['Hopper-v3']['Fixed'])
    
    setup_simplex_plot(4, "Amateur", "amateur", z_min=6, 
                       rerun_score=scores_data['amateur']['Hopper-v3']['Rerun'],
                       fixed_score=scores_data['amateur']['Hopper-v3']['Fixed'])


    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=m[2], color='w', label=m[3], markerfacecolor=m[1], markersize=12) for m in methods_config
    ]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='FairDICE (Rerun)', markerfacecolor='#d62728', markersize=12))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='FairDICE (Fixed)', markerfacecolor='#9467bd', markersize=12))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=16)

    output_path = os.path.join(base_path, "replication_figure_6.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Combined Final Figure to {output_path}")

if __name__ == "__main__":
    plot_combined_figure_6()
