import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from data_loader_6 import load_data
import sys
import os


try:
    from scores import scores
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from scores import scores

def plot_combined_figure_6():

    fig = plt.figure(figsize=(24, 6))
    plt.subplots_adjust(wspace=0.3)
    
    fig.suptitle("MO-Hopper-3obj", fontsize=20, y=0.95)

    methods_config = [
        ('bc', '#1f77b4', 'o', 'BC(P)'), 
        ('modt', '#2ca02c', 's', 'MODT(P)'), 
        ('rvs', '#ff7f0e', '^', 'MORvS(P)')
    ]


    def setup_returns_plot(idx, title, dataset_type, fd_point):
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
        
        ax.view_init(elev=20, azim=80)


        for m, color, marker, _ in methods_config:
            _, points, _ = load_data(dataset_type, m)
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=marker, s=30, alpha=0.4, depthshade=False, linewidths=0)


        ax.scatter(*fd_point, c='#d62728', marker='o', s=60, depthshade=False, edgecolors='white', linewidths=2)
        ax.text(fd_point[0], fd_point[1], fd_point[2] + 2000, 
                f"({fd_point[0]:.1f}, {fd_point[1]:.1f}, {fd_point[2]:.1f})", 
                color='#d62728', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))
        

    def setup_simplex_plot(idx, title, dataset_type, z_min, target_score):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        ax.set_title(title, fontsize=18)
        ax.set_zlabel("NSW Score", fontsize=14, labelpad=10)
        ax.set_zlim(z_min, 18)
        
        v_w1, v_w2, v_w3 = np.array([-1, -0.5, z_min]), np.array([1, -0.5, z_min]), np.array([0, 1.0, z_min])
        

        for v1, v2 in [(v_w2, v_w1), (v_w1, v_w3), (v_w3, v_w2)]:
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [z_min, z_min], 'k-', lw=1, alpha=0.3)


        if target_score:
            ax.plot_trisurf([-1, 1, 0], [-0.5, -0.5, 1.0], [target_score]*3, color='#d62728', alpha=0.3, shade=False)
            ax.text(0, 1.0, target_score + 2.5, f"{target_score:.2f}", color='#d62728', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))


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

    setup_returns_plot(1, "Expert", "expert", np.array([1149.80613158, 1125.92841047, 1317.32675833]))
    
    setup_returns_plot(2, "Amateur", "amateur", np.array([2368.64291278, 2945.32947587, 1975.69500442]))
    
    setup_simplex_plot(3, "Expert", "expert", z_min=8, target_score=scores.get('Hopper-v3/expert/Rerun/1.0', 0))
    
    setup_simplex_plot(4, "Amateur", "amateur", z_min=6, target_score=scores.get('Hopper-v3/amateur/Rerun/1.0', 0))


    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=m[2], color='w', label=m[3], markerfacecolor=m[1], markersize=12) for m in methods_config
    ]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='FairDICE', markerfacecolor='#d62728', markersize=12))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=16)

    output_path = "/home/scur0076/PEDA/figures/replication_figure_6.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Combined Final Figure to {output_path}")

if __name__ == "__main__":
    plot_combined_figure_6()
