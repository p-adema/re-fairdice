import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import itertools


def load_summary(path: str) -> Dict:
    """Load a summary.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics(returns: List[float]) -> Tuple[float, float, float]:
    returns = np.array(returns)
    n = len(returns)
    
    utilitarian = np.sum(returns)
    
    sum_returns = np.sum(returns)
    sum_squared = np.sum(returns ** 2)
    if sum_squared > 0:
        jains_fairness = (sum_returns ** 2) / (n * sum_squared)
    else:
        jains_fairness = 0.0
    
    eps = 1e-10
    safe_returns = np.maximum(returns, eps)
    nash_social_welfare = np.sum(np.log(safe_returns))
    
    return nash_social_welfare, utilitarian, jains_fairness


def load_run_data(run_path: str, num_seeds: int = 100) -> Dict[str, List[float]]:
    nash_values = []
    utilitarian_values = []
    jains_values = []
    
    run_path = Path(run_path)
    
    found_seeds = 0
    for seed in range(num_seeds + 50):
        summary_path = run_path / f"seed_{seed}" / "eval" / "summary.json"
        if summary_path.exists():
            try:
                summary = load_summary(str(summary_path))
                returns = summary.get("best_returns", [])
                if returns:
                    nash, util, jains = compute_metrics(returns)
                    nash_values.append(nash)
                    utilitarian_values.append(util)
                    jains_values.append(jains)
                    found_seeds += 1
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
        
        if found_seeds >= num_seeds:
            break
            
    return {
        "nash": nash_values,
        "utilitarian": utilitarian_values,
        "jains": jains_values
    }


def compute_mean_and_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and 95% confidence interval."""
    values = np.array(values)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    
    z = 1.96 if confidence == 0.95 else 1.645
    ci = z * std / np.sqrt(n)
    
    return mean, ci


def scan_runs(base_dir: str, timestamp_id: str) -> Tuple[List[float], List[str]]:
    """
    Scans a directory for folders matching the timestamp_id and extracts beta values.
    
    Args:
        base_dir: Root directory containing result folders (e.g., 'results_imbalanced')
        timestamp_id: Unique timestamp/ID string to match (e.g., '20260127_004414')
        
    Returns:
        Tuple of (sorted_betas, sorted_paths)
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist.")
        return [], []

    found_runs = []
    
    beta_pattern = re.compile(r"beta([\d\.]+(?:e-?\d+)?)")

    for item in base_path.iterdir():
        if item.is_dir() and timestamp_id in item.name:
            match = beta_pattern.search(item.name)
            if match:
                beta_val = float(match.group(1))
                found_runs.append((beta_val, str(item)))
            else:
                print(f"Skipping {item.name}: matched timestamp but could not parse beta.")

    found_runs.sort(key=lambda x: x[0])
    
    if not found_runs:
        print(f"Warning: No runs found for ID '{timestamp_id}' in '{base_dir}'")
        return [], []

    betas, paths = zip(*found_runs)
    return list(betas), list(paths)


def aggregate_experiment(
    label: str, 
    base_dir: str, 
    timestamp_id: str, 
    num_seeds: int = 100
) -> Dict:
    """
    Scan for runs and aggregate data for a single experiment configuration.
    """
    betas, paths = scan_runs(base_dir, timestamp_id)
    
    if not betas:
        return None

    results = {
        "betas": betas,
        "nash_mean": [], "nash_ci": [],
        "utilitarian_mean": [], "utilitarian_ci": [],
        "jains_mean": [], "jains_ci": []
    }
    
    for run_path in paths:
        data = load_run_data(run_path, num_seeds)
        
        nash_mean, nash_ci = compute_mean_and_ci(data["nash"])
        util_mean, util_ci = compute_mean_and_ci(data["utilitarian"])
        jains_mean, jains_ci = compute_mean_and_ci(data["jains"])
        
        results["nash_mean"].append(nash_mean)
        results["nash_ci"].append(nash_ci)
        results["utilitarian_mean"].append(util_mean)
        results["utilitarian_ci"].append(util_ci)
        results["jains_mean"].append(jains_mean)
        results["jains_ci"].append(jains_ci)
        
    return results


def plot_metrics(
    method_data: Dict[str, Dict],
    output_path: str = "metrics_comparison.png",
    figsize: Tuple[int, int] = (15, 4),
    dpi: int = 150
):
    """
    Generate the comparison plot for arbitrary labels.
    """
    fixed_styles = {
        "Utilitarian (α=0)": {"color": "#1f77b4", "marker": "o"},
        "FairDICE (α=0.5)":  {"color": "#f5c242", "marker": "s"},
        "FairDICE (α=1.0)":  {"color": "#d62728", "marker": "o"},
        "FairDICE (α=1.25)": {"color": "#9467bd", "marker": "o"},
    }
    
    default_cycler = itertools.cycle(plt.cm.tab10.colors)
    marker_cycler = itertools.cycle(['o', 's', '^', 'D', 'v', 'p'])
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metric_configs = [
        ("nash_mean", "nash_ci", "Nash Social Welfare"),
        ("utilitarian_mean", "utilitarian_ci", "Utilitarian"),
        ("jains_mean", "jains_ci", "Jain's fairness")
    ]
    
    for ax, (mean_key, ci_key, title) in zip(axes, metric_configs):
        
        local_color_cycler = itertools.cycle(plt.cm.tab10.colors)
        local_marker_cycler = itertools.cycle(['o', 's', '^', 'D', 'v', 'p'])
        
        assigned_styles = {}

        for method_name, data in method_data.items():
            if data is None: continue
            
            betas = np.array(data["betas"])
            means = np.array(data[mean_key])
            cis = np.array(data[ci_key])
            
            if method_name in fixed_styles:
                style = fixed_styles[method_name]
                color = style["color"]
                marker = style["marker"]
                next(local_color_cycler)
                next(local_marker_cycler)
            else:
                color = next(local_color_cycler)
                marker = next(local_marker_cycler)
            
            sort_idx = np.argsort(betas)
            betas = betas[sort_idx]
            means = means[sort_idx]
            cis = cis[sort_idx]
            
            ax.errorbar(
                betas, means, yerr=cis,
                label=method_name,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.5,
                capsize=3,
                capthick=1,
                alpha=0.9
            )
        
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, which="both", ls="--")
    
    handles, labels = axes[0].get_legend_handles_labels()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def load_minecart_run_data(run_path: str, num_seeds: int = 10) -> List[List[float]]:
    """
    Load all seeds' best_returns from a Minecart run.
    
    Returns:
        List of [ore1, ore2, fuel] vectors, one per seed.
    """
    returns_list = []
    run_path = Path(run_path)
    
    for seed in range(num_seeds + 50):
        summary_path = run_path / f"seed_{seed}" / "eval" / "summary.json"
        if summary_path.exists():
            try:
                summary = load_summary(str(summary_path))
                returns = summary.get("best_returns", [])
                if returns and len(returns) == 3:
                    returns_list.append(returns)
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
        
        if len(returns_list) >= num_seeds:
            break
    
    return returns_list


def scan_minecart_runs(base_dir: str, timestamp_id: str) -> Dict[float, List[Tuple[float, str]]]:
    """
    Scans directory for Minecart runs, grouping by alpha.
    
    Args:
        base_dir: Root directory containing result folders
        timestamp_id: Unique timestamp/ID string to match
        
    Returns:
        Dict mapping alpha -> list of (beta, path) tuples, sorted by beta
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist.")
        return {}
    
    beta_pattern = re.compile(r"beta([\d\.]+(?:e-?\d+)?)")
    alpha_pattern = re.compile(r"alpha([\d\.]+(?:e-?\d+)?)")
    
    runs_by_alpha = {}
    
    for item in base_path.iterdir():
        if item.is_dir() and timestamp_id in item.name:
            beta_match = beta_pattern.search(item.name)
            alpha_match = alpha_pattern.search(item.name)
            
            if beta_match and alpha_match:
                beta_val = float(beta_match.group(1))
                alpha_val = float(alpha_match.group(1))
                
                if alpha_val not in runs_by_alpha:
                    runs_by_alpha[alpha_val] = []
                runs_by_alpha[alpha_val].append((beta_val, str(item)))
            else:
                print(f"Skipping {item.name}: could not parse beta/alpha.")
    
    for alpha in runs_by_alpha:
        runs_by_alpha[alpha].sort(key=lambda x: x[0])
    
    return runs_by_alpha


def compute_pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute 2D Pareto front (maximization).
    
    Args:
        points: Nx2 array of points
    
    Returns:
        Array of Pareto-optimal points, sorted by first coordinate
    """
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points)
    
    points = np.unique(points, axis=0)
    
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]
    
    pareto_front = [sorted_points[0]]
    max_y = sorted_points[0, 1]
    
    for point in sorted_points[1:]:
        if point[1] > max_y:
            pareto_front.append(point)
            max_y = point[1]
    
    pareto_front = np.array(pareto_front)
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
    
    return pareto_front


def plot_minecart_pareto(
    base_dir: str,
    timestamp_id: str,
    output_path: str = "minecart_pareto.pdf",
    num_seeds: int = 3,
    show_pareto_front: bool = False,
    objective_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 150
):
    """
    Plot Pareto fronts for Minecart environment.
    
    Creates a 2-row (alpha=0, alpha=1) x 3-column (objective pairs) grid.
    Each point represents a single seed's result. Points are colored by beta value.
    
    Args:
        base_dir: Directory containing the run folders
        timestamp_id: Timestamp prefix to filter runs (e.g., '20260129_024138')
        output_path: Where to save the plot
        num_seeds: Number of seeds per run
        show_pareto_front: If True, compute and draw Pareto front line
        objective_names: Names for the 3 objectives (default: ["Ore1", "Ore2", "Fuel"])
        figsize: Figure size
        dpi: Resolution
    """
    if objective_names is None:
        objective_names = ["Ore1", "Ore2", "Fuel"]
    
    obj_pairs = [(0, 1), (0, 2), (1, 2)]
    
    runs_by_alpha = scan_minecart_runs(base_dir, timestamp_id)
    
    if not runs_by_alpha:
        print("No runs found.")
        return
    
    alphas = sorted(runs_by_alpha.keys())
    n_rows = len(alphas)
    
    print(f"Found {n_rows} alpha values: {alphas}")
    
    all_betas = set()
    for alpha in alphas:
        for beta, _ in runs_by_alpha[alpha]:
            all_betas.add(beta)
    all_betas = sorted(all_betas)
    
    cmap = plt.cm.viridis
    log_betas = np.log10(all_betas)
    norm = plt.Normalize(log_betas.min(), log_betas.max())
    beta_to_color = {beta: cmap(norm(np.log10(beta))) for beta in all_betas}
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, alpha in enumerate(alphas):
        beta_runs = runs_by_alpha[alpha]
        
        for col_idx, (obj_i, obj_j) in enumerate(obj_pairs):
            ax = axes[row_idx, col_idx]
            
            all_points = []
            
            for beta, run_path in beta_runs:
                returns_list = load_minecart_run_data(run_path, num_seeds)
                
                if not returns_list:
                    print(f"  No data for alpha={alpha}, beta={beta}")
                    continue
                
                points = np.array([[r[obj_i], r[obj_j]] for r in returns_list])
                all_points.extend(points.tolist())
                
                color = beta_to_color[beta]
                
                ax.scatter(
                    points[:, 0], points[:, 1],
                    c=[color] * len(points),
                    label=f"β={beta}" if col_idx == 0 else None,
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidths=0.5
                )
            
            if show_pareto_front and all_points:
                all_points_arr = np.array(all_points)
                pareto = compute_pareto_front_2d(all_points_arr)
                if len(pareto) > 1:
                    ax.plot(
                        pareto[:, 0], pareto[:, 1],
                        'k--', linewidth=1.5, alpha=0.7,
                        label='Pareto Front' if col_idx == 0 else None
                    )
            
            ax.set_xlabel(objective_names[obj_i], fontsize=11)
            ax.set_ylabel(objective_names[obj_j], fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if row_idx == 0:
                ax.set_title(f"{objective_names[obj_i]} vs {objective_names[obj_j]}", fontsize=12)
        
        axes[row_idx, 0].annotate(
            f"α = {alpha}",
            xy=(-0.35, 0.5),
            xycoords='axes fraction',
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90
        )
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.88)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$\beta$', fontsize=11)
    cbar.set_ticks(log_betas)
    cbar.set_ticklabels([f"{b}" for b in all_betas])
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def compute_nsw(returns: List[float]) -> float:
    """Compute Nash Social Welfare (sum of logs) from a returns vector."""
    eps = 1e-10
    safe_returns = np.maximum(np.array(returns), eps)
    return np.sum(np.log(safe_returns))


def print_nsw_table(
    runs_by_alpha: Dict[float, List[Tuple[float, str]]],
    num_seeds: int = 3
):
    """
    Print a table of Nash Social Welfare values.
    Rows = alphas, Columns = betas.
    Values are mean ± std over seeds.
    """
    alphas = sorted(runs_by_alpha.keys())
    
    all_betas = set()
    for alpha in alphas:
        for beta, _ in runs_by_alpha[alpha]:
            all_betas.add(beta)
    all_betas = sorted(all_betas)
    
    nsw_data = {}
    
    for alpha in alphas:
        beta_runs = {beta: path for beta, path in runs_by_alpha[alpha]}
        for beta in all_betas:
            if beta in beta_runs:
                returns_list = load_minecart_run_data(beta_runs[beta], num_seeds)
                if returns_list:
                    nsw_values = [compute_nsw(r) for r in returns_list]
                    nsw_data[(alpha, beta)] = (np.mean(nsw_values), np.std(nsw_values))
                else:
                    nsw_data[(alpha, beta)] = (np.nan, np.nan)
            else:
                nsw_data[(alpha, beta)] = (np.nan, np.nan)
    
    print("\n" + "=" * 140)
    print("Nash Social Welfare (mean ± std)")
    print("=" * 140)
    
    header = "     α \\ β  "
    for beta in all_betas:
        header += f" | {beta:>16}"
    print(header)
    print("-" * len(header))
    
    for alpha in alphas:
        row = f"{alpha:>12.1f}"
        for beta in all_betas:
            mean, std = nsw_data[(alpha, beta)]
            if np.isnan(mean):
                row += f" | {'N/A':>16}"
            else:
                row += f" | {mean:>7.2f} ± {std:<5.2f}"
        print(row)
    
    print("=" * 80 + "\n")


def plot_minecart_pareto_multi_timestamp(
    base_dir: str,
    alpha_timestamps: Dict[float, str],
    output_path: str = "minecart_pareto.pdf",
    num_seeds: int = 3,
    show_pareto_front: bool = False,
    objective_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 150
):
    """
    Plot Pareto fronts for Minecart when different alphas have different timestamps.
    
    Args:
        base_dir: Directory containing the run folders
        alpha_timestamps: Dict mapping alpha value to its timestamp ID
                         e.g., {0.0: "20260129_024138", 1.0: "20260129_024219"}
        output_path: Where to save the plot
        num_seeds: Number of seeds per run
        show_pareto_front: If True, compute and draw Pareto front line
        objective_names: Names for the 3 objectives
        figsize: Figure size
        dpi: Resolution
    """
    if objective_names is None:
        objective_names = ["Ore1", "Ore2", "Fuel"]
    
    obj_pairs = [(0, 1), (0, 2), (1, 2)]
    
    runs_by_alpha = {}
    for alpha, ts_id in alpha_timestamps.items():
        alpha_runs = scan_minecart_runs(base_dir, ts_id)
        if alpha in alpha_runs:
            runs_by_alpha[alpha] = alpha_runs[alpha]
        elif len(alpha_runs) == 1:
            runs_by_alpha[alpha] = list(alpha_runs.values())[0]
    
    if not runs_by_alpha:
        print("No runs found.")
        return
    
    alphas = sorted(runs_by_alpha.keys())
    n_rows = len(alphas)
    
    print(f"Found {n_rows} alpha values: {alphas}")
    
    print_nsw_table(runs_by_alpha, num_seeds)
    
    all_betas = set()
    for alpha in alphas:
        for beta, _ in runs_by_alpha[alpha]:
            all_betas.add(beta)
    all_betas = sorted(all_betas)
    
    cmap = plt.cm.viridis
    log_betas = np.log10(all_betas)
    norm = plt.Normalize(log_betas.min(), log_betas.max())
    beta_to_color = {beta: cmap(norm(np.log10(beta))) for beta in all_betas}
    
    all_data_by_col = {col_idx: [] for col_idx in range(3)}
    
    for alpha in alphas:
        beta_runs = runs_by_alpha[alpha]
        for col_idx, (obj_i, obj_j) in enumerate(obj_pairs):
            for beta, run_path in beta_runs:
                returns_list = load_minecart_run_data(run_path, num_seeds)
                if returns_list:
                    points = [[r[obj_i], r[obj_j]] for r in returns_list]
                    all_data_by_col[col_idx].extend(points)
    
    col_ranges = {}
    for col_idx in range(3):
        if all_data_by_col[col_idx]:
            points = np.array(all_data_by_col[col_idx])
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            col_ranges[col_idx] = {
                'xlim': (x_min - x_pad, x_max + x_pad),
                'ylim': (y_min - y_pad, y_max + y_pad)
            }
        else:
            col_ranges[col_idx] = None
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, alpha in enumerate(alphas):
        beta_runs = runs_by_alpha[alpha]
        
        for col_idx, (obj_i, obj_j) in enumerate(obj_pairs):
            ax = axes[row_idx, col_idx]
            all_points = []
            
            for beta, run_path in beta_runs:
                returns_list = load_minecart_run_data(run_path, num_seeds)
                
                if not returns_list:
                    print(f"  No data for alpha={alpha}, beta={beta}")
                    continue
                
                points = np.array([[r[obj_i], r[obj_j]] for r in returns_list])
                all_points.extend(points.tolist())
                
                color = beta_to_color[beta]
                
                ax.scatter(
                    points[:, 0], points[:, 1],
                    c=[color] * len(points),
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidths=0.5
                )
            
            if show_pareto_front and all_points:
                all_points_arr = np.array(all_points)
                pareto = compute_pareto_front_2d(all_points_arr)
                if len(pareto) > 1:
                    ax.plot(
                        pareto[:, 0], pareto[:, 1],
                        'k--', linewidth=1.5, alpha=0.7
                    )
            
            if col_ranges[col_idx]:
                ax.set_xlim(col_ranges[col_idx]['xlim'])
                ax.set_ylim(col_ranges[col_idx]['ylim'])
            
            if row_idx != 0:
                ax.set_xlabel(objective_names[obj_i], fontsize=11)
            ax.set_ylabel(objective_names[obj_j], fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if row_idx == 0:
                ax.set_title(f"{objective_names[obj_i]} vs {objective_names[obj_j]}", fontsize=12)
        
        axes[row_idx, 0].annotate(
            f"α = {alpha}",
            xy=(-0.35, 0.5),
            xycoords='axes fraction',
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90
        )
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.88)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$\beta$', fontsize=11)
    cbar.set_ticks(log_betas)
    cbar.set_ticklabels([f"{b}" for b in all_betas])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    is_minecart = True
    
    if is_minecart:
        
        BASE_DIR = "./results_mcd" 
        
        ALPHA_TIMESTAMPS = {
            0.0: "20260129_024138",
            1.0: "20260129_024219",
        }
        
        NUM_SEEDS = 3
        OUTPUT_FILE = "minecart_pareto.pdf"
        SHOW_PARETO_FRONT = True
        
        plot_minecart_pareto_multi_timestamp(
            base_dir=BASE_DIR,
            alpha_timestamps=ALPHA_TIMESTAMPS,
            output_path=OUTPUT_FILE,
            num_seeds=NUM_SEEDS,
            show_pareto_front=SHOW_PARETO_FRONT,
            objective_names=["Ore1", "Ore2", "Fuel"],
        )
        
    else:       
        experiments_config = [
            {
                "label": "Utilitarian (α=0)", 
                "dir": "results_mcd", 
                "id": "20260123_164918"
            },
            {
                "label": "FairDICE (α=1.0)",
                "dir": "results_mcd",
                "id": "20260123_165043",
            },
        ]

        NUM_SEEDS = 50
        OUTPUT_FILE = "fourrooms_no_bug.pdf"

        aggregated_data = {}

        for exp in experiments_config:
            data = aggregate_experiment(
                label=exp["label"],
                base_dir=exp["dir"],
                timestamp_id=exp["id"],
                num_seeds=NUM_SEEDS
            )
            if data:
                aggregated_data[exp["label"]] = data

        if aggregated_data:
            plot_metrics(aggregated_data, output_path=OUTPUT_FILE)
        else:
            print("No data found to plot.")