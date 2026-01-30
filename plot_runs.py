"""
Plotting script for generating fairness metrics comparison plots.
Supports automated scanning of beta sweeps via timestamps and custom labeling.
"""

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
    """
    Compute the three evaluation metrics from returns.
    """
    returns = np.array(returns)
    n = len(returns)
    
    # Utilitarian welfare: sum of returns
    utilitarian = np.sum(returns)
    
    # Jain's Fairness Index: (sum R_i)^2 / (n * sum R_i^2)
    sum_returns = np.sum(returns)
    sum_squared = np.sum(returns ** 2)
    if sum_squared > 0:
        jains_fairness = (sum_returns ** 2) / (n * sum_squared)
    else:
        jains_fairness = 0.0
    
    # Nash Social Welfare: sum of log(R_i)
    eps = 1e-10
    safe_returns = np.maximum(returns, eps)
    nash_social_welfare = np.sum(np.log(safe_returns))
    
    return nash_social_welfare, utilitarian, jains_fairness

def load_run_data(run_path: str, num_seeds: int = 100) -> Dict[str, List[float]]:
    """
    Load data from a single run (all seeds).
    """
    nash_values = []
    utilitarian_values = []
    jains_values = []
    
    run_path = Path(run_path)
    
    # Handle cases where num_seeds is larger than actual found seeds without crashing
    # or iterate directory if seed folder naming isn't strictly sequential
    found_seeds = 0
    for seed in range(num_seeds + 50): # Look a bit beyond to be safe or strictly scan
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
    
    # Regex to extract beta. Matches "beta" followed by digits, dots, or scientific notation
    # Example matches: beta0.1, beta100.0, beta1e-4
    beta_pattern = re.compile(r"beta([\d\.]+(?:e-?\d+)?)")

    for item in base_path.iterdir():
        if item.is_dir() and timestamp_id in item.name:
            match = beta_pattern.search(item.name)
            if match:
                beta_val = float(match.group(1))
                found_runs.append((beta_val, str(item)))
            else:
                print(f"Skipping {item.name}: matched timestamp but could not parse beta.")

    # Sort by beta
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
    print(f"Processing '{label}' (ID: {timestamp_id})...")
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
    # Pre-defined styles for consistency if labels match; otherwise fallback to cycler
    fixed_styles = {
        "Utilitarian (α=0)": {"color": "#1f77b4", "marker": "o"}, # Blue
        "FairDICE (α=0.5)":  {"color": "#f5c242", "marker": "s"}, # Yellow
        "FairDICE (α=1.0)":  {"color": "#d62728", "marker": "o"}, # Red
        "FairDICE (α=1.25)": {"color": "#9467bd", "marker": "o"}, # Purple
    }
    
    # Fallback cycler for custom labels
    # Uses Tableu 10 colors excluding red/blue if they are already taken, or just a standard list
    default_cycler = itertools.cycle(plt.cm.tab10.colors)
    marker_cycler = itertools.cycle(['o', 's', '^', 'D', 'v', 'p'])
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metric_configs = [
        ("nash_mean", "nash_ci", "Nash Social Welfare"),
        ("utilitarian_mean", "utilitarian_ci", "Utilitarian"),
        ("jains_mean", "jains_ci", "Jain's fairness")
    ]
    
    for ax, (mean_key, ci_key, title) in zip(axes, metric_configs):
        
        # Reset cyclers for each subplot so colors match across plots
        local_color_cycler = itertools.cycle(plt.cm.tab10.colors)
        local_marker_cycler = itertools.cycle(['o', 's', '^', 'D', 'v', 'p'])
        
        # Keep track of assigned styles to ensure consistency across subplots
        assigned_styles = {}

        for method_name, data in method_data.items():
            if data is None: continue
            
            betas = np.array(data["betas"])
            means = np.array(data[mean_key])
            cis = np.array(data[ci_key])
            
            # Determine style
            if method_name in fixed_styles:
                style = fixed_styles[method_name]
                color = style["color"]
                marker = style["marker"]
                # Advance cycler anyway to keep sync if mixed (optional)
                next(local_color_cycler)
                next(local_marker_cycler)
            else:
                # Get dynamic style
                color = next(local_color_cycler)
                marker = next(local_marker_cycler)
            
            # Sort by beta
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
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(method_data), 4),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    
    experiments_config = [
        {
            "label": "Utilitarian (80/10/10 dataset)", 
            "dir": "results_imbalanced", 
            "id": "20260127_004414"
        },
        {
            "label": "Utilitarian (33/33/33 dataset)", 
            "dir": "results_fix_sample", 
            "id": "20260129_190522"
        },
        {
            "label": "FairDICE α=1.0 (80/10/10 dataset)", 
            "dir": "results_imbalanced", 
            "id": "20260127_004354"
        },
        {
            "label": "FairDICE α=1.0 (33/33/34 dataset)", 
            "dir": "results_fix_sample", 
            "id": "20260129_190517"
        },
    ]

    NUM_SEEDS = 50
    OUTPUT_FILE = "fourrooms_uniform_vs_greedy.pdf"

    experiments_config = [
        {
            "label": "Utilitarian (80/10/10 dataset)", 
            "dir": "results_imbalanced", 
            "id": "20260127_004414"
        },
        {
            "label": "Utilitarian (33/33/33 dataset)", 
            "dir": "results_fix_sample", 
            "id": "20260129_190522"
        },
        {
            "label": "FairDICE α=1.0 (uniform dataset)", 
            "dir": "results_uniform", 
            "id": "20260130_192921"
        }
    ]

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