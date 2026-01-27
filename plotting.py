"""
Plotting script for generating fairness metrics comparison plots.

This script generates plots similar to Figure 2 from the paper, showing:
- Nash Social Welfare
- Utilitarian Welfare  
- Jain's Fairness Index

across different beta values for Utilitarian and FairDICE with various alpha values.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def load_summary(path: str) -> Dict:
    """Load a summary.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics(returns: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three evaluation metrics from returns.
    
    Args:
        returns: List of returns R_i for each objective
        
    Returns:
        Tuple of (nash_social_welfare, utilitarian, jains_fairness)
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
    # Handle non-positive returns by using a small epsilon
    eps = 1e-10
    safe_returns = np.maximum(returns, eps)
    nash_social_welfare = np.sum(np.log(safe_returns))
    
    return nash_social_welfare, utilitarian, jains_fairness


def load_run_data(run_path: str, num_seeds: int = 100) -> Dict[str, List[float]]:
    """
    Load data from a single run (all seeds).
    
    Args:
        run_path: Path to the run folder containing seed_i/eval/summary.json
        num_seeds: Number of seeds to load (0 to num_seeds-1)
        
    Returns:
        Dictionary with lists of metrics across seeds
    """
    nash_values = []
    utilitarian_values = []
    jains_values = []
    
    run_path = Path(run_path)
    
    for seed in range(num_seeds):
        summary_path = run_path / f"seed_{seed}" / "eval" / "summary.json"
        if summary_path.exists():
            summary = load_summary(str(summary_path))
            returns = summary.get("best_returns", [])
            if returns:
                nash, util, jains = compute_metrics(returns)
                nash_values.append(nash)
                utilitarian_values.append(util)
                jains_values.append(jains)
    
    return {
        "nash": nash_values,
        "utilitarian": utilitarian_values,
        "jains": jains_values
    }


def compute_mean_and_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute mean and 95% confidence interval.
    
    Args:
        values: List of values
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, ci_half_width)
    """
    values = np.array(values)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # 95% CI: mean ± 1.96 * std / sqrt(n)
    z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
    ci = z * std / np.sqrt(n)
    
    return mean, ci


def aggregate_runs(run_paths: List[str], betas: List[float], num_seeds: int = 100) -> Dict:
    """
    Aggregate data from multiple runs (different beta values).
    
    Args:
        run_paths: List of paths to run folders, one per beta value
        betas: List of beta values corresponding to run_paths
        num_seeds: Number of seeds per run
        
    Returns:
        Dictionary with aggregated metrics (means and CIs) for each beta
    """
    results = {
        "betas": betas,
        "nash_mean": [],
        "nash_ci": [],
        "utilitarian_mean": [],
        "utilitarian_ci": [],
        "jains_mean": [],
        "jains_ci": []
    }
    
    for run_path in run_paths:
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
    Generate the three-panel comparison plot.
    
    Args:
        method_data: Dictionary mapping method names to their aggregated results
                     e.g., {"Utilitarian (α=0)": {...}, "FairDICE (α=1.0)": {...}}
        output_path: Path to save the figure
        figsize: Figure size (width, height)
        dpi: Resolution
    """
    # Color scheme similar to the paper
    colors = {
        "Utilitarian (α=0)": "#1f77b4",      # Blue
        "FairDICE (α=0.5)": "#f5c242",        # Yellow/Gold
        "FairDICE (α=1.0)": "#d62728",        # Red
        "FairDICE (α=1.25)": "#9467bd"        # Purple
    }
    
    markers = {
        "Utilitarian (α=0)": "o",
        "FairDICE (α=0.5)": "s",
        "FairDICE (α=1.0)": "o",
        "FairDICE (α=1.25)": "o"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metric_configs = [
        ("nash_mean", "nash_ci", "Nash Social Welfare"),
        ("utilitarian_mean", "utilitarian_ci", "Utilitarian"),
        ("jains_mean", "jains_ci", "Jain's fairness")
    ]
    
    for ax, (mean_key, ci_key, title) in zip(axes, metric_configs):
        for method_name, data in method_data.items():
            betas = np.array(data["betas"])
            means = np.array(data[mean_key])
            cis = np.array(data[ci_key])
            
            color = colors.get(method_name, "#333333")
            marker = markers.get(method_name, "o")
            
            # Sort by beta for proper line plotting
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
                capthick=1
            )
        
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Add legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(method_data),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def generate_plot_from_paths(
    utilitarian_paths: List[str],
    fairdice_alpha1_paths: List[str],
    fairdice_alpha1_25_paths: List[str],
    betas: List[float],
    output_path: str = "metrics_comparison.png",
    fairdice_alpha0_5_paths: List[str] = None,
    num_seeds: int = 100
):
    """
    Main function to generate the comparison plot from run paths.
    
    Args:
        utilitarian_paths: List of paths for Utilitarian (α=0) runs, one per beta
        fairdice_alpha1_paths: List of paths for FairDICE (α=1.0) runs, one per beta
        fairdice_alpha1_25_paths: List of paths for FairDICE (α=1.25) runs, one per beta
        betas: List of beta values corresponding to the paths
        output_path: Output path for the plot
        fairdice_alpha0_5_paths: Optional paths for FairDICE (α=0.5) runs
        num_seeds: Number of seeds per run (default 100)
    """
    method_data = {}
    
    print("Loading Utilitarian (α=0) data...")
    method_data["Utilitarian (α=0)"] = aggregate_runs(utilitarian_paths, betas, num_seeds)
    
    if fairdice_alpha0_5_paths is not None:
        print("Loading FairDICE (α=0.5) data...")
        method_data["FairDICE (α=0.5)"] = aggregate_runs(fairdice_alpha0_5_paths, betas, num_seeds)
    
    print("Loading FairDICE (α=1.0) data...")
    method_data["FairDICE (α=1.0)"] = aggregate_runs(fairdice_alpha1_paths, betas, num_seeds)
    
    print("Loading FairDICE (α=1.25) data...")
    method_data["FairDICE (α=1.25)"] = aggregate_runs(fairdice_alpha1_25_paths, betas, num_seeds)
    
    print("Generating plot...")
    plot_metrics(method_data, output_path)


# Example usage
if __name__ == "__main__":
    # Beta values (x-axis)
    # betas = [1e-4, 1e-3, 1e-2, 0.05, 1e-1, 1e0, 5, 1e1, 1e2]
    
    # Example path structure (modify these to match your actual paths):
    # utilitarian_paths = [
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta0.0001_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta0.001_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta0.01_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta0.05_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta0.1_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta1.0_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta5.0_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta10.0_alpha0.0_50seeds",
    #     "results_nofix/20260124_182146_FairDICE_MO-FourRooms-v0_beta100.0_alpha0.0_50seeds",

    # ]

    # fairdice_alpha0_5_paths = [
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta0.0001_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta0.001_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta0.01_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta0.05_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta0.1_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta1.0_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta5.0_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta10.0_alpha0.5_50seeds",
    #     "results_nofix/20260124_182155_FairDICE_MO-FourRooms-v0_beta100.0_alpha0.5_50seeds",
    # ]
    
    # fairdice_alpha1_paths = [
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta0.0001_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta0.001_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta0.01_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta0.05_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta0.1_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta1.0_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta5.0_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta10.0_alpha1.0_50seeds",
    #     "results_nofix/20260124_182209_FairDICE_MO-FourRooms-v0_beta100.0_alpha1.0_50seeds",
    # ]

    # fairdice_alpha1_25_paths = [
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta0.0001_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta0.001_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta0.01_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta0.05_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta0.1_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta1.0_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta5.0_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta10.0_alpha1.25_50seeds",
    #     "results_nofix/20260124_182217_FairDICE_MO-FourRooms-v0_beta100.0_alpha1.25_50seeds",
    # ]

    fix_paths = [
        "results_fix_reconfirm/20260126_152651_FairDICE_MO-FourRooms-v0_beta0.01_alpha1.0_20seeds",
        "results_fix_reconfirm/20260126_152651_FairDICE_MO-FourRooms-v0_beta1.0_alpha1.0_20seeds",
        "results_fix_reconfirm/20260126_152651_FairDICE_MO-FourRooms-v0_beta10.0_alpha1.0_20seeds",
    ]
    nofix_paths = [
        "results_nofix_reconfirm/20260126_152543_FairDICE_MO-FourRooms-v0_beta0.01_alpha1.0_20seeds",
        "results_nofix_reconfirm/20260126_152543_FairDICE_MO-FourRooms-v0_beta1.0_alpha1.0_20seeds",
        "results_nofix_reconfirm/20260126_152543_FairDICE_MO-FourRooms-v0_beta10.0_alpha1.0_20seeds"
    ]
    betas = [0.01, 1.0, 10.0]
    
    # For demonstration, here's how you would call the function:
    # generate_plot_from_paths(
    #     utilitarian_paths=utilitarian_paths,
    #     fairdice_alpha1_paths=fairdice_alpha1_paths,
    #     fairdice_alpha1_25_paths=fairdice_alpha1_25_paths,
    #     betas=betas,
    #     output_path="metrics_comparison_nofix.png",
    #     fairdice_alpha0_5_paths=fairdice_alpha0_5_paths,  # Optional
    #     num_seeds=100
    # )
    generate_plot_from_paths(
        utilitarian_paths=fix_paths,
        fairdice_alpha1_paths=fix_paths,
        fairdice_alpha1_25_paths=nofix_paths,
        betas=betas,
        output_path="metrics_comparison_reconfirm.png",
        fairdice_alpha0_5_paths=None,  # Optional
        num_seeds=20
    )
