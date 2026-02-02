"""
Algorithm Comparison Framework for Construction Heuristics.

This module provides a general framework for comparing construction heuristic algorithms
on the SCF-PDP problem across multiple instance sizes.
"""

import gc
import time
from datetime import datetime
from enum import StrEnum
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.stats import wilcoxon, norm

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.utils import find_project_root


class InstanceType(StrEnum):
    COMPETITION = "competition"
    TEST = "test"
    TRAIN = "train"


@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm to test."""
    name: str  # e.g., "Greedy", "Randomized-k10"
    algorithm_class: type
    init_kwargs: dict


@dataclass
class InstanceResult:
    """Results for a single instance run."""
    instance_name: str
    instance_size: int
    algorithm_name: str
    objective_value: float
    construction_time: float
    total_time: float
    parsing_time: float


@dataclass
class ComparisonResults:
    """Comparison between two algorithms for one instance size."""
    instance_size: str
    algorithm1_name: str
    algorithm2_name: str
    algorithm1_mean_obj: float
    algorithm1_std_obj: float
    algorithm2_mean_obj: float
    algorithm2_std_obj: float
    mean_percent_diff: float  # (alg1 - alg2) / alg1 * 100
    std_percent_diff: float
    algorithm1_wins: int
    algorithm2_wins: int
    ties: int
    n_instances: int
    # Statistical test results
    wilcoxon_statistic: float | None = None
    wilcoxon_p_value: float | None = None
    is_significant: bool = False  # p < 0.05
    # Raw percentage differences for boxplot
    percent_diffs_array: np.ndarray | None = None
    # Construction time statistics
    algorithm1_mean_time: float | None = None  # seconds
    algorithm2_mean_time: float | None = None  # seconds


@dataclass
class ExperimentSummary:
    """Complete summary across all instance sizes."""
    comparison_results: list[ComparisonResults]
    algorithm1_total_wins: int
    algorithm2_total_wins: int
    total_ties: int
    total_instances: int


def run_algorithm_on_instance(
    instance: SCFPDPInstance,
    algorithm_config: AlgorithmConfig
) -> InstanceResult:
    """
    Run a single algorithm on a single instance and collect results.

    Args:
        instance: SCFPDPInstance
        algorithm_config: Configuration for the algorithm to run

    Returns:
        InstanceResult with timing and objective information
    """
    start_time = time.time()

    # Load instance
    solution = SCFPDPSolution(instance)
    parsing_time = time.time() - start_time

    # Run construction heuristic
    construction_start = time.time()
    heuristic = algorithm_config.algorithm_class(solution, **algorithm_config.init_kwargs)
    heuristic.construct()
    construction_time = time.time() - construction_start

    # Calculate objective
    objective = solution.calc_objective()
    total_time = time.time() - start_time

    return InstanceResult(
        instance_name=str(instance),
        instance_size=instance.n,
        algorithm_name=algorithm_config.name,
        objective_value=objective,
        construction_time=construction_time,
        total_time=total_time,
        parsing_time=parsing_time
    )


def process_single_instance(instance_file, algorithm1_config, algorithm2_config):
    instance = SCFPDPInstance(str(instance_file))
    alg1_result = run_algorithm_on_instance(instance, algorithm1_config)
    alg2_result = run_algorithm_on_instance(instance, algorithm2_config)
    return alg1_result, alg2_result


def compare_on_single_size(
    instance_size: str,
    instance_type: InstanceType,
    algorithm1_config: AlgorithmConfig,
    algorithm2_config: AlgorithmConfig,
) -> ComparisonResults:
    """
    Compare two algorithms on all instances of a given size.

    Args:
        instance_size: Size of instances to test (e.g., "100", "500")
        instance_type: Type of instances (TEST, TRAIN, COMPETITION)
        algorithm1_config: Configuration for first algorithm
        algorithm2_config: Configuration for second algorithm

    Returns:
        ComparisonResults with statistics and win counts
    """
    # Load all instance files
    project_root = find_project_root()
    instance_dir = project_root / "scfpdp_instances" / instance_size / instance_type
    instance_files = sorted(instance_dir.glob("*.txt"))

    if not instance_files:
        raise ValueError(f"No instances found in {instance_dir}")

    # Run both algorithms on each instance in parallel
    # Limit workers for large instances to avoid memory issues
    instance_size_int = int(instance_size)
    if instance_size_int >= 5000:
        n_workers = min(2, len(instance_files))  # Max 2 workers for large instances
    else:
        n_workers = min(cpu_count(), len(instance_files))

    process_func = partial(process_single_instance,
                          algorithm1_config=algorithm1_config,
                          algorithm2_config=algorithm2_config)

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, instance_files, chunksize=1),
            total=len(instance_files),
            desc=f"Size {instance_size}"
        ))

    alg1_results = [r[0] for r in results]
    alg2_results = [r[1] for r in results]

    # Clean up memory
    del results
    gc.collect()

    # Extract objective values
    alg1_objs = np.array([r.objective_value for r in alg1_results])
    alg2_objs = np.array([r.objective_value for r in alg2_results])

    # Calculate statistics
    alg1_mean = float(np.mean(alg1_objs))
    alg1_std = float(np.std(alg1_objs, ddof=1))
    alg2_mean = float(np.mean(alg2_objs))
    alg2_std = float(np.std(alg2_objs, ddof=1))

    # Extract construction times
    alg1_times = np.array([r.construction_time for r in alg1_results])
    alg2_times = np.array([r.construction_time for r in alg2_results])

    # Calculate mean construction times
    alg1_mean_time = float(np.mean(alg1_times))
    alg2_mean_time = float(np.mean(alg2_times))

    # Calculate pairwise percentage differences
    percent_diffs = (alg1_objs - alg2_objs) / alg1_objs * 100
    mean_percent_diff = float(np.mean(percent_diffs))
    std_percent_diff = float(np.std(percent_diffs, ddof=1))

    # Count wins (lower objective is better for minimization)
    alg1_wins = int(np.sum(alg1_objs < alg2_objs))
    alg2_wins = int(np.sum(alg2_objs < alg1_objs))
    ties = int(np.sum(alg1_objs == alg2_objs))

    # Perform Wilcoxon signed-rank test (non-parametric paired test)
    # Test on raw objective differences (not percentage) for statistical validity
    wilcoxon_stat = None
    wilcoxon_p = None
    is_significant = False
    try:
        # Wilcoxon requires non-zero differences
        diffs = alg1_objs - alg2_objs
        if not np.all(diffs == 0):
            stat, p_value = wilcoxon(alg1_objs, alg2_objs, alternative='two-sided')
            wilcoxon_stat = float(stat)
            wilcoxon_p = float(p_value)
            is_significant = p_value < 0.05
    except ValueError as e:
        # Wilcoxon can fail with too few samples or all-zero differences
        print(f"Wilcoxon failed: {e}")
        pass

    sig_marker = "*" if is_significant else ""
    p_str = f"p={wilcoxon_p:.4f}{sig_marker}" if wilcoxon_p is not None else "p=N/A"
    print(f"  {algorithm1_config.name} wins: {alg1_wins}, {algorithm2_config.name} wins: {alg2_wins}, Ties: {ties} | {p_str}")

    return ComparisonResults(
        instance_size=instance_size,
        algorithm1_name=algorithm1_config.name,
        algorithm2_name=algorithm2_config.name,
        algorithm1_mean_obj=alg1_mean,
        algorithm1_std_obj=alg1_std,
        algorithm2_mean_obj=alg2_mean,
        algorithm2_std_obj=alg2_std,
        mean_percent_diff=mean_percent_diff,
        std_percent_diff=std_percent_diff,
        algorithm1_wins=alg1_wins,
        algorithm2_wins=alg2_wins,
        ties=ties,
        n_instances=len(instance_files),
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p_value=wilcoxon_p,
        is_significant=is_significant,
        percent_diffs_array=percent_diffs,
        algorithm1_mean_time=alg1_mean_time,
        algorithm2_mean_time=alg2_mean_time,
    )


def compare_algorithms_across_sizes(
    instance_sizes: list[str],
    instance_type: InstanceType,
    algorithm1_config: AlgorithmConfig,
    algorithm2_config: AlgorithmConfig,
) -> ExperimentSummary:
    """
    Main entry point: compare two algorithms across multiple instance sizes.

    Args:
        instance_sizes: List of instance sizes to test (e.g., ["100", "500", "1000"])
        instance_type: Type of instances (TEST, TRAIN, COMPETITION)
        algorithm1_config: Configuration for first algorithm
        algorithm2_config: Configuration for second algorithm

    Returns:
        ExperimentSummary with complete comparison results
    """
    print("="*80)
    print(f"  CONSTRUCTION HEURISTICS COMPARISON: {algorithm1_config.name} vs {algorithm2_config.name}")
    print("="*80)
    print(f"Instance Type: {instance_type.value}")
    print(f"Instance Sizes: {', '.join(instance_sizes)}")

    # Compare on each instance size
    comparison_results = []
    for instance_size in instance_sizes:
        comp = compare_on_single_size(
            instance_size=instance_size,
            instance_type=instance_type,
            algorithm1_config=algorithm1_config,
            algorithm2_config=algorithm2_config,
        )
        comparison_results.append(comp)

    # Aggregate total wins
    algorithm1_total_wins = sum(comp.algorithm1_wins for comp in comparison_results)
    algorithm2_total_wins = sum(comp.algorithm2_wins for comp in comparison_results)
    total_ties = sum(comp.ties for comp in comparison_results)
    total_instances = sum(comp.n_instances for comp in comparison_results)

    summary = ExperimentSummary(
        comparison_results=comparison_results,
        algorithm1_total_wins=algorithm1_total_wins,
        algorithm2_total_wins=algorithm2_total_wins,
        total_ties=total_ties,
        total_instances=total_instances
    )

    # Print summary table
    print_comparison_table(summary)

    return summary


def print_comparison_table(summary: ExperimentSummary) -> None:
    """
    Print formatted comparison table to console using pandas.

    Args:
        summary: ExperimentSummary containing all comparison results
    """
    # Build table data
    data = []
    for comp in summary.comparison_results:
        # Format p-value with significance marker
        if comp.wilcoxon_p_value is not None:
            p_str = f"{comp.wilcoxon_p_value:.4f}"
        else:
            p_str = "N/A"

        row = {
            'Size': comp.instance_size,
            f'{comp.algorithm1_name} Obj': f"{comp.algorithm1_mean_obj:.1f} ± {comp.algorithm1_std_obj:.1f}",
            f'{comp.algorithm2_name} Obj': f"{comp.algorithm2_mean_obj:.1f} ± {comp.algorithm2_std_obj:.1f}",
            '% Diff': f"{comp.mean_percent_diff:+.2f} ± {comp.std_percent_diff:.2f}",
            'Wins (1/2/T)': f"{comp.algorithm1_wins}/{comp.algorithm2_wins}/{comp.ties}",
            'p-value': p_str,
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate sum of objectives across all instances
    alg1_sum_obj = sum(comp.algorithm1_mean_obj * comp.n_instances for comp in summary.comparison_results)
    alg2_sum_obj = sum(comp.algorithm2_mean_obj * comp.n_instances for comp in summary.comparison_results)

    # Add total row
    total_row = {
        'Size': 'TOTAL',
        f'{summary.comparison_results[0].algorithm1_name} Obj': f'{alg1_sum_obj:.1f}',
        f'{summary.comparison_results[0].algorithm2_name} Obj': f'{alg2_sum_obj:.1f}',
        '% Diff': '',
        'Wins (1/2/T)': f"{summary.algorithm1_total_wins}/{summary.algorithm2_total_wins}/{summary.total_ties}",
        'p-value': '',
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # Print table
    print("\n" + "="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Print legend
    print("\nLegend:")
    print(f"  - Obj: Mean objective value ± standard deviation")
    print(f"  - % Diff: ({summary.comparison_results[0].algorithm1_name} - {summary.comparison_results[0].algorithm2_name}) / {summary.comparison_results[0].algorithm1_name} * 100")
    print(f"    For minimization: (negative = {summary.comparison_results[0].algorithm1_name} better, positive = {summary.comparison_results[0].algorithm2_name} better)")
    print(f"  - Wins: ({summary.comparison_results[0].algorithm1_name} wins / {summary.comparison_results[0].algorithm2_name} wins / Ties)")
    print(f"  - p-value: Wilcoxon signed-rank test (p<0.05)")
    print(f"  - Total instances tested: {summary.total_instances}")
    print("="*100 + "\n")


def plot_distribution_histograms(summary: ExperimentSummary, save_plot: bool = False) -> None:
    """
    Plot histograms of objective differences to assess normality.

    This helps justify using Wilcoxon (non-parametric) vs t-test (parametric).
    If distributions are not normal, Wilcoxon is the appropriate test.

    Args:
        summary: ExperimentSummary containing all comparison results
        save_plot: Whether to save the plot to file
    """
    if not summary.comparison_results:
        print("No results to plot")
        return

    n_sizes = len(summary.comparison_results)
    alg1_name = summary.comparison_results[0].algorithm1_name
    alg2_name = summary.comparison_results[0].algorithm2_name

    # Create tiled grid layout (better for 5+ subplots)
    # Calculate grid dimensions: aim for roughly square layout
    n_cols = 2 if n_sizes > 4 else min(n_sizes, 2)
    n_rows = (n_sizes + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Flatten axes array for easier iteration
    if n_sizes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'Distribution of Objective Differences: {alg1_name} - {alg2_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    for ax, comp in zip(axes, summary.comparison_results):
        if comp.percent_diffs_array is None:
            continue

        # Get difference data (use percentage differences for interpretability)
        diffs = comp.percent_diffs_array

        # Plot histogram
        ax.hist(diffs, bins=min(20, len(diffs) // 2), density=True, alpha=0.6,
                color='skyblue', edgecolor='black', label='Data')

        # Fit and plot normal distribution
        mu, std = norm.fit(diffs)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 201)
        y = norm.pdf(x, mu, std)
        ax.plot(x, y, 'r-', linewidth=2, label=f'Normal fit\n(μ={mu:.2f}, σ={std:.2f})')

        # Add vertical line at mean
        ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Mean: {mu:.2f}%')

        # Add vertical line at median (robust to outliers)
        median = np.median(diffs)
        ax.axvline(median, color='green', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'Median: {median:.2f}%')

        ax.set_xlabel('% Difference', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'n={comp.instance_size}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots if grid has more spaces than plots
    for idx in range(n_sizes, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_sizes_str = '-'.join([comp.instance_size for comp in summary.comparison_results])
        filename = plots_dir / f"{alg1_name}_vs_{alg2_name}_distributions_{instance_sizes_str}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved as '{filename}'")
        plt.close()
    else:
        plt.show()


def plot_comparison_results(summary: ExperimentSummary, save_plot: bool = False) -> None:
    """
    Plot comparison results showing percentage differences, win counts, and boxplots.

    Args:
        summary: ExperimentSummary containing all comparison results
        save_plot: Whether to save the plot to file
    """
    if not summary.comparison_results:
        print("No results to plot")
        return

    # Extract data
    instance_sizes = [comp.instance_size for comp in summary.comparison_results]
    alg1_wins = [comp.algorithm1_wins for comp in summary.comparison_results]
    alg2_wins = [comp.algorithm2_wins for comp in summary.comparison_results]
    ties = [comp.ties for comp in summary.comparison_results]

    alg1_name = summary.comparison_results[0].algorithm1_name
    alg2_name = summary.comparison_results[0].algorithm2_name

    # Calculate sum of objectives across all instances
    alg1_sum_obj = sum(comp.algorithm1_mean_obj * comp.n_instances for comp in summary.comparison_results)
    alg2_sum_obj = sum(comp.algorithm2_mean_obj * comp.n_instances for comp in summary.comparison_results)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Algorithm Comparison: {alg1_name} vs {alg2_name}', fontsize=14, fontweight='bold')

    # Plot 1: Boxplot of Percentage Differences
    # Collect percentage differences arrays for boxplot
    percent_diffs_arrays = []
    for comp in summary.comparison_results:
        if comp.percent_diffs_array is not None:
            percent_diffs_arrays.append(comp.percent_diffs_array)
        else:
            percent_diffs_arrays.append(np.array([comp.mean_percent_diff]))

    bp = ax1.boxplot(percent_diffs_arrays, labels=instance_sizes, patch_artist=True)

    # Color boxes based on which algorithm wins more (aligned with win bar colors)
    for i, (box, comp) in enumerate(zip(bp['boxes'], summary.comparison_results)):
        if comp.algorithm1_wins > comp.algorithm2_wins:
            box.set_facecolor('lightblue')
        elif comp.algorithm2_wins > comp.algorithm1_wins:
            box.set_facecolor('lightgreen')
        else:
            box.set_facecolor('lightgray')
        box.set_alpha(0.7)

    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')
    ax1.set_xlabel('Instance Size (n)', fontsize=12)
    ax1.set_ylabel('% Difference', fontsize=12)
    ax1.set_title('Distribution of % Differences in Avg Objective Value', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Auto-scale y-axis to wrap around data with padding
    all_values = np.concatenate(percent_diffs_arrays)
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    padding = y_range * 0.15 if y_range > 0 else 1
    ax1.set_ylim(y_min - padding, y_max + padding)

    # Add legend for boxplot colors
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.7, label=f'{alg1_name} stronger'),
        Patch(facecolor='lightgreen', alpha=0.7, label=f'{alg2_name} stronger'),
        Patch(facecolor='lightgray', alpha=0.7, label='Tied')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=9)

    # Plot 2: Win Distribution
    x = np.arange(len(instance_sizes))
    width = 0.25

    ax2.bar(x - width, alg1_wins, width, label=f'{alg1_name} wins', color='lightblue', alpha=0.8)
    ax2.bar(x, alg2_wins, width, label=f'{alg2_name} wins', color='lightgreen', alpha=0.8)
    ax2.bar(x + width, ties, width, label='Ties', color='lightgray', alpha=0.8)

    ax2.set_xlabel('Instance Size (n)', fontsize=12)
    ax2.set_ylabel('Number of Wins', fontsize=12)
    ax2.set_title('Win Distribution by Instance Size', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(instance_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Enhanced summary statistics text box with p-values per size
    total_obj_percent_diff = (alg1_sum_obj - alg2_sum_obj) / alg1_sum_obj * 100

    if total_obj_percent_diff > 0:
        comparison_text = f"{alg2_name}'s total objective is better than {alg1_name}'s by {total_obj_percent_diff:.4f}% on average"
    elif total_obj_percent_diff < 0:
        comparison_text = f"{alg1_name}'s total objective is better than {alg2_name}'s by {abs(total_obj_percent_diff):.4f}% on average"
    else:
        comparison_text = f"{alg1_name}'s and {alg2_name}'s total objectives are equal"

    # Build p-value string per size
    p_value_lines = []
    for comp in summary.comparison_results:
        if comp.wilcoxon_p_value is not None:
            p_str = f"{comp.wilcoxon_p_value:.6f}" if comp.wilcoxon_p_value >= 0.0001 else f"{comp.wilcoxon_p_value:.2e}"
        else:
            p_str = "N/A"
        p_value_lines.append(f"n={comp.instance_size}: p={p_str}")

    p_values_by_size = " | ".join(p_value_lines)

    summary_text = (
        f'OVERALL: {summary.total_instances} instances | '
        f'Total Wins: {alg1_name}={summary.algorithm1_total_wins}, {alg2_name}={summary.algorithm2_total_wins}, Ties={summary.total_ties} \n'
        f'{comparison_text}\n'
        f'P-VALUES (Wilcoxon): {p_values_by_size}'
    )
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
             verticalalignment='bottom', multialignment='left')

    plt.tight_layout(rect=(0, 0.10, 1, 1))

    if save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = plots_dir / f"{alg1_name}_vs_{alg2_name}_{'-'.join(instance_sizes)}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    all_instance_sizes = ["50", "100", "200", "500", "1000", "2000", "5000", "10000"]
    instances_to_run = all_instance_sizes[:4]

    # summary = compare_greedy_vs_randomized(
    #     instance_sizes=instances_to_run,
    #     instance_type=InstanceType.TEST,
    #     randomized_top_k=10
    # )

    # Plot comparison results
    # plot_comparison_results(summary, save_plot=True)