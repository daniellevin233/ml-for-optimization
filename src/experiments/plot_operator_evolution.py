"""
Operator selection evolution plotting for ALNS experiments.

Visualizes how operator selection changes over time (RQ4).
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.alns.alns import ALNSStatistics


def compute_operator_frequencies(
    statistics: list[ALNSStatistics],
    operator_type: str = "destroy",
    n_segments: int = 5
) -> tuple[list[str], np.ndarray]:
    """
    Compute operator selection frequencies over time segments.

    Args:
        statistics: List of ALNSStatistics from ALNS run
        operator_type: "destroy" or "repair"
        n_segments: Number of time segments to divide iterations into

    Returns:
        Tuple of (operator_names, frequency_matrix) where frequency_matrix
        is shape (n_operators, n_segments) with selection frequencies
    """
    total_iterations = len(statistics)
    segment_size = total_iterations // n_segments

    # Get operator names from first statistic
    if operator_type == "destroy":
        operator_field = "destroy_operator_used"
    else:
        operator_field = "repair_operator_used"

    # Count selections per segment
    operator_counts = defaultdict(lambda: [0] * n_segments)

    for i, stat in enumerate(statistics):
        segment_idx = min(i // segment_size, n_segments - 1)
        operator_name = getattr(stat, operator_field)
        operator_counts[operator_name][segment_idx] += 1

    # Convert to frequency matrix
    operator_names = sorted(operator_counts.keys())
    frequency_matrix = np.zeros((len(operator_names), n_segments))

    for seg_idx in range(n_segments):
        segment_total = sum(counts[seg_idx] for counts in operator_counts.values())
        if segment_total > 0:
            for op_idx, op_name in enumerate(operator_names):
                frequency_matrix[op_idx, seg_idx] = operator_counts[op_name][seg_idx] / segment_total * 100

    return operator_names, frequency_matrix


def plot_operator_evolution(
    statistics: list[ALNSStatistics],
    operator_type: str = "destroy",
    title: str = None,
    save_path: Path = None,
    n_segments: int = 5
):
    """
    Plot operator selection evolution as stacked bar chart.

    Args:
        statistics: List of ALNSStatistics from ALNS run
        operator_type: "destroy" or "repair"
        title: Plot title
        save_path: Optional path to save plot
        n_segments: Number of time segments
    """
    operator_names, frequency_matrix = compute_operator_frequencies(
        statistics, operator_type, n_segments
    )

    # Create segment labels
    total_iterations = len(statistics)
    segment_size = total_iterations // n_segments
    segment_labels = [f"{i*segment_size}-{min((i+1)*segment_size, total_iterations)}"
                     for i in range(n_segments)]

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_segments)
    bottom = np.zeros(n_segments)

    colors = plt.cm.Set3(np.linspace(0, 1, len(operator_names)))

    for op_idx, op_name in enumerate(operator_names):
        ax.bar(x, frequency_matrix[op_idx], bottom=bottom, label=op_name,
               color=colors[op_idx], width=0.8, alpha=0.9)
        bottom += frequency_matrix[op_idx]

    ax.set_xlabel('Iteration Range', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title(title or f'{operator_type.capitalize()} Operator Selection Evolution',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels, rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Operator evolution plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def compute_operator_rewards(
    statistics: list[ALNSStatistics],
    operator_type: str = "destroy"
) -> dict[str, tuple[float, int]]:
    """
    Compute average reward per operator.

    Args:
        statistics: List of ALNSStatistics from ALNS run
        operator_type: "destroy" or "repair"

    Returns:
        Dict mapping operator name to (avg_reward, selection_count)
    """
    if operator_type == "destroy":
        operator_field = "destroy_operator_used"
    else:
        operator_field = "repair_operator_used"

    operator_rewards = defaultdict(lambda: [0.0, 0])  # [total_reward, count]

    for i in range(1, len(statistics)):
        curr_stat = statistics[i]
        prev_stat = statistics[i-1]

        operator_name = getattr(curr_stat, operator_field)

        # Calculate reward based on improvement
        if curr_stat.best_objective < prev_stat.best_objective:
            reward = 10.0  # New best
        elif curr_stat.current_objective < prev_stat.current_objective:
            reward = 1.0   # Improvement
        else:
            reward = 0.0   # No improvement

        operator_rewards[operator_name][0] += reward
        operator_rewards[operator_name][1] += 1

    # Compute averages
    result = {}
    for op_name, (total_reward, count) in operator_rewards.items():
        avg_reward = total_reward / count if count > 0 else 0.0
        result[op_name] = (avg_reward, count)

    return result


def plot_operator_performance(
    statistics_dict: dict[str, list[ALNSStatistics]],
    operator_type: str = "destroy",
    save_path: Path = None
):
    """
    Plot average reward per operator across multiple algorithms.

    Args:
        statistics_dict: Dict mapping algorithm name to ALNSStatistics list
        operator_type: "destroy" or "repair"
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    all_operator_names = set()
    for stats_list in statistics_dict.values():
        rewards = compute_operator_rewards(stats_list, operator_type)
        all_operator_names.update(rewards.keys())

    operator_names = sorted(all_operator_names)
    n_operators = len(operator_names)
    n_algorithms = len(statistics_dict)

    x = np.arange(n_operators)
    width = 0.8 / n_algorithms

    for alg_idx, (alg_name, stats_list) in enumerate(statistics_dict.items()):
        rewards = compute_operator_rewards(stats_list, operator_type)
        avg_rewards = [rewards.get(op, (0, 0))[0] for op in operator_names]

        ax.bar(x + alg_idx * width, avg_rewards, width, label=alg_name, alpha=0.8)

    ax.set_xlabel('Operator', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(f'{operator_type.capitalize()} Operator Performance',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_algorithms - 1) / 2)
    ax.set_xticklabels(operator_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Operator performance plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    results_dir = Path("src/experiments/results/")
    save_dir = Path("plots/")

    if results_dir.exists():
        # Load statistics for one instance
        instance_name = "instance61_nreq100_nveh2_gamma91"
        algorithm = "Adaptive"

        stats_file = results_dir / f"{instance_name}_{algorithm}.pkl"
        if not stats_file.exists():
            print(f"Statistics file {stats_file} not found")
            exit(0)

        with open(stats_file, 'rb') as f:
            statistics = pickle.load(f)

        # Plot destroy operator evolution
        plot_operator_evolution(
            statistics,
            operator_type="destroy",
            title=f"Destroy Operator Evolution - {algorithm}",
            save_path=save_dir / f"{instance_name}_{algorithm}_destroy_evolution.png",
            n_segments=5
        )

        # Plot repair operator evolution
        plot_operator_evolution(
            statistics,
            operator_type="repair",
            title=f"Repair Operator Evolution - {algorithm}",
            save_path=save_dir / f"{instance_name}_{algorithm}_repair_evolution.png",
            n_segments=5
        )
    else:
        print(f"Results directory {results_dir} not found")
        print("Run experiments first to generate statistics")