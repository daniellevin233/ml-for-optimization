import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.alns.alns import ALNSStatistics


def plot_convergence(
    statistics_dict: dict[str, list[ALNSStatistics]],
    title: str = "ALNS Convergence Comparison",
    save_path: Path = None
):
    """
    Plot convergence curves for multiple ALNS runs.

    Args:
        statistics_dict: Dict mapping algorithm name to list of ALNSStatistics
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Adaptive': 'blue', 'Ucb1': 'orange', 'Thompson': 'green'}

    for alg_name, stats_list in statistics_dict.items():
        iterations = [s.iteration for s in stats_list]
        best_objectives = [s.best_objective for s in stats_list]

        color = colors.get(alg_name, 'gray')
        ax.plot(iterations, best_objectives, label=alg_name, color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Objective', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_convergence_multiple_instances(
    results_dir: Path,
    instance_names: list[str],
    algorithms: list[str] = ["Adaptive", "Ucb1", "Thompson"],
    save_dir: Path = None
):
    """
    Plot convergence for multiple instances, one plot per instance.

    Args:
        results_dir: Directory containing pickled ALNS statistics
        instance_names: List of instance filenames
        algorithms: List of algorithm names to plot
        save_dir: Directory to save plots
    """
    for instance_name in instance_names:
        statistics_dict = {}

        for alg in algorithms:
            stats_file = results_dir / f"{instance_name}_{alg}.pkl"
            if stats_file.exists():
                with open(stats_file, 'rb') as f:
                    statistics_dict[alg] = pickle.load(f)

        if statistics_dict:
            title = f"Convergence: {instance_name}"
            save_path = save_dir / f"convergence_{instance_name}.png" if save_dir else None
            plot_convergence(statistics_dict, title=title, save_path=save_path)


if __name__ == "__main__":
    """
    Example usage:
        python -m src.experiments.plot_convergence
    """
    # Example: Load and plot convergence
    results_dir = Path("results/mab_experiment")
    save_dir = Path("plots/convergence")

    if results_dir.exists():
        instance_names = ["test_instance_n50_01"]  # User will specify
        plot_convergence_multiple_instances(
            results_dir=results_dir,
            instance_names=instance_names,
            save_dir=save_dir
        )
    else:
        print(f"Results directory {results_dir} not found")
        print("Run experiments first to generate statistics")