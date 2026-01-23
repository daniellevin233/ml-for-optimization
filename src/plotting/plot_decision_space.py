"""
Visualization: Decision Space for Algorithm Selection.

2D scatter plots showing which algorithm is best for different problem instances.
Helps visualize the separation between SAT-best and SA-best instances.
"""
import pandas as pd
from matplotlib import pyplot as plt

from src.utils import find_project_root


def load_combined_dataset() -> pd.DataFrame:
    """Load the combined dataset with both SAT and SA results."""
    path = find_project_root() / "data" / "combined_dataset.csv"
    return pd.read_csv(path)


def plot_decision_space(df: pd.DataFrame, save: bool = True):
    """
    Plot 2D scatter showing algorithm selection in problem space.

    Creates two subplots:
    1. problem_size vs w (weeks)
    2. total_golfers vs w (weeks)

    Args:
        df: Combined dataset
        save: Whether to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Algorithm Selection Decision Space", fontsize=16, fontweight="bold")

    # Filter instances with valid best_algo
    df_valid = df[df['best_algo'].notna()].copy()

    # Split by best algorithm
    sat_best = df_valid[df_valid['best_algo'] == 'sat']
    sa_best = df_valid[df_valid['best_algo'] == 'sa']

    # Plot 1: problem_size vs w
    ax1.scatter(
        sat_best['problem_size'],
        sat_best['w'],
        c='steelblue',
        s=80,
        alpha=0.6,
        edgecolors='darkblue',
        linewidths=1,
        label=f'SAT Best ({len(sat_best)})',
        marker='o',
    )
    ax1.scatter(
        sa_best['problem_size'],
        sa_best['w'],
        c='coral',
        s=80,
        alpha=0.6,
        edgecolors='darkred',
        linewidths=1,
        label=f'SA Best ({len(sa_best)})',
        marker='s',
    )

    ax1.set_xlabel("Problem Size (m × n × w)", fontsize=12)
    ax1.set_ylabel("Weeks (w)", fontsize=12)
    ax1.set_title("Problem Size vs Weeks", fontsize=13, fontweight="bold")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: total_golfers vs w
    ax2.scatter(
        sat_best['total_golfers'],
        sat_best['w'],
        c='steelblue',
        s=80,
        alpha=0.6,
        edgecolors='darkblue',
        linewidths=1,
        label=f'SAT Best ({len(sat_best)})',
        marker='o',
    )
    ax2.scatter(
        sa_best['total_golfers'],
        sa_best['w'],
        c='coral',
        s=80,
        alpha=0.6,
        edgecolors='darkred',
        linewidths=1,
        label=f'SA Best ({len(sa_best)})',
        marker='s',
    )

    ax2.set_xlabel("Total Golfers (m × n)", fontsize=12)
    ax2.set_ylabel("Weeks (w)", fontsize=12)
    ax2.set_title("Total Golfers vs Weeks", fontsize=13, fontweight="bold")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / "decision_space.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to {filename}")

    plt.show()


if __name__ == "__main__":
    df = load_combined_dataset()
    print(f"Loaded {len(df)} instances")
    print(f"SAT best: {len(df[df['best_algo'] == 'sat'])}")
    print(f"SA best: {len(df[df['best_algo'] == 'sa'])}")
    print()

    plot_decision_space(df, save=True)
