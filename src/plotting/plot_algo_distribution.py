"""
Visualization: Algorithm Selection Distribution by Parameter.

Box plots comparing SAT vs SA runtime performance grouped by m, n, w parameters.
Shows which algorithm performs better for different parameter values.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils import find_project_root

TIMEOUT_SECONDS = 60.0


def load_combined_dataset() -> pd.DataFrame:
    """Load the combined dataset with both SAT and SA results."""
    path = find_project_root() / "data" / "combined_dataset.csv"
    return pd.read_csv(path)


def plot_algo_distribution_by_parameter(df: pd.DataFrame, save: bool = True):
    """
    Plot side-by-side box plots comparing SAT vs SA runtime by parameter.

    Args:
        df: Combined dataset
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Algorithm Performance Distribution by Parameter", fontsize=16, fontweight="bold")

    params = [("m", "Groups (m)"), ("n", "Golfers per Group (n)"), ("w", "Weeks (w)")]

    for ax, (param, label) in zip(axes, params):
        unique_vals = sorted(df[param].dropna().unique())

        # Prepare data for each parameter value
        sat_data = []
        sa_data = []
        positions_sat = []
        positions_sa = []

        for val in unique_vals:
            subset = df[df[param] == val]

            # SAT runtimes (exclude timeouts)
            sat_runtimes = subset[subset['sat_runtime'].notna()]['sat_runtime'].values
            if len(sat_runtimes) > 0:
                sat_data.append(sat_runtimes)
                positions_sat.append(val - 0.15)  # Offset left

            # SA runtimes (exclude timeouts)
            sa_runtimes = subset[subset['sa_runtime'].notna()]['sa_runtime'].values
            if len(sa_runtimes) > 0:
                sa_data.append(sa_runtimes)
                positions_sa.append(val + 0.15)  # Offset right

        # Plot SAT box plots
        if sat_data:
            bp_sat = ax.boxplot(
                sat_data,
                positions=positions_sat,
                widths=0.25,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp_sat["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.7)
            for whisker in bp_sat["whiskers"]:
                whisker.set_color("steelblue")
            for cap in bp_sat["caps"]:
                cap.set_color("steelblue")
            for median in bp_sat["medians"]:
                median.set_color("darkblue")
                median.set_linewidth(2)

        # Plot SA box plots
        if sa_data:
            bp_sa = ax.boxplot(
                sa_data,
                positions=positions_sa,
                widths=0.25,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp_sa["boxes"]:
                patch.set_facecolor("coral")
                patch.set_alpha(0.7)
            for whisker in bp_sa["whiskers"]:
                whisker.set_color("coral")
            for cap in bp_sa["caps"]:
                cap.set_color("coral")
            for median in bp_sa["medians"]:
                median.set_color("darkred")
                median.set_linewidth(2)

        # Overlay scatter points colored by best_algo
        for val in unique_vals:
            subset = df[df[param] == val]

            # SAT points
            sat_subset = subset[subset['sat_runtime'].notna()]
            for _, row in sat_subset.iterrows():
                color = 'blue' if row['best_algo'] == 'sat' else 'lightblue'
                jitter = np.random.uniform(-0.08, 0.08)
                ax.scatter(
                    val - 0.15 + jitter,
                    row['sat_runtime'],
                    c=color,
                    s=20,
                    alpha=0.6,
                    edgecolors='none',
                    zorder=3,
                )

            # SA points
            sa_subset = subset[subset['sa_runtime'].notna()]
            for _, row in sa_subset.iterrows():
                color = 'red' if row['best_algo'] == 'sa' else 'lightcoral'
                jitter = np.random.uniform(-0.08, 0.08)
                ax.scatter(
                    val + 0.15 + jitter,
                    row['sa_runtime'],
                    c=color,
                    s=20,
                    alpha=0.6,
                    edgecolors='none',
                    zorder=3,
                )

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Runtime (seconds)" if ax == axes[0] else "", fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(min(unique_vals) - 0.5, max(unique_vals) + 0.5)
        ax.set_xticks(unique_vals)
        ax.set_xticklabels([str(int(v)) for v in unique_vals])
        ax.grid(True, alpha=0.3, axis="y")

        # Timeout reference line
        ax.axhline(y=TIMEOUT_SECONDS, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
               markersize=10, alpha=0.7, label='SAT'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='coral',
               markersize=10, alpha=0.7, label='SA'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Best: SAT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Best: SA'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.96))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / "algo_distribution_by_param.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to {filename}")

    plt.show()


if __name__ == "__main__":
    df = load_combined_dataset()
    print(f"Loaded {len(df)} instances")
    print(f"SAT best: {len(df[df['best_algo'] == 'sat'])}")
    print(f"SA best: {len(df[df['best_algo'] == 'sa'])}")
    print()

    plot_algo_distribution_by_parameter(df, save=True)