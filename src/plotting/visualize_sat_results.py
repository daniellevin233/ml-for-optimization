"""Visualization of SAT solver results on Social Golfer Problem instances."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils import find_project_root

TIMEOUT_SECONDS = 60.0


def load_dataset() -> pd.DataFrame:
    """Load the classification dataset."""
    path = find_project_root() / "data" / "sat_dataset.csv"
    return pd.read_csv(path)


def plot_boxplots_by_parameter(df: pd.DataFrame, save: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SAT Runtime Distribution by Parameter", fontsize=14, fontweight="bold")

    df_plot = df.copy()
    df_plot["is_timeout"] = df_plot["sat_runtime"].isna()
    df_plot["sat_runtime_plot"] = df_plot["sat_runtime"].fillna(TIMEOUT_SECONDS)

    params = [("m", "Groups (m)"), ("n", "Golfers per Group (n)"), ("w", "Weeks (w)")]

    for ax, (param, label) in zip(axes, params):
        unique_vals = sorted(df_plot[param].unique())

        solved_data = []
        positions = []
        for val in unique_vals:
            subset = df_plot[(df_plot[param] == val) & (~df_plot["is_timeout"])]
            if not subset.empty:
                solved_data.append(subset["sat_runtime"].values)
                positions.append(val)

        if solved_data:
            bp = ax.boxplot(solved_data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)

        for val in unique_vals:
            timeout_subset = df_plot[(df_plot[param] == val) & (df_plot["is_timeout"])]
            if not timeout_subset.empty:
                jitter = np.random.uniform(-0.15, 0.15, size=len(timeout_subset))
                ax.scatter(
                    [val] * len(timeout_subset) + jitter,
                    [TIMEOUT_SECONDS] * len(timeout_subset),
                    c="red",
                    marker="X",
                    s=30,
                    alpha=0.7,
                    zorder=5,
                )

        for val in unique_vals:
            solved_subset = df_plot[(df_plot[param] == val) & (~df_plot["is_timeout"])]
            if not solved_subset.empty:
                jitter = np.random.uniform(-0.15, 0.15, size=len(solved_subset))
                ax.scatter(
                    [val] * len(solved_subset) + jitter,
                    solved_subset["sat_runtime"].values,
                    c="blue",
                    marker="o",
                    s=30,
                    alpha=0.5,
                    zorder=4,
                )

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Runtime (seconds)" if ax == axes[0] else "", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlim(min(unique_vals) - 0.5, max(unique_vals) + 0.5)
        ax.set_xticks(unique_vals)
        ax.set_xticklabels([str(v) for v in unique_vals])
        ax.grid(True, alpha=0.3, axis="y")

        # Add timeout line
        ax.axhline(y=TIMEOUT_SECONDS, color="red", linestyle="--", alpha=0.5, linewidth=1)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, alpha=0.7, label="Solved"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red", markersize=10, label="Timeout (60s)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "sat_boxplots_by_param.png", dpi=300, bbox_inches="tight")
        print(f"Saved to {plots_dir / 'sat_boxplots_by_param.png'}")

    plt.show()


def plot_all(save: bool = False):
    """Generate all SAT result visualizations."""
    df = load_dataset()
    print(f"Loaded {len(df)} instances")
    print(f"  Solved: {df['sat_runtime'].notna().sum()}")
    print(f"  Timeout: {df['sat_runtime'].isna().sum()}")
    print()

    plot_boxplots_by_parameter(df, save=save)


if __name__ == "__main__":
    plot_all(save=True)