from datetime import datetime
from dataclasses import dataclass

from matplotlib import pyplot as plt

from src.utils import find_project_root


@dataclass
class PlotData:
    """Configuration for a single plot.

    Always use lists for y_values, colors, markers, and labels.
    For single series: y_values=[[1,2,3]], colors=['blue'], labels=['My Line']
    For multiple series: y_values=[[1,2,3], [4,5,6]], colors=['blue', 'red'], labels=['Line 1', 'Line 2']
    """
    x_values: list[float]
    y_values: list[list[float]]  # List of series (each series is a list of values)
    x_label: str
    y_label: str
    title: str
    colors: list[str] = None  # One color per series (defaults to ['blue', 'red', 'green', 'orange', ...])
    alpha: float = 0.7
    plot_type: str = 'bar'  # 'bar' or 'line'
    markers: list[str] = None  # One marker per series (defaults to ['o', 's', '^', 'D', ...])
    linewidth: float = 2
    markersize: float = 8
    bar_width: float = 5
    bar_width_relative: float | None = None  # Relative bar width (0.0-1.0), overrides bar_width if set
    x_ticks: list[int] | None = None  # Optional custom x-tick values
    x_tick_labels: list[str] | None = None  # Optional custom x-tick labels
    x_scale: str = 'linear'  # 'linear' or 'log' for x-axis
    y_scale: str = 'linear'  # 'linear' or 'log' for y-axis
    labels: list[str] | None = None  # Labels for legend (one per series)

    def __post_init__(self):
        """Set default colors, markers, and labels if not provided."""
        n_series = len(self.y_values)

        if self.colors is None:
            self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:n_series]

        if self.markers is None:
            self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p'][:n_series]

        if self.labels is None:
            self.labels = [f'Series {i+1}' for i in range(n_series)]


@dataclass
class ExperimentPlotConfig:
    """Configuration for experiment plots (typically 2 subplots side by side)."""
    algorithm_name: str
    plot_suptitle: str
    plot1: PlotData  # typically objective values
    plot2: PlotData  # typically runtimes
    save_plot: bool = False
    figsize: tuple[float, float] = (14, 5)


def plot_experiment_results(config: ExperimentPlotConfig):
    """
    Unified plotting function for experiment results (2 subplots side by side).

    Args:
        config: ExperimentPlotConfig containing all plot parameters
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize)
    fig.suptitle(config.plot_suptitle, fontsize=14, fontweight='bold')

    # Calculate bar width for plot 1 if relative width is specified
    bar_width1 = config.plot1.bar_width
    if config.plot1.bar_width_relative is not None and len(config.plot1.x_values) > 1:
        x_range = max(config.plot1.x_values) - min(config.plot1.x_values)
        bar_width1 = x_range * config.plot1.bar_width_relative / len(config.plot1.x_values)

    # Plot 1 - iterate over all series
    for i, y_vals in enumerate(config.plot1.y_values):
        if config.plot1.plot_type == 'bar':
            ax1.bar(config.plot1.x_values, y_vals,
                    color=config.plot1.colors[i], alpha=config.plot1.alpha,
                    label=config.plot1.labels[i], width=bar_width1)
        else:  # line
            ax1.plot(config.plot1.x_values, y_vals,
                    marker=config.plot1.markers[i], linewidth=config.plot1.linewidth,
                    markersize=config.plot1.markersize, color=config.plot1.colors[i],
                    label=config.plot1.labels[i], alpha=config.plot1.alpha)

    ax1.set_xlabel(config.plot1.x_label, fontsize=12)
    ax1.set_ylabel(config.plot1.y_label, fontsize=12)
    ax1.set_title(config.plot1.title, fontsize=14, fontweight='bold')
    ax1.set_xscale(config.plot1.x_scale)
    ax1.set_yscale(config.plot1.y_scale)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Set custom x-ticks for plot 1 if provided
    if config.plot1.x_ticks is not None:
        ax1.set_xticks(config.plot1.x_ticks)
    if config.plot1.x_tick_labels is not None:
        ax1.set_xticklabels(config.plot1.x_tick_labels, rotation=45, ha='right')

    # Calculate bar width for plot 2 if relative width is specified
    bar_width2 = config.plot2.bar_width
    if config.plot2.bar_width_relative is not None and len(config.plot2.x_values) > 1:
        x_range = max(config.plot2.x_values) - min(config.plot2.x_values)
        bar_width2 = x_range * config.plot2.bar_width_relative / len(config.plot2.x_values)

    # Plot 2 - iterate over all series
    for i, y_vals in enumerate(config.plot2.y_values):
        if config.plot2.plot_type == 'bar':
            ax2.bar(config.plot2.x_values, y_vals,
                    color=config.plot2.colors[i], alpha=config.plot2.alpha,
                    label=config.plot2.labels[i], width=bar_width2)
        else:  # line
            ax2.plot(config.plot2.x_values, y_vals,
                    marker=config.plot2.markers[i], linewidth=config.plot2.linewidth,
                    markersize=config.plot2.markersize, color=config.plot2.colors[i],
                    label=config.plot2.labels[i], alpha=config.plot2.alpha)

    ax2.set_xlabel(config.plot2.x_label, fontsize=12)
    ax2.set_ylabel(config.plot2.y_label, fontsize=12)
    ax2.set_title(config.plot2.title, fontsize=14, fontweight='bold')
    ax2.set_xscale(config.plot2.x_scale)
    ax2.set_yscale(config.plot2.y_scale)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Set custom x-ticks for plot 2 if provided
    if config.plot2.x_ticks is not None:
        ax2.set_xticks(config.plot2.x_ticks)
    if config.plot2.x_tick_labels is not None:
        ax2.set_xticklabels(config.plot2.x_tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    if config.save_plot:
        project_root = find_project_root()
        plots_dir = project_root / "plots"
        plots_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = plots_dir / f"{config.plot_suptitle.lower().replace(' ', '_')}_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

    plt.show()
