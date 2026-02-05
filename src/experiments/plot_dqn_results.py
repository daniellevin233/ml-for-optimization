import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from src.utils import find_project_root
from src.algorithm_comparison import ComparisonResults, ExperimentSummary, plot_comparison_results


def plot_training_curves():
    """Plot training progress: improvement over episodes and by instance size."""
    project_root = find_project_root()
    data_path = project_root / "trained_models/dqn/dqn_training_data.pkl"

    with open(data_path, "rb") as f:
        episode_data = pickle.load(f)

    episodes = list(range(1, len(episode_data) + 1))
    improvements = [e["improvement"] for e in episode_data]
    sizes = [e["size"] for e in episode_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Improvement over episodes
    ax1.plot(episodes, improvements, 'o-', linewidth=2, markersize=4)
    ax1.axhline(y=np.mean(improvements), color='r', linestyle='--', label=f'Mean: {np.mean(improvements):.1f}%')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Improvement (%)', fontsize=12)
    ax1.set_title('DQN Training: Improvement per Episode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Improvement by instance size (boxplot)
    unique_sizes = sorted(set(sizes))
    size_improvements = {s: [e["improvement"] for e in episode_data if e["size"] == s] for s in unique_sizes}

    positions = range(len(unique_sizes))
    bp = ax2.boxplot([size_improvements[s] for s in unique_sizes], positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'n={s}' for s in unique_sizes])
    ax2.set_xlabel('Instance Size', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('DQN Training: Improvement by Instance Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = project_root / "plots/dqn_training_curves.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path.name}")
    plt.close()


def plot_parameter_evolution(instance_name, algorithm="DQN"):
    """Plot parameter choices overlaid with stagnation, and objective convergence."""
    project_root = find_project_root()
    stats_path = project_root / f"src/experiments/results/dqn_evaluation/{instance_name}_{algorithm}.pkl"

    if not stats_path.exists():
        print(f"  Skipping parameter evolution (stats not found for {instance_name})")
        return

    with open(stats_path, "rb") as f:
        statistics = pickle.load(f)

    severities = []
    temp_mults = []
    current_objs = []
    best_objs = []
    stagnation = []

    iters_no_improve = 0
    prev_best = statistics[0].best_objective if statistics else float('inf')

    for stat in statistics:
        selector_stats = stat.destroy_selector_stats
        if "severity_chosen" in selector_stats:
            severities.append(selector_stats["severity_chosen"])
            temp_mults.append(selector_stats["temp_multiplier_chosen"])
            current_objs.append(stat.current_objective)
            best_objs.append(stat.best_objective)

            # Track stagnation
            if stat.best_objective < prev_best:
                prev_best = stat.best_objective
                iters_no_improve = 0
            else:
                iters_no_improve += 1
            stagnation.append(iters_no_improve)

    if not severities:
        return

    iterations = list(range(len(severities)))

    # Normalize stagnation to [0, 1] for overlay
    max_stag = max(stagnation) if max(stagnation) > 0 else 1
    stag_normalized = [s / max_stag for s in stagnation]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Severity + normalized stagnation
    ax1.plot(iterations, severities, linewidth=1.5, alpha=0.9, color='blue', label='Severity')
    ax1_stag = ax1.twinx()
    ax1_stag.fill_between(iterations, 0, stag_normalized, alpha=0.15, color='red')
    ax1_stag.set_ylabel(f'Stagnation (max={max_stag})', fontsize=10, color='red')
    ax1_stag.set_ylim([0, 1.2])
    ax1_stag.tick_params(axis='y', labelcolor='red')
    ax1.set_ylabel('Severity', fontsize=11, color='blue')
    ax1.set_ylim([0, 0.5])
    ax1.set_title(f'DQN Parameter Choices: {instance_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Temp multiplier + normalized stagnation
    ax2.plot(iterations, temp_mults, linewidth=1.5, alpha=0.9, color='orange', label='Temp Multiplier')
    ax2_stag = ax2.twinx()
    ax2_stag.fill_between(iterations, 0, stag_normalized, alpha=0.15, color='red', label=f'Stagnation')
    ax2_stag.set_ylabel(f'Stagnation (max={max_stag})', fontsize=10, color='red')
    ax2_stag.set_ylim([0, 1.2])
    ax2_stag.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('Temp Multiplier', fontsize=11, color='orange')
    ax2.set_ylim([0.3, 2.2])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Objective convergence
    ax3.plot(iterations, best_objs, linewidth=2, alpha=0.9, color='green', label='Best Objective')
    ax3.set_ylabel('Objective Value', fontsize=11)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_title('Objective Convergence', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = project_root / f"plots/dqn_parameter_evolution_{instance_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path.name}")
    plt.close()


def plot_comparison():
    """Convert results to comparison format, print statistics, and generate plots."""
    project_root = find_project_root()
    summary_path = project_root / "src/experiments/results/dqn_evaluation/summary.pkl"

    with open(summary_path, "rb") as f:
        results = pickle.load(f)

    # Group by size
    sizes = ["50", "100", "200", "500"]
    comparison_results = []

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON: DQN vs Thompson")
    print("="*80)

    for size in sizes:
        size_results = [r for r in results if f"_n{size}_" in r.instance_name or f"nreq{size}" in r.instance_name]

        if not size_results:
            continue

        dqn_objs = np.array([r.final_objective for r in size_results if r.algorithm_name == "DQN"])
        thompson_objs = np.array([r.final_objective for r in size_results if r.algorithm_name == "Thompson"])

        if len(dqn_objs) == 0 or len(thompson_objs) == 0:
            continue

        percent_diffs = (dqn_objs - thompson_objs) / thompson_objs * 100

        dqn_wins = int(np.sum(dqn_objs < thompson_objs))
        thompson_wins = int(np.sum(thompson_objs < dqn_objs))
        ties = int(np.sum(dqn_objs == thompson_objs))

        try:
            stat, p_value = wilcoxon(dqn_objs, thompson_objs)
            is_significant = p_value < 0.05
        except:
            stat, p_value = None, None
            is_significant = False

        # Print statistics for this size
        print(f"\nSize n={size} ({len(dqn_objs)} instances):")
        print(f"  Thompson: {np.mean(thompson_objs):8.1f} ± {np.std(thompson_objs):6.1f}")
        print(f"  DQN:      {np.mean(dqn_objs):8.1f} ± {np.std(dqn_objs):6.1f}")
        print(f"  Wins: Thompson={thompson_wins}, DQN={dqn_wins}, Ties={ties}")
        if p_value is not None:
            print(f"  Wilcoxon p-value: {p_value:.4f}")

        comp = ComparisonResults(
            instance_size=size,
            algorithm1_name="Thompson",
            algorithm2_name="DQN",
            algorithm1_mean_obj=float(np.mean(thompson_objs)),
            algorithm1_std_obj=float(np.std(thompson_objs)),
            algorithm2_mean_obj=float(np.mean(dqn_objs)),
            algorithm2_std_obj=float(np.std(dqn_objs)),
            mean_percent_diff=float(np.mean(percent_diffs)),
            std_percent_diff=float(np.std(percent_diffs)),
            algorithm1_wins=thompson_wins,
            algorithm2_wins=dqn_wins,
            ties=ties,
            n_instances=len(dqn_objs),
            wilcoxon_statistic=float(stat) if stat is not None else None,
            wilcoxon_p_value=float(p_value) if p_value is not None else None,
            is_significant=is_significant,
            percent_diffs_array=percent_diffs,
        )
        comparison_results.append(comp)

    summary = ExperimentSummary(
        comparison_results=comparison_results,
        algorithm1_total_wins=sum(c.algorithm1_wins for c in comparison_results),
        algorithm2_total_wins=sum(c.algorithm2_wins for c in comparison_results),
        total_ties=sum(c.ties for c in comparison_results),
        total_instances=sum(c.n_instances for c in comparison_results),
    )

    print("\n" + "="*80)
    print(f"OVERALL: {summary.total_instances} instances")
    print(f"  Thompson wins: {summary.algorithm1_total_wins}")
    print(f"  DQN wins:      {summary.algorithm2_total_wins}")
    print(f"  Ties:          {summary.total_ties}")
    print("="*80)

    plot_comparison_results(summary, save_plot=True)
    print(f"\n✓ Saved: Thompson_vs_DQN_*.png")


def main():
    # 1. Training curves
    plot_training_curves()

    # 2. Comparison plots + statistics
    plot_comparison()

    # 3. Parameter evolution (sample instance)
    project_root = find_project_root()
    results_dir = project_root / "src/experiments/results/dqn_evaluation"
    dqn_files = sorted(results_dir.glob("*_DQN.pkl"))
    if dqn_files:
        sample_instance = dqn_files[0].stem.replace("_DQN", "")
        plot_parameter_evolution(sample_instance)
    else:
        print("  No DQN results found")

    print("\nPlots saved to: plots/")
    print("  - dqn_training_curves.png")
    print("  - Thompson_vs_DQN_*.png (comparison)")
    print("  - dqn_parameter_evolution_*.png")


if __name__ == "__main__":
    main()