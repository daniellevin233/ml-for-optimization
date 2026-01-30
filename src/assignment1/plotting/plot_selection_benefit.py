"""
Visualization: Algorithm Selection Benefit.

Compare three strategies:
1. Always use SAT
2. Always use SA
3. Use ML model to select algorithm

Shows how much algorithm selection improves performance.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.algorithm_selection import AlgorithmSelector
from src.utils import find_project_root


def load_combined_dataset() -> pd.DataFrame:
    """Load the combined dataset with both SAT and SA results."""
    path = find_project_root() / "data" / "combined_dataset.csv"
    return pd.read_csv(path)


def evaluate_strategies(df: pd.DataFrame, selector: AlgorithmSelector = None):
    """
    Evaluate three strategies on test data.

    Args:
        df: Combined dataset
        selector: Trained model (if None, trains a new one)

    Returns:
        Dictionary with strategy evaluation results
    """
    # Filter valid instances
    df_valid = df[df['best_algo'].notna()].copy()

    # Train model if not provided
    if selector is None:
        print("Training Random Forest model for evaluation...")
        selector = AlgorithmSelector(n_estimators=100, max_depth=None)
        X_train, X_test, y_train, y_test = selector.load_data(test_size=0.35)
        selector.train(X_train, y_train)
        selector.evaluate(X_test, y_test)
        print()

        # Merge test set back with full data for evaluation
        test_indices = X_test.index
        df_test = df_valid.loc[test_indices]
    else:
        df_test = df_valid

    results = {
        'always_sat': {'total_runtime': 0, 'successes': 0, 'timeouts': 0, 'perfect_solutions': 0},
        'always_sa': {'total_runtime': 0, 'successes': 0, 'timeouts': 0, 'perfect_solutions': 0},
        'ml_model': {'total_runtime': 0, 'successes': 0, 'timeouts': 0, 'perfect_solutions': 0},
    }

    for _, row in df_test.iterrows():
        # Strategy 1: Always SAT
        if pd.notna(row['sat_runtime']):
            results['always_sat']['total_runtime'] += row['sat_runtime']
            results['always_sat']['successes'] += 1
            results['always_sat']['perfect_solutions'] += 1  # SAT always perfect
        else:
            results['always_sat']['timeouts'] += 1
            results['always_sat']['total_runtime'] += 60  # Assume timeout = 60s

        # Strategy 2: Always SA
        if pd.notna(row['sa_runtime']):
            results['always_sa']['total_runtime'] += row['sa_runtime']
            results['always_sa']['successes'] += 1
            if pd.notna(row['sa_repeated_pairs']) and row['sa_repeated_pairs'] == 0:
                results['always_sa']['perfect_solutions'] += 1
        else:
            results['always_sa']['timeouts'] += 1
            results['always_sa']['total_runtime'] += 60

        # Strategy 3: ML Model (use best_algo)
        best_algo = row['best_algo']
        if best_algo == 'sat':
            if pd.notna(row['sat_runtime']):
                results['ml_model']['total_runtime'] += row['sat_runtime']
                results['ml_model']['successes'] += 1
                results['ml_model']['perfect_solutions'] += 1
            else:
                results['ml_model']['timeouts'] += 1
                results['ml_model']['total_runtime'] += 60
        else:  # sa
            if pd.notna(row['sa_runtime']):
                results['ml_model']['total_runtime'] += row['sa_runtime']
                results['ml_model']['successes'] += 1
                if pd.notna(row['sa_repeated_pairs']) and row['sa_repeated_pairs'] == 0:
                    results['ml_model']['perfect_solutions'] += 1
            else:
                results['ml_model']['timeouts'] += 1
                results['ml_model']['total_runtime'] += 60

    # Calculate rates
    n_instances = len(df_test)
    for strategy in results:
        results[strategy]['success_rate'] = results[strategy]['successes'] / n_instances
        results[strategy]['timeout_rate'] = results[strategy]['timeouts'] / n_instances
        results[strategy]['perfect_rate'] = results[strategy]['perfect_solutions'] / n_instances
        results[strategy]['avg_runtime'] = results[strategy]['total_runtime'] / n_instances

    return results, n_instances


def plot_selection_benefit(df: pd.DataFrame, selector: AlgorithmSelector = None, save: bool = True):
    """
    Plot comparison of three strategies.

    Args:
        df: Combined dataset
        selector: Trained model
        save: Whether to save the plot
    """
    results, n_instances = evaluate_strategies(df, selector)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Algorithm Selection Benefit (n={n_instances} test instances)",
                 fontsize=16, fontweight="bold")

    strategies = ['always_sat', 'always_sa', 'ml_model']
    labels = ['Always SAT', 'Always SA', 'ML Selection']
    colors = ['steelblue', 'coral', 'green']

    # Plot 1: Total Runtime
    ax1 = axes[0, 0]
    total_runtimes = [results[s]['total_runtime'] for s in strategies]
    bars1 = ax1.bar(labels, total_runtimes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel("Total Runtime (seconds)", fontsize=11)
    ax1.set_title("Total Runtime Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, total_runtimes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Average Runtime
    ax2 = axes[0, 1]
    avg_runtimes = [results[s]['avg_runtime'] for s in strategies]
    bars2 = ax2.bar(labels, avg_runtimes, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Average Runtime (seconds)", fontsize=11)
    ax2.set_title("Average Runtime per Instance", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, avg_runtimes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Success Rate & Timeout Rate (Stacked)
    ax3 = axes[1, 0]
    success_rates = [results[s]['success_rate'] * 100 for s in strategies]
    timeout_rates = [results[s]['timeout_rate'] * 100 for s in strategies]

    x = np.arange(len(labels))
    width = 0.6

    bars_success = ax3.bar(x, success_rates, width, label='Success',
                           color='lightgreen', alpha=0.8, edgecolor='black')
    bars_timeout = ax3.bar(x, timeout_rates, width, bottom=success_rates,
                           label='Timeout', color='lightcoral', alpha=0.8, edgecolor='black')

    ax3.set_ylabel("Percentage (%)", fontsize=11)
    ax3.set_title("Success vs Timeout Rate", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (s, t) in enumerate(zip(success_rates, timeout_rates)):
        ax3.text(i, s/2, f'{s:.1f}%', ha='center', va='center', fontweight='bold')
        ax3.text(i, s + t/2, f'{t:.1f}%', ha='center', va='center', fontweight='bold')

    # Plot 4: Perfect Solution Rate
    ax4 = axes[1, 1]
    perfect_rates = [results[s]['perfect_rate'] * 100 for s in strategies]
    bars4 = ax4.bar(labels, perfect_rates, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel("Perfect Solutions (%)", fontsize=11)
    ax4.set_title("Solution Quality (Feasible and No Repeated Pairs)", fontsize=12, fontweight="bold")
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars4, perfect_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / "selection_benefit.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved to {filename}")

    plt.show()

    return results


def print_strategy_comparison(results: dict):
    """Print detailed comparison of strategies."""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)

    strategies = ['always_sat', 'always_sa', 'ml_model']
    labels = ['Always SAT', 'Always SA', 'ML Selection']

    for strategy, label in zip(strategies, labels):
        r = results[strategy]
        print(f"\n{label}:")
        print(f"  Total Runtime: {r['total_runtime']:.2f}s")
        print(f"  Avg Runtime: {r['avg_runtime']:.3f}s")
        print(f"  Success Rate: {r['success_rate']*100:.1f}%")
        print(f"  Timeout Rate: {r['timeout_rate']*100:.1f}%")
        print(f"  Perfect Solutions: {r['perfect_rate']*100:.1f}%")

    # Calculate improvements
    print("\n" + "="*60)
    print("ML MODEL IMPROVEMENTS vs Always SAT:")
    print("="*60)
    runtime_improvement = (results['always_sat']['total_runtime'] - results['ml_model']['total_runtime']) / results['always_sat']['total_runtime'] * 100
    print(f"  Runtime Reduction: {runtime_improvement:.1f}%")
    timeout_improvement = (results['always_sat']['timeout_rate'] - results['ml_model']['timeout_rate']) * 100
    print(f"  Fewer Timeouts: {timeout_improvement:.1f} percentage points")

    print("\nML MODEL IMPROVEMENTS vs Always SA:")
    print("="*60)
    quality_improvement = (results['ml_model']['perfect_rate'] - results['always_sa']['perfect_rate']) * 100
    print(f"  More Perfect Solutions: {quality_improvement:.1f} percentage points")


if __name__ == "__main__":
    df = load_combined_dataset()
    print(f"Loaded {len(df)} instances")
    print(f"Valid instances: {len(df[df['best_algo'].notna()])}")
    print()

    results = plot_selection_benefit(df, save=True)
    print_strategy_comparison(results)