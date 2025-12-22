"""
Visualization: Feature Importance from Random Forest Model.

Bar chart showing which features most influence algorithm selection decisions.
"""
import pandas as pd
from matplotlib import pyplot as plt

from src.algorithm_selection import AlgorithmSelector, run_algorithm_selection_experiment
from src.utils import find_project_root


def plot_feature_importance(
    selector: AlgorithmSelector = None,
    save: bool = True,
):
    """
    Plot feature importance from trained Random Forest model.

    Args:
        selector: Trained AlgorithmSelector (if None, trains a new one)
        save: Whether to save the plot
    """
    # Train model if not provided
    if selector is None:
        print("Training Random Forest model...")
        selector = run_algorithm_selection_experiment(test_size=0.35, n_estimators=100)
        print()

    # Get feature importances
    importance_df = selector.get_feature_importance()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bar chart
    colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
    bars = ax.barh(
        importance_df['feature'],
        importance_df['importance'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )

    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(
            row['importance'] + 0.005,
            i,
            f"{row['importance']:.3f}",
            va='center',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Feature Importance for Algorithm Selection", fontsize=14, fontweight="bold")
    ax.set_xlim(0, importance_df['importance'].max() * 1.15)
    ax.grid(True, alpha=0.3, axis='x')

    # Add model info
    info_text = (
        f"Model: Random Forest\n"
        f"Trees: {selector.n_estimators}\n"
        f"Test Accuracy: {selector.test_accuracy:.3f}"
    )
    ax.text(
        0.98, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )

    plt.tight_layout()

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / "feature_importance.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved to {filename}")

    plt.show()

    return importance_df


def plot_feature_importance_comparison(save: bool = True):
    """
    Compare feature importance across different model configurations.

    Args:
        save: Whether to save the plot
    """
    print("Training models with different configurations...")

    configs = [
        {"n_estimators": 50, "max_depth": 5, "label": "RF-50-d5"},
        {"n_estimators": 100, "max_depth": None, "label": "RF-100-full"},
        {"n_estimators": 200, "max_depth": 10, "label": "RF-200-d10"},
    ]

    all_importances = []

    for config in configs:
        selector = AlgorithmSelector(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
        )
        X_train, X_test, y_train, y_test = selector.load_data(test_size=0.35)
        selector.train(X_train, y_train)
        selector.evaluate(X_test, y_test)

        importance_df = selector.get_feature_importance()
        importance_df['model'] = config['label']
        all_importances.append(importance_df)

    # Combine all importances
    combined = pd.concat(all_importances)

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    features = combined['feature'].unique()
    x = range(len(features))
    width = 0.25

    for i, config in enumerate(configs):
        model_data = combined[combined['model'] == config['label']]
        importances = [
            model_data[model_data['feature'] == f]['importance'].values[0]
            for f in features
        ]
        ax.bar(
            [xi + i * width for xi in x],
            importances,
            width,
            label=config['label'],
            alpha=0.8,
        )

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        plots_dir = find_project_root() / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = plots_dir / "feature_importance_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved to {filename}")

    plt.show()


if __name__ == "__main__":
    # Single model feature importance
    importance_df = plot_feature_importance(save=True)
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))