"""
MAB-based operator selection comparison experiment.

Compares three selection strategies:
- AdaptiveWeights (baseline)
- UCB1
- ThompsonSampling

Uses the algorithm_comparison framework for statistical comparison.
"""

from pathlib import Path
from dataclasses import dataclass

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.alns.alns import ALNS
from src.alns.config import ALNSConfig
from src.alns.operators import create_all_destroy_operators, create_all_repair_operators
from src.alns.selectors import AdaptiveWeightSelector, UCB1Selector, ThompsonSamplingSelector
from src.algorithm_comparison import (
    AlgorithmConfig,
    InstanceResult,
    compare_algorithms_across_sizes,
    plot_comparison_results,
    InstanceType
)


@dataclass
class ALNSWrapper:
    """Wrapper to make ALNS compatible with algorithm_comparison framework."""
    selector_type: str  # "adaptive", "ucb1", or "thompson"
    config: ALNSConfig = None

    def __init__(self, solution: SCFPDPSolution, selector_type: str, config: ALNSConfig = None):
        """Initialize ALNS with specified selector type."""
        self.selector_type = selector_type
        self.solution = solution
        self.config = config if config else ALNSConfig()

    def construct(self):
        """Run ALNS (mimicking construction heuristic interface)."""
        # Create initial solution using construction heuristic
        constructor = FlexiblePickupAndDropoffConstructionHeuristic(self.solution)
        constructor.construct()

        # Create operators
        destroy_ops = create_all_destroy_operators(self.config)
        repair_ops = create_all_repair_operators(self.config)

        # Create selectors based on type
        if self.selector_type == "adaptive":
            destroy_selector = AdaptiveWeightSelector(destroy_ops)
            repair_selector = AdaptiveWeightSelector(repair_ops)
        elif self.selector_type == "ucb1":
            destroy_selector = UCB1Selector(destroy_ops, exploration_constant=1.0)
            repair_selector = UCB1Selector(repair_ops, exploration_constant=1.0)
        elif self.selector_type == "thompson":
            destroy_selector = ThompsonSamplingSelector(destroy_ops)
            repair_selector = ThompsonSamplingSelector(repair_ops)
        else:
            raise ValueError(f"Unknown selector type: {self.selector_type}")

        # Run ALNS
        alns = ALNS(
            self.solution,
            config=self.config,
            destroy_selector=destroy_selector,
            repair_selector=repair_selector,
        )
        best_solution = alns.run()

        # Copy best solution back to self.solution
        self.solution.routes = best_solution.routes
        self.solution.served_requests = best_solution.served_requests


def run_comparison(
    instance_sizes: list[str],
    instance_dir: Path,
    alg1_selector: str = "adaptive",
    alg2_selector: str = "thompson",
    max_iterations: int = 1000,
    max_time_seconds: float = 300.0,
):
    """
    Run comparison between two selector strategies.

    Args:
        instance_sizes: List of instance sizes (e.g., ["50", "100"])
        instance_dir: Base directory containing instance folders
        alg1_selector: First selector type ("adaptive", "ucb1", "thompson")
        alg2_selector: Second selector type
        max_iterations: Max ALNS iterations
        max_time_seconds: Max ALNS runtime
    """
    # Create ALNS config
    config = ALNSConfig(
        max_iterations=max_iterations,
        max_time_seconds=max_time_seconds,
        log_interval=10000,  # Minimal logging
    )

    # Create algorithm configs
    alg1_config = AlgorithmConfig(
        name=alg1_selector.capitalize(),
        algorithm_class=ALNSWrapper,
        init_kwargs={"selector_type": alg1_selector, "config": config}
    )

    alg2_config = AlgorithmConfig(
        name=alg2_selector.capitalize(),
        algorithm_class=ALNSWrapper,
        init_kwargs={"selector_type": alg2_selector, "config": config}
    )

    # Run comparison using existing framework
    # Note: This expects instances in instance_dir/<size>/test/*.txt
    summary = compare_algorithms_across_sizes(
        instance_sizes=instance_sizes,
        instance_type=InstanceType.TEST,
        algorithm1_config=alg1_config,
        algorithm2_config=alg2_config,
    )

    # Plot results
    plot_comparison_results(summary, save_plot=True)

    return summary


if __name__ == "__main__":
    instance_sizes = ["50", "100", "200", "500"]

    print("Running MAB Selector Comparison")
    print("="*80)

    # Comparison 1: Adaptive vs Thompson Sampling
    print("\n1. Adaptive Weights vs Thompson Sampling")
    summary_thompson = run_comparison(
        instance_sizes=instance_sizes,
        instance_dir=Path("scfpdp_instances"),
        alg1_selector="adaptive",
        alg2_selector="thompson",
        max_iterations=500,  # Quick test
    )

    # Comparison 2: Adaptive vs UCB1
    print("\n2. Adaptive Weights vs UCB1")
    summary_ucb1 = run_comparison(
        instance_sizes=instance_sizes,
        instance_dir=Path("scfpdp_instances"),
        alg1_selector="adaptive",
        alg2_selector="ucb1",
        max_iterations=500,
    )

    # Comparison 3: UCB1 vs Thompson Sampling
    print("\n3. UCB1 vs Thompson Sampling")
    summary_ucb_thompson = run_comparison(
        instance_sizes=instance_sizes,
        instance_dir=Path("scfpdp_instances"),
        alg1_selector="ucb1",
        alg2_selector="thompson",
        max_iterations=500,
    )

    print("\n" + "="*80)
    print("All comparisons complete. Check plots/ directory for results.")