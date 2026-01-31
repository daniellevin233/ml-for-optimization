"""
Detailed ALNS experiment runner that saves statistics for analysis.

Runs ALNS with different selectors and saves:
- Final objectives
- Full ALNSStatistics for convergence plotting
- Operator selection data for evolution analysis
"""

import pickle
from pathlib import Path
from dataclasses import dataclass
import time

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.alns.alns import ALNS
from src.alns.config import ALNSConfig
from src.alns.operators import create_all_destroy_operators, create_all_repair_operators
from src.alns.selectors import AdaptiveWeightSelector, UCB1Selector, ThompsonSamplingSelector


@dataclass
class ExperimentResult:
    """Results from a single ALNS run."""
    instance_name: str
    algorithm_name: str
    initial_objective: float
    final_objective: float
    improvement_pct: float
    runtime_seconds: float
    iterations: int
    statistics: list


def run_single_experiment(
    instance_path: Path,
    selector_type: str,
    config: ALNSConfig
) -> ExperimentResult:
    """
    Run ALNS with specified selector on a single instance.

    Args:
        instance_path: Path to instance file
        selector_type: "adaptive", "ucb1", or "thompson"
        config: ALNS configuration

    Returns:
        ExperimentResult with statistics
    """
    start_time = time.time()

    # Load instance and create initial solution
    instance = SCFPDPInstance(str(instance_path))
    solution = SCFPDPSolution(instance)
    constructor = FlexiblePickupAndDropoffConstructionHeuristic(solution)
    constructor.construct()
    initial_objective = solution.calc_objective()

    # Create operators
    destroy_ops = create_all_destroy_operators(config)
    repair_ops = create_all_repair_operators(config)

    # Create selectors based on type
    if selector_type == "adaptive":
        destroy_selector = AdaptiveWeightSelector(destroy_ops)
        repair_selector = AdaptiveWeightSelector(repair_ops)
    elif selector_type == "ucb1":
        destroy_selector = UCB1Selector(destroy_ops, exploration_constant=1.0)
        repair_selector = UCB1Selector(repair_ops, exploration_constant=1.0)
    elif selector_type == "thompson":
        destroy_selector = ThompsonSamplingSelector(destroy_ops)
        repair_selector = ThompsonSamplingSelector(repair_ops)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")

    # Run ALNS
    alns = ALNS(
        solution,
        config=config,
        destroy_selector=destroy_selector,
        repair_selector=repair_selector,
    )
    best_solution = alns.run()
    final_objective = best_solution.calc_objective()

    runtime = time.time() - start_time
    improvement = (initial_objective - final_objective) / initial_objective * 100

    return ExperimentResult(
        instance_name=instance_path.stem,
        algorithm_name=selector_type.capitalize(),
        initial_objective=initial_objective,
        final_objective=final_objective,
        improvement_pct=improvement,
        runtime_seconds=runtime,
        iterations=len(alns.statistics),
        statistics=alns.statistics
    )


def run_experiment_suite(
    instance_paths: list[Path],
    algorithms: list[str] = ["adaptive", "ucb1", "thompson"],
    config: ALNSConfig = None,
    results_dir: Path = Path("results/mab_detailed")
):
    """
    Run complete experiment suite and save results.

    Args:
        instance_paths: List of instance file paths
        algorithms: List of selector types to test
        config: ALNS configuration
        results_dir: Directory to save results
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = ALNSConfig(max_iterations=1000, log_interval=10000)

    print(f"Running experiments on {len(instance_paths)} instances")
    print(f"Algorithms: {algorithms}")
    print(f"Max iterations: {config.max_iterations}")
    print("="*80)

    all_results = []

    for instance_path in instance_paths:
        print(f"\nInstance: {instance_path.name}")

        for selector_type in algorithms:
            print(f"  {selector_type.capitalize()}...", end=" ", flush=True)

            result = run_single_experiment(instance_path, selector_type, config)

            # Save statistics
            stats_file = results_dir / f"{result.instance_name}_{result.algorithm_name}.pkl"
            with open(stats_file, 'wb') as f:
                pickle.dump(result.statistics, f)

            all_results.append(result)

            print(f"Final: {result.final_objective:.2f} ({result.improvement_pct:+.1f}%), "
                  f"{result.iterations} iters, {result.runtime_seconds:.1f}s")

    # Save summary
    summary_file = results_dir / "summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)

    print("\n" + "="*80)
    print("Experiment complete")
    print(f"Results saved to {results_dir}")
    print(f"Statistics files: {len(all_results)}")

    # Print summary table
    print_summary_table(all_results)


def print_summary_table(results: list[ExperimentResult]):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Group by instance
    from collections import defaultdict
    by_instance = defaultdict(list)
    for r in results:
        by_instance[r.instance_name].append(r)

    for instance_name, instance_results in by_instance.items():
        print(f"\n{instance_name}:")
        for r in instance_results:
            print(f"  {r.algorithm_name:12s}: {r.final_objective:8.2f} "
                  f"({r.improvement_pct:+6.1f}%) [{r.iterations:4d} iters]")


if __name__ == "__main__":
    instance_paths = [
        Path("scfpdp_instances/50/test_instance_n50_01.txt"),
        Path("scfpdp_instances/50/test_instance_n50_02.txt"),
    ]

    config = ALNSConfig(
        max_iterations=500,
        max_time_seconds=60.0,
        log_interval=10000
    )

    run_experiment_suite(
        instance_paths=instance_paths,
        algorithms=["adaptive", "ucb1", "thompson"],
        config=config
    )

    print("\nTo plot results, run:")
    print("  python -m src.experiments.plot_convergence")
    print("  python -m src.experiments.plot_operator_evolution")