import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from src.alns.config import ALNSConfig
from src.experiments.run_detailed_experiment import run_single_experiment
from src.utils import find_project_root


def run_instance_algorithms(instance_path, algorithms, config):
    return [run_single_experiment(instance_path, alg, config) for alg in algorithms]


def evaluate_dqn():
    """
    Evaluate DQN vs Thompson on test instances.

    Tests on 4 sizes (50, 100, 200, 500) with 30 instances each.
    """
    project_root = find_project_root()

    instance_paths = []
    sizes = ["50", "100", "200", "500"]
    for size in sizes:
        paths = sorted((project_root / f"scfpdp_instances/{size}/test").glob("*.txt"))
        instance_paths.extend(paths)
        print(f"Size {size}: {len(paths)} instances")

    print(f"\nTotal: {len(instance_paths)} instances")
    print("Algorithms: DQN, Thompson")

    config = ALNSConfig(max_iterations=500, max_time_seconds=100, log_interval=10000)
    algorithms = ["dqn", "thompson"]

    # Run in parallel
    n_workers = min(cpu_count(), len(instance_paths))
    process_func = partial(run_instance_algorithms, algorithms=algorithms, config=config)

    with Pool(n_workers) as pool:
        results_nested = list(tqdm(
            pool.imap(process_func, instance_paths, chunksize=1),
            total=len(instance_paths),
            desc="Evaluating DQN vs Thompson"
        ))

    results = [r for batch in results_nested for r in batch]

    results_dir = project_root / "src/experiments/results/dqn_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        stats_file = results_dir / f"{result.instance_name}_{result.algorithm_name}.pkl"
        with open(stats_file, "wb") as f:
            pickle.dump(result.statistics, f)

    with open(results_dir / "summary.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {results_dir}")
    print(f"  Total: {len(results)} runs")


if __name__ == "__main__":
    evaluate_dqn()