"""
Combined dataset generation: merges SAT and Simulated Annealing results.

Takes sat_dataset.csv and augments it with SA results from Sim_Anneal_dataset.csv.
For missing instances, runs SA on the fly.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import pandas as pd

from src.Sim_Anneal_dataset import SocialGolferSolver
from src.utils import find_project_root


def _sa_worker(m: int, n: int, w: int, queue: Queue):
    """Worker function that runs SA in a separate process."""
    solver = SocialGolferSolver(max_iterations=5000, initial_temp=10.0, cooling_rate=0.995)
    total_golfers = m * n

    start_time = time.time()
    result = solver.solve(total_golfers, n, w)
    runtime = time.time() - start_time

    # Calculate sa_result: total_meetings / meetings_per_week = w (weeks achieved)
    meetings_per_week = m * (n * (n - 1) // 2)
    total_meetings = meetings_per_week * w
    sa_result_weeks = total_meetings / meetings_per_week if meetings_per_week > 0 else 0

    queue.put({
        'sa_runtime': runtime,
        'sa_result': sa_result_weeks,
        'sa_repeated_pairs': result.get('repeated_pairs', None),
    })


def solve_sa_with_timeout(m: int, n: int, w: int, timeout_seconds: float = 60.0) -> dict:
    """Run SA solver in a subprocess with hard timeout."""
    queue = Queue()
    process = Process(target=_sa_worker, args=(m, n, w, queue))

    start_time = time.time()
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {'sa_runtime': None, 'sa_result': None, 'sa_repeated_pairs': None}

    if not queue.empty():
        result = queue.get()
        result['sa_runtime'] = time.time() - start_time
        return result

    return {'sa_runtime': None, 'sa_result': None, 'sa_repeated_pairs': None}


def load_sa_dataset() -> pd.DataFrame:
    """Load the Sim_Anneal_dataset.csv file."""
    path = find_project_root() / "data" / "Sim_Anneal_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_sa_result(sa_df: pd.DataFrame, m: int, n: int, w: int) -> Optional[dict]:
    """
    Find SA result for given instance parameters.

    Mapping: classification(m, n, w) = SA(groups_per_week, group_size, weeks)
    """
    if sa_df.empty:
        return None

    # Filter for matching instance
    match = sa_df[
        (sa_df['groups_per_week'] == m) &
        (sa_df['group_size'] == n) &
        (sa_df['weeks'] == w)
    ]

    if match.empty:
        return None

    # Use the first match
    row = match.iloc[0]

    # Extract and transform SA results
    sa_runtime = row.get('sa_runtime_seconds', None)
    meetings_per_week = row.get('meetings_per_week', m * (n * (n - 1) // 2))
    total_meetings = row.get('total_meetings', meetings_per_week * w)

    sa_result_weeks = total_meetings / meetings_per_week if meetings_per_week > 0 else None

    return {
        'sa_runtime': sa_runtime,
        'sa_result': sa_result_weeks,
        'sa_repeated_pairs': row.get('sa_repeated_pairs', None),
    }


def compute_best_algo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'best_algo' column based on algorithm performance.

    Logic:
    1. If SAT failed (NaN runtime) → 'sa'
    2. If SAT succeeded and SA has repeated pairs (sa_repeated_pairs > 0) → 'sat'
    3. If SAT succeeded and SA is perfect (sa_repeated_pairs == 0) → choose faster one
    """
    def select_best(row):
        sat_runtime = row['sat_runtime']
        sa_runtime = row['sa_runtime']
        sa_repeated_pairs = row['sa_repeated_pairs']

        # SAT failed → SA
        if pd.isna(sat_runtime):
            return 'sa'

        # SAT succeeded and SA has repeated pairs → SAT
        if not pd.isna(sa_repeated_pairs) and sa_repeated_pairs > 0:
            return 'sat'

        # Both succeeded, SA is perfect → choose faster
        if not pd.isna(sa_runtime):
            return 'sat' if sat_runtime <= sa_runtime else 'sa'

        # SA failed or missing → SAT
        return 'sat'

    df['best_algo'] = df.apply(select_best, axis=1)
    return df


def _process_instance(
    row: pd.Series,
    sa_df: pd.DataFrame,
    timeout_seconds: float,
) -> dict:
    """Process a single instance: find or run SA."""
    instance_name = row['instance_name']
    m = int(row['m'])
    n = int(row['n'])
    w = int(row['w'])

    # Try to find existing SA result
    sa_result = find_sa_result(sa_df, m, n, w)

    if sa_result is None:
        # Run SA on the fly
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running SA for {instance_name}...")
        sa_result = solve_sa_with_timeout(m, n, w, timeout_seconds)

        if sa_result['sa_runtime'] is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] SA TIMEOUT: {instance_name}")

    # Combine classification data with SA results
    combined_row = row.to_dict()
    combined_row.update(sa_result)

    return combined_row


def generate_combined_dataset(
    classification_path: Optional[Path] = None,
    sa_dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    timeout_seconds: float = 60.0,
    flush_interval: int = 5,
    n_workers: int = 5,
) -> pd.DataFrame:
    """
    Generate combined dataset with SAT and SA results.

    Args:
        classification_path: Path to sat_dataset.csv (SAT results)
        sa_dataset_path: Path to Sim_Anneal_dataset.csv
        output_path: Path to save combined CSV
        timeout_seconds: Timeout for running SA on missing instances
        flush_interval: Flush results every N instances
        n_workers: Number of parallel workers

    Returns:
        Combined DataFrame with SAT and SA results
    """
    if classification_path is None:
        classification_path = find_project_root() / "data" / "sat_dataset.csv"

    if sa_dataset_path is None:
        sa_dataset_path = find_project_root() / "data" / "Sim_Anneal_dataset.csv"

    if output_path is None:
        output_path = find_project_root() / "data" / "combined_dataset.csv"

    # Load classification dataset (SAT results)
    if not classification_path.exists():
        raise FileNotFoundError(f"Classification dataset not found: {classification_path}")

    classification_df = pd.read_csv(classification_path)

    # Load SA dataset
    sa_df = load_sa_dataset()
    if not sa_df.empty:
        print(f"Loaded {len(sa_df)} SA results from {sa_dataset_path}")
    else:
        print("No existing SA dataset found, will run SA for all instances")

    # Load existing combined dataset if available
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        combined_df = pd.read_csv(output_path)
        processed_instances = set(combined_df['instance_name'].tolist())
        print(f"Loaded existing combined dataset with {len(combined_df)} instances")
    else:
        combined_df = pd.DataFrame()
        processed_instances = set()

    # Find instances that need processing
    pending_rows = classification_df[~classification_df['instance_name'].isin(processed_instances)]

    if pending_rows.empty:
        print("All instances already processed.")
        # Recompute best_algo if missing or update it
        if 'best_algo' not in combined_df.columns or combined_df['best_algo'].isna().any():
            combined_df = compute_best_algo(combined_df)
            combined_df.to_csv(output_path, index=False)
            print(f"Updated best_algo column in {output_path}")
        return combined_df

    print(f"Skipping {len(processed_instances)} already processed instances.")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {len(pending_rows)} remaining instances with {n_workers} workers...")

    new_rows = []
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_instance, row, sa_df, timeout_seconds): row['instance_name']
            for _, row in pending_rows.iterrows()
        }

        for future in as_completed(futures):
            row = future.result()
            new_rows.append(row)
            completed += 1

            if completed % flush_interval == 0:
                new_df = pd.DataFrame(new_rows)
                combined_df = pd.concat([combined_df, new_df], ignore_index=True)
                combined_df.to_csv(output_path, index=False)
                new_rows = []
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed {completed}/{len(pending_rows)} instances to disk.")

    # Final flush
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([combined_df, new_df], ignore_index=True)

    # Compute best_algo column
    combined_df = compute_best_algo(combined_df)
    combined_df.to_csv(output_path, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed final {len(new_rows) if new_rows else 0} instances to disk.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Combined dataset saved to {output_path}")
    print(f"Total instances: {len(combined_df)}")
    print(f"SAT solved: {combined_df['sat_runtime'].notna().sum()}")
    print(f"SA completed: {combined_df['sa_runtime'].notna().sum()}")
    print(f"Best algo: SAT={len(combined_df[combined_df['best_algo']=='sat'])}, SA={len(combined_df[combined_df['best_algo']=='sa'])}")

    return combined_df


if __name__ == "__main__":
    df = generate_combined_dataset(timeout_seconds=60.0, flush_interval=5, n_workers=5)
    print("\nCombined dataset preview:")
    print(df.head(10))