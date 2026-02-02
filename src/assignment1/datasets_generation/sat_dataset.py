from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import pandas as pd

from src.instance import SocialGolferInstance
from src.sat import SATSocialGolferSolver
from src.utils import find_project_root


@dataclass
class InstanceFeatures:
    """Algorithm-agnostic features extracted from a Social Golfer Problem instance."""
    instance_name: str
    m: int  # number of groups
    n: int  # golfers per group
    w: int  # target weeks
    total_golfers: int  # m * n
    problem_size: int  # m * n * w - rough complexity indicator


def extract_features(instance: SocialGolferInstance, instance_name: str) -> InstanceFeatures:
    return InstanceFeatures(
        instance_name=instance_name,
        m=instance.m,
        n=instance.n,
        w=instance.w,
        total_golfers=instance.total_golfers,
        problem_size=instance.m * instance.n * instance.w,
    )


def _sat_worker(m: int, n: int, w: int, queue: Queue):
    """Worker function that runs in a separate process."""
    instance = SocialGolferInstance.__new__(SocialGolferInstance)
    instance.m = m
    instance.n = n
    instance.w = w
    instance.total_golfers = m * n

    solver = SATSocialGolferSolver(instance)
    satisfiable, runtime = solver.is_satisfiable()
    queue.put((satisfiable, runtime))


def solve_sat_with_timeout(
    instance: SocialGolferInstance,
    timeout_seconds: float = 60.0,
) -> tuple[Optional[int], Optional[float]]:
    """Run SAT solver in a subprocess with hard timeout."""
    queue = Queue()
    process = Process(
        target=_sat_worker,
        args=(instance.m, instance.n, instance.w, queue),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, None

    if not queue.empty():
        satisfiable, runtime = queue.get()
        if satisfiable:
            return instance.w, runtime
        else:
            return None, runtime

    return None, None


def _process_instance(instance_file: Path, timeout_seconds: float) -> dict:
    """Process a single instance and return the result row."""
    instance_name = instance_file.name
    instance = SocialGolferInstance(instance_name)
    features = extract_features(instance, instance_name)

    result, runtime = solve_sat_with_timeout(instance, timeout_seconds)

    if runtime is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TIMEOUT: {instance_name}")

    return {
        "instance_name": features.instance_name,
        "m": features.m,
        "n": features.n,
        "w": features.w,
        "total_golfers": features.total_golfers,
        "problem_size": features.problem_size,
        "sat_runtime": runtime,
        "sat_result": result,
    }


def generate_sat_dataset(
    instances_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    timeout_seconds: float = 60.0,
    flush_interval: int = 5,
    n_workers: int = 5,
) -> pd.DataFrame:
    if instances_dir is None:
        instances_dir = find_project_root() / "sgp_instances"

    if output_path is None:
        output_path = find_project_root() / "data" / "sat_dataset.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df = pd.read_csv(output_path)
        processed_instances = set(df["instance_name"].tolist())
    else:
        df = pd.DataFrame()
        processed_instances = set()

    instance_files = sorted(instances_dir.glob("*.txt"))
    pending_files = [f for f in instance_files if f.name not in processed_instances]

    if not pending_files:
        print("All instances already processed.")
        return df

    print(f"Skipping {len(processed_instances)} already processed instances.")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {len(pending_files)} remaining instances with {n_workers} workers...")

    new_rows = []
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_instance, f, timeout_seconds): f
            for f in pending_files
        }

        for future in as_completed(futures):
            row = future.result()
            new_rows.append(row)
            completed += 1

            if completed % flush_interval == 0:
                new_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(output_path, index=False)
                new_rows = []
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed {completed}/{len(pending_files)} instances to disk ({','.join([f.name for f in pending_files])})")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(output_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed final {len(new_rows)} instances to disk.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed {len(new_rows)} instances to disk ({','.join([f.name for f in pending_files])})")

    return df


if __name__ == "__main__":
    d = generate_sat_dataset(timeout_seconds=60.0)
    print("\nDataset preview:\n")
    print(d.head(10))