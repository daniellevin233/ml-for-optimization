from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm with tunable parameters."""

    # ===== STOPPING CRITERIA =====
    max_iterations: int = 10000
    max_time_seconds: float = 300.0  # 5 minutes
    max_iterations_without_improvement: int = 1000

    # ===== WEIGHT MANAGEMENT =====
    weight_update_period: int = 100  # Update weights every p iterations
    reaction_factor: float = 0.1  # γ ∈ [0, 1]; higher = faster adaptation towards successful destroy/repair operations

    # ===== DESTROY PARAMETERS =====
    min_removal_percentage: float = 0.10  # Remove at least 10% of served requests
    max_removal_percentage: float = 0.40  # Remove at most 40% of served requests

    # ===== SIMULATED ANNEALING PARAMETERS =====
    initial_temperature: float = 100.0
    cooling_rate: float = 0.99  # T = T * cooling_rate each iteration

    # ===== SUCCESS SCORING =====
    # Operators are rewarded based on solution quality to guide weight adaptation.
    # Higher scores incentivize operators that find better solutions.
    # The ratio (typically 5:1 to 13:1 in literature) balances quality vs. acceptance rate.
    #
    # Example: If operator used 10 times:
    #   - Finds 1 new best → score = 10, success_rate = 10/10 = 1.0
    #   - Gets 5 accepted  → score = 5,  success_rate = 5/10 = 0.5
    # This creates preference for quality improvements over mere acceptances.
    score_new_best: float = 10.0  # Reward for finding new global best
    score_accepted: float = 1.0   # Reward for accepted solution (not new best)
    # score_rejected is implicitly 0.0

    # ===== LOGGING =====
    log_interval: int = 1000  # Print progress every N iterations

    @staticmethod
    def from_tuned_params(
        instance_size: int,
        tuning_dir: Path = None,
        **override_params
    ) -> 'ALNSConfig':
        """
        Load tuned parameters for a given instance size from tuning directory.

        If exact size not found and fallback_to_smaller=True, searches for the largest
        available size smaller than the requested size.
        """
        if tuning_dir is None:
            from src.utils import find_project_root
            project_root = find_project_root()
            tuning_dir = project_root / "src" / "algorithms" / "alns" / "tuning"

        # Try exact size first
        config_file = tuning_dir / f"tuned_params_n{instance_size}.json"

        if not config_file.exists():
            # Find all available tuned configs
            available_configs = sorted(tuning_dir.glob("tuned_params_n*.json"))

            # Extract sizes from filenames
            available_sizes = []
            for file in available_configs:
                try:
                    # Extract size from filename like "tuned_params_n50.json" or "tuned_params_n50_20260117_180530.json"
                    filename = file.stem  # Remove .json
                    if "_" in filename and not filename.endswith("_n" + str(instance_size)):
                        # Skip timestamped versions if non-timestamped exists
                        base_file = tuning_dir / (filename.split("_")[0] + "_" + filename.split("_")[1] + ".json")
                        if base_file.exists():
                            continue

                    size_str = filename.split("_n")[1].split("_")[0]
                    size = int(size_str)
                    if size < instance_size:
                        available_sizes.append((size, file))
                except (IndexError, ValueError):
                    continue

            if available_sizes:
                # Use largest size smaller than target
                fallback_size, config_file = max(available_sizes, key=lambda x: x[0])
                # print(f"[ALNS Config] No tuned parameters found for n={instance_size}")
                # print(f"[ALNS Config] Falling back to n={fallback_size} (next smaller available size)")
                # print(f"[ALNS Config] Config file: {config_file}")
            else:
                raise FileNotFoundError(
                    f"No tuned parameters found for n={instance_size} or any smaller size in {tuning_dir}"
                )

        # Load the config file
        with open(config_file, 'r') as f:
            tuning_result = json.load(f)

        best_params = tuning_result["best_params"]

        # Create config with tuned parameters
        config = ALNSConfig(
            weight_update_period=best_params["weight_update_period"],
            reaction_factor=best_params["reaction_factor"],
            min_removal_percentage=best_params["min_removal_pct"],
            max_removal_percentage=best_params["max_removal_pct"],
            initial_temperature=best_params["initial_temp"],
            cooling_rate=best_params["cooling_rate"],
            score_new_best=best_params["score_new_best"],
            score_accepted=best_params["score_accepted"],
            **override_params
        )

        # print(f"[ALNS Config] Loaded tuned parameters successfully")
        return config
