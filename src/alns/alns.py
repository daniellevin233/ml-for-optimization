from dataclasses import dataclass
import random
import time
import numpy as np

from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution
from src.alns.config import ALNSConfig
from src.alns.operators import (
    DestroyOperator,
    RepairOperator,
    create_all_destroy_operators,
    create_all_repair_operators
)
from src.alns.selectors import OperatorSelector, AdaptiveWeightSelector


@dataclass
class ALNSStatistics:
    """Track ALNS performance statistics per iteration."""
    iteration: int
    best_objective: float
    current_objective: float
    temperature: float
    destroy_operator_used: str
    repair_operator_used: str
    improvement: bool
    accepted: bool
    time_elapsed: float
    destroy_selector_stats: dict
    repair_selector_stats: dict


class ALNS:
    """Adaptive Large Neighborhood Search for SCF-PDP."""

    def __init__(
        self,
        initial_solution: SCFPDPSolution,
        destroy_operators: list[DestroyOperator] = None,
        repair_operators: list[RepairOperator] = None,
        config: ALNSConfig = None,
        destroy_selector: OperatorSelector = None,
        repair_selector: OperatorSelector = None,
    ):
        self.initial_solution = initial_solution

        self.config = config if config is not None else ALNSConfig()

        # Auto-instantiate all operators if not provided
        self.destroy_operators = (
            destroy_operators if destroy_operators is not None
            else create_all_destroy_operators(self.config)
        )
        self.repair_operators = (
            repair_operators if repair_operators is not None
            else create_all_repair_operators(self.config)
        )

        # Use provided selectors or default to adaptive weights
        self.destroy_selector = (
            destroy_selector if destroy_selector is not None
            else AdaptiveWeightSelector(
                self.destroy_operators,
                reaction_factor=self.config.reaction_factor,
                update_period=self.config.weight_update_period,
                score_new_best=self.config.score_new_best,
                score_accepted=self.config.score_accepted,
            )
        )
        self.repair_selector = (
            repair_selector if repair_selector is not None
            else AdaptiveWeightSelector(
                self.repair_operators,
                reaction_factor=self.config.reaction_factor,
                update_period=self.config.weight_update_period,
                score_new_best=self.config.score_new_best,
                score_accepted=self.config.score_accepted,
            )
        )

        # Best and current solutions
        self.best_solution = initial_solution.copy()
        self.best_objective = initial_solution.calc_objective()
        self.initial_objective = self.best_objective
        self.current_solution = initial_solution.copy()
        self.current_objective = self.best_objective

        # Simulated Annealing state
        self.temperature = self.config.initial_temperature

        # Statistics
        self.statistics: list[ALNSStatistics] = []
        self.start_time = None
        self.iterations_without_improvement = 0

    def run(self) -> SCFPDPSolution:
        """Main ALNS loop."""
        self.start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # Check stopping criteria
            if self._should_stop(iteration):
                break

            # Build state dict for RL selectors
            state = {
                "current_objective": self.current_objective,
                "best_objective": self.best_objective,
                "initial_objective": self.initial_objective,
                "temperature": self.temperature,
                "initial_temperature": self.config.initial_temperature,
                "iterations_without_improvement": self.iterations_without_improvement,
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
                "statistics": self.statistics,
            }

            if self.destroy_selector.__class__.__name__ == "DQNSelector":
                destroy_op, repair_op, severity, temp_mult = self.destroy_selector.select(state)

                old_min_sev, old_max_sev = self.config.min_removal_percentage, self.config.max_removal_percentage
                self.config.min_removal_percentage = severity
                self.config.max_removal_percentage = severity
            else:
                destroy_op = self.destroy_selector.select(state)
                repair_op = self.repair_selector.select(state)
                temp_mult = 1.0  # No temperature control for MAB

            # Apply destroy-repair
            destroyed_solution = destroy_op.apply(self.current_solution)
            new_solution = repair_op.apply(destroyed_solution)
            new_objective = new_solution.calc_objective()

            # Evaluate and accept/reject
            delta = new_objective - self.current_objective
            accepted, score = self._accept_solution(new_objective, delta)

            if accepted:
                self.current_solution = new_solution
                self.current_objective = new_objective

                # Check if new global best
                if new_objective < self.best_objective:
                    self.best_solution = new_solution.copy()
                    self.best_objective = new_objective
                    self.iterations_without_improvement = 0
                else:
                    self.iterations_without_improvement += 1
            else:
                self.iterations_without_improvement += 1

            # Update selectors with reward signal
            if self.destroy_selector.__class__.__name__ == "DQNSelector":
                # DQN gets next state for learning
                next_state = {**state, "current_objective": self.current_objective, "best_objective": self.best_objective}
                self.destroy_selector.update(None, score, next_state)
                # Restore config
                self.config.min_removal_percentage, self.config.max_removal_percentage = old_min_sev, old_max_sev
            else:
                self.destroy_selector.update(destroy_op, score)
                self.repair_selector.update(repair_op, score)

            # Update temperature (geometric cooling + DQN multiplier)
            self.temperature *= self.config.cooling_rate
            if self.destroy_selector.__class__.__name__ == "DQNSelector":
                self.temperature *= temp_mult

            # Logging
            if (iteration + 1) % self.config.log_interval == 0:
                self._log_progress(iteration + 1)

            # Track statistics
            self._record_statistics(
                iteration, destroy_op, repair_op,
                new_objective < self.current_objective, accepted
            )

        # Final log
        # elapsed = time.time() - self.start_time
        # print(f"\nALNS completed: Best objective = {self.best_objective:.2f} in {elapsed:.1f}s")
        return self.best_solution

    def _accept_solution(
        self,
        new_objective: float,
        delta: float
    ) -> tuple[bool, float]:
        """
        Decide whether to accept new solution using Simulated Annealing.

        Returns: (accepted: bool, score: float)
        """
        # Always accept improvement
        if delta < 0:
            return True, self.config.score_new_best if new_objective < self.best_objective else self.config.score_accepted

        # Non-improvement: Metropolis criterion
        acceptance_prob = np.exp(-delta / self.temperature)
        if random.random() < acceptance_prob:
            return True, self.config.score_accepted
        else:
            return False, 0.0

    def _should_stop(self, iteration: int) -> bool:
        """Check if stopping criteria are met."""
        # Max iterations
        if iteration >= self.config.max_iterations:
            return True

        # Max time
        elapsed = time.time() - self.start_time
        if elapsed >= self.config.max_time_seconds:
            return True

        # Max iterations without improvement
        if self.iterations_without_improvement >= self.config.max_iterations_without_improvement:
            return True

        return False

    def _log_progress(self, iteration: int):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        print(f"Iteration {iteration}: "
              f"Best={self.best_objective:.2f}, "
              f"Current={self.current_objective:.2f}, "
              f"Temp={self.temperature:.2f}, "
              f"Time={elapsed:.1f}s")

    def _record_statistics(
        self,
        iteration: int,
        destroy_op: DestroyOperator,
        repair_op: RepairOperator,
        improvement: bool,
        accepted: bool
    ):
        """Record iteration statistics."""
        stat = ALNSStatistics(
            iteration=iteration,
            best_objective=self.best_objective,
            current_objective=self.current_objective,
            temperature=self.temperature,
            destroy_operator_used=destroy_op.name,
            repair_operator_used=repair_op.name,
            improvement=improvement,
            accepted=accepted,
            time_elapsed=time.time() - self.start_time,
            destroy_selector_stats=self.destroy_selector.get_statistics(),
            repair_selector_stats=self.repair_selector.get_statistics()
        )
        self.statistics.append(stat)

def run_alns_small_instance():
    print("="*80)
    print("Testing ALNS on small instance")
    print("="*80)

    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')
    # competition_instance = SCFPDPInstance('1000/competition/instance61_nreq1000_nveh20_gamma879.txt')
    # competition_instance = SCFPDPInstance('2000/competition/instance61_nreq2000_nveh40_gamma1829.txt')

    instance = test_instance
    print(f"Loaded instance: n={instance.n}, K={instance.n_K}, C={instance.C}, gamma={instance.gamma}")

    # Create initial solution using construction heuristic
    print("\nCreating initial solution with FlexiblePickupAndDropoff heuristic...")
    initial_solution = SCFPDPSolution(instance)
    constructor = FlexiblePickupAndDropoffConstructionHeuristic(initial_solution)
    constructor.construct()
    initial_obj = initial_solution.calc_objective()
    print(f"Initial solution objective: {initial_obj:.2f}")
    print(f"Served requests: {len(initial_solution.get_all_served_requests())}/{instance.gamma}")

    # Configure ALNS
    # Option 1: Use tuned parameters (automatically loads best config for instance size)
    try:
        config = ALNSConfig.from_tuned_params(
            instance_size=instance.n,
            max_time_seconds=300.0,  # Override time limit
            log_interval=100
        )
    except FileNotFoundError as e:
        print(f"\n[ALNS Config] {e}")
        print(f"[ALNS Config] Using default parameters instead\n")
        # Option 2: Use default/manual parameters
        config = ALNSConfig(
            max_iterations=1000,
            max_time_seconds=300.0,
            max_iterations_without_improvement=100,
            weight_update_period=50,
            reaction_factor=0.1,
            min_removal_percentage=0.30,
            max_removal_percentage=0.50,
            initial_temperature=50.0,
            cooling_rate=0.995,
            score_new_best=10.0,
            score_accepted=1.0,
            log_interval=100,
        )

    # Run ALNS (auto-instantiates all operators from registry)
    print("\nRunning ALNS...")
    print("-"*80)
    alns = ALNS(initial_solution, config=config)

    print(f"\nALNS Configuration:")
    print(f"  Destroy operators: {[op.name for op in alns.destroy_operators]}")
    print(f"  Repair operators: {[op.name for op in alns.repair_operators]}")
    print(f"  Total combinations: {len(alns.destroy_operators)} × {len(alns.repair_operators)} = {len(alns.destroy_operators) * len(alns.repair_operators)}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Max time: {config.max_time_seconds}s")
    print("-"*80)

    best_solution = alns.run()

    # Print results
    print("-"*80)
    print("\nResults:")
    print(f"  Initial objective: {initial_obj:.2f}")
    print(f"  Final objective: {best_solution.calc_objective():.2f}")
    print(f"  Improvement: {initial_obj - best_solution.calc_objective():.2f} ({100*(initial_obj - best_solution.calc_objective())/initial_obj:.2f}%)")
    print(f"  Total iterations: {len(alns.statistics)}")

    # Print final selector statistics
    print(f"\nFinal operator selector statistics:")
    print(f"  Destroy selector: {type(alns.destroy_selector).__name__}")
    destroy_stats = alns.destroy_selector.get_statistics()
    for key, value in destroy_stats.items():
        if isinstance(value, dict):
            print(f"    {key}:")
            for name, val in value.items():
                print(f"      {name}: {val:.3f}" if isinstance(val, float) else f"      {name}: {val}")
        else:
            print(f"    {key}: {value:.3f}" if isinstance(value, float) else f"    {key}: {value}")

    print(f"  Repair selector: {type(alns.repair_selector).__name__}")
    repair_stats = alns.repair_selector.get_statistics()
    for key, value in repair_stats.items():
        if isinstance(value, dict):
            print(f"    {key}:")
            for name, val in value.items():
                print(f"      {name}: {val:.3f}" if isinstance(val, float) else f"      {name}: {val}")
        else:
            print(f"    {key}: {value:.3f}" if isinstance(value, float) else f"    {key}: {value}")

    # Verify solution
    try:
        best_solution.check()
        print("\n✓ Solution is valid!")
    except ValueError as e:
        print(f"\n✗ Solution is invalid: {e}")
        return False


if __name__ == '__main__':
    success = run_alns_small_instance()
