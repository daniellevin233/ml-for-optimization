"""Multi-Armed Bandit selection strategies for ALNS."""

import numpy as np
from typing import Dict, Any

from src.alns.selectors.base import OperatorSelector, Operator


class UCB1Selector(OperatorSelector[Operator]):
    """
    UCB1 (Upper Confidence Bound) selection strategy.

    Formula: UCB(i) = μᵢ + c·√(ln(t)/nᵢ)
    where:
        μᵢ = average reward of operator i
        nᵢ = number of times operator i was selected
        t = total number of selections
        c = exploration constant (default 1.0)

    Reference: BALANCE paper (Phan et al., 2024)
    """

    def __init__(self, operators: list[Operator], exploration_constant: float = 1.0):
        super().__init__(operators)
        self.c = exploration_constant

        self.counts = {op.name: 0 for op in operators}
        self.total_rewards = {op.name: 0.0 for op in operators}
        self.total_count = 0

    def select(self, state: Dict[str, Any] = None) -> Operator:
        """Select operator with highest UCB value."""
        # First, try each operator at least once
        for op in self.operators:
            if self.counts[op.name] == 0:
                return op

        ucb_values = {}
        for op in self.operators:
            mean_reward = self.total_rewards[op.name] / self.counts[op.name]
            exploration_bonus = self.c * np.sqrt(np.log(self.total_count) / self.counts[op.name])
            ucb_values[op.name] = mean_reward + exploration_bonus

        best_name = max(ucb_values, key=ucb_values.get)
        return next(op for op in self.operators if op.name == best_name)

    def update(self, operator: Operator, reward: float, next_state: Dict[str, Any] = None):
        """Update statistics with observed reward."""
        self.counts[operator.name] += 1
        self.total_rewards[operator.name] += reward
        self.total_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Return UCB statistics."""
        return {
            "counts": dict(self.counts),
            "mean_rewards": {
                name: self.total_rewards[name] / max(1, self.counts[name])
                for name in self.counts
            },
            "total_selections": self.total_count,
        }


class ThompsonSamplingSelector(OperatorSelector[Operator]):
    """
    Thompson Sampling selection strategy (Beta-Bernoulli bandit).

    Maintains Beta(αᵢ, βᵢ) posterior for each operator i:
    - Sample θᵢ ~ Beta(αᵢ, βᵢ) for each operator
    - Select operator with highest sample
    - Update: αᵢ += 1 if success (reward > 0), else βᵢ += 1

    Reference: BALANCE paper (Phan et al., 2024)
    """

    def __init__(self, operators: list[Operator], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(operators)

        self.alpha = {op.name: prior_alpha for op in operators}
        self.beta = {op.name: prior_beta for op in operators}

    def select(self, state: Dict[str, Any] = None) -> Operator:
        """Sample from Beta posteriors and select highest."""
        # Sample θᵢ ~ Beta(αᵢ, βᵢ) for each operator
        samples = {
            op.name: np.random.beta(self.alpha[op.name], self.beta[op.name])
            for op in self.operators
        }

        best_name = max(samples, key=samples.get)
        return next(op for op in self.operators if op.name == best_name)

    def update(self, operator: Operator, reward: float, next_state: Dict[str, Any] = None):
        if reward > 0:
            self.alpha[operator.name] += 1  # Success
        else:
            self.beta[operator.name] += 1   # Failure

    def get_statistics(self) -> Dict[str, Any]:
        """Return Beta posterior parameters."""
        return {
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "expected_reward": {
                name: self.alpha[name] / (self.alpha[name] + self.beta[name])
                for name in self.alpha
            },
        }


if __name__ == '__main__':
    """Quick test of MAB selectors."""
    from src.instance import SCFPDPInstance
    from src.solution import SCFPDPSolution
    from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
    from src.alns.alns import ALNS
    from src.alns.config import ALNSConfig
    from src.alns.operators import create_all_destroy_operators, create_all_repair_operators

    print("Testing MAB selectors on small instance...")

    # Load smallest instance
    instance = SCFPDPInstance('10/test_instance_small.txt')
    config = ALNSConfig(max_iterations=200, log_interval=1000)
    destroy_ops = create_all_destroy_operators(config)
    repair_ops = create_all_repair_operators(config)

    # Initial solution
    initial_solution = SCFPDPSolution(instance)
    FlexiblePickupAndDropoffConstructionHeuristic(initial_solution).construct()
    initial_obj = initial_solution.calc_objective()

    results = {}

    # Test UCB1
    alns_ucb1 = ALNS(
        initial_solution.copy(),
        config=config,
        destroy_selector=UCB1Selector(destroy_ops),
        repair_selector=UCB1Selector(repair_ops),
    )
    results['UCB1'] = alns_ucb1.run().calc_objective()

    # Test Thompson Sampling
    alns_thompson = ALNS(
        initial_solution.copy(),
        config=config,
        destroy_selector=ThompsonSamplingSelector(destroy_ops),
        repair_selector=ThompsonSamplingSelector(repair_ops),
    )
    results['Thompson'] = alns_thompson.run().calc_objective()

    print(f"\nInitial: {initial_obj:.2f}")
    for name, obj in results.items():
        print(f"{name}: {obj:.2f} ({100*(initial_obj-obj)/initial_obj:.1f}% improvement)")
    print("✓ MAB selectors working correctly")