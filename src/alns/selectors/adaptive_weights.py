"""Adaptive weight selection strategy (ALNS baseline)."""

import random
from typing import Dict, Any

from src.alns.selectors.base import OperatorSelector, Operator


class AdaptiveWeightSelector(OperatorSelector[Operator]):
    """
    Adaptive weight selection using roulette wheel (baseline ALNS approach).

    This selector implements the standard ALNS adaptive weight mechanism:
    - Maintains weights for each operator
    - Selects operators probabilistically (roulette wheel)
    - Updates weights periodically based on accumulated scores
    - Uses reaction factor to control adaptation speed

    Weight update formula:
        ρ_i ← ρ_i·(1-γ) + γ·(s_i/a_i)

    Where:
        ρ_i: weight of operator i
        γ: reaction factor (how quickly weights adapt)
        s_i: total score of operator i in period
        a_i: number of applications of operator i in period
    """

    def __init__(
        self,
        operators: list[Operator],
        reaction_factor: float = 0.1,
        update_period: int = 100,
        score_new_best: float = 10.0,
        score_accepted: float = 1.0,
    ):
        super().__init__(operators)
        self.reaction_factor = reaction_factor
        self.update_period = update_period
        self.score_new_best = score_new_best
        self.score_accepted = score_accepted

        self.weights = {op.name: 1.0 for op in operators}

        self.applications = {op.name: 0 for op in operators}
        self.scores = {op.name: 0.0 for op in operators}
        self.iteration = 0

    def select(self, state: Dict[str, Any] = None) -> Operator:
        # Normalize weights to probabilities
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            # Fallback to uniform if all weights are zero
            return random.choice(self.operators)

        probabilities = [self.weights[op.name] / total_weight for op in self.operators]
        selected = random.choices(self.operators, weights=probabilities)[0]
        return selected

    def update(self, operator: Operator, reward: float, next_state: Dict[str, Any] = None):
        # Accumulate statistics
        self.applications[operator.name] += 1
        self.scores[operator.name] += reward
        self.iteration += 1

        # Update weights periodically
        if self.iteration % self.update_period == 0:
            self._update_weights()

    def _update_weights(self):
        """
        Apply adaptive weight update formula to all operators.

        Formula: ρ_i ← ρ_i·(1-γ) + γ·(s_i/a_i)
        """
        gamma = self.reaction_factor

        for op_name in self.weights:
            if self.applications[op_name] > 0:
                success_rate = self.scores[op_name] / self.applications[op_name]
                self.weights[op_name] = (
                    self.weights[op_name] * (1 - gamma) +
                    gamma * success_rate
                )

            # Reset counters for next period
            self.applications[op_name] = 0
            self.scores[op_name] = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "iteration": self.iteration,
        }