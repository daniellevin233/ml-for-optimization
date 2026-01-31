"""Base classes for operator selection strategies in ALNS."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any

from src.alns.operators import DestroyOperator, RepairOperator


Operator = TypeVar('Operator', DestroyOperator, RepairOperator)


class OperatorSelector(ABC, Generic[Operator]):
    """
    Abstract base class for operator selection strategies.

    This interface allows different selection strategies (adaptive weights, MAB, RL)
    to be used interchangeably in the ALNS algorithm.
    """

    def __init__(self, operators: list[Operator]):
        self.operators = operators
        self.operator_names = [op.name for op in operators]

    @abstractmethod
    def select(self, state: Dict[str, Any] = None) -> Operator:
        """
        Select an operator to use.

        Args:
            state: Optional state information (for RL-based selectors).
                   For stateless selectors (MAB, adaptive weights), this is ignored.

        Returns:
            Selected operator
        """
        pass

    @abstractmethod
    def update(self, operator: Operator, reward: float, next_state: Dict[str, Any] = None):
        """
        Update selector based on operator performance.

        Args:
            operator: The operator that was used
            reward: Reward signal (e.g., 10.0 for new best, 1.0 for improvement, 0.0 otherwise)
            next_state: Optional next state (for RL selectors)
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return selection statistics for analysis and debugging.

        Returns:
            Dictionary with statistics (e.g., weights, counts, etc.)
        """
        return {}