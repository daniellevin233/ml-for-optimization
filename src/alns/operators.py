from abc import ABC, abstractmethod
from copy import deepcopy
import random
from enum import Enum

from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
from src.solution import SCFPDPSolution
from src.alns.config import ALNSConfig


class DestroyedSCFPDPSolution:
    """Wrapper for SCFPDPSolution with removed_requests tracking for ALNS repair operators."""

    def __init__(self, original_solution: SCFPDPSolution):
        self.solution = deepcopy(original_solution)
        self.removed_requests: set[int] = set()


class DestroyOperator(ABC):
    """Base class for destroy operators."""

    def __init__(self, name: str, config: ALNSConfig):
        self.name = name
        self.config = config

    @abstractmethod
    def apply(self, solution: SCFPDPSolution) -> DestroyedSCFPDPSolution:
        """
        Remove requests from solution.

        Returns a DestroyedSCFPDPSolution with removed_requests tracked.
        """
        pass

    def _calculate_removal_count(self, solution: SCFPDPSolution) -> int:
        """Calculate how many requests to remove based on config percentages."""
        n_served = len(solution.get_all_served_requests())
        min_removal = max(1, int(n_served * self.config.min_removal_percentage))
        max_removal = max(min_removal, int(n_served * self.config.max_removal_percentage))
        return random.randint(min_removal, max_removal)


class RepairOperator(ABC):
    """Base class for repair operators."""

    def __init__(self, name: str, config: ALNSConfig):
        self.name = name
        self.config = config

    @abstractmethod
    def apply(self, destroyed_solution: DestroyedSCFPDPSolution) -> SCFPDPSolution:
        """
        Reinsert removed requests into solution.

        Returns the repaired SCFPDPSolution.
        """
        pass


# ===== DESTROY OPERATORS =====

class RandomRemovalOperator(DestroyOperator):
    """Randomly remove requests from solution."""

    def apply(self, solution: SCFPDPSolution) -> DestroyedSCFPDPSolution:
        destroyed = DestroyedSCFPDPSolution(solution)
        q = self._calculate_removal_count(solution)

        served_requests = list(solution.get_all_served_requests())
        to_remove = random.sample(served_requests, min(q, len(served_requests)))

        # Remove from routes
        for request_id in to_remove:
            for route in destroyed.solution.routes:
                if request_id in route.served_requests:
                    route.remove_request(request_id)
                    destroyed.removed_requests.add(request_id)
                    break

        return destroyed


class WorstCostRemovalOperator(DestroyOperator):
    """Remove requests that contribute most to the objective."""

    def apply(self, solution: SCFPDPSolution) -> DestroyedSCFPDPSolution:
        destroyed = DestroyedSCFPDPSolution(solution)
        q = self._calculate_removal_count(solution)

        served_requests = list(solution.get_all_served_requests())
        if not served_requests:
            return destroyed

        # Calculate cost of each request (distance saved if removed)
        request_costs = []
        for request_id in served_requests:
            # Find route containing this request
            route_with_request = None
            for route in destroyed.solution.routes:
                if request_id in route.served_requests:
                    route_with_request = route
                    break

            if route_with_request is None:
                continue

            # Calculate removal cost (negative = improvement when removed)
            pickup_idx = request_id
            dropoff_idx = request_id + destroyed.solution.inst.n

            try:
                pickup_pos = route_with_request.route.index(pickup_idx)
                dropoff_pos = route_with_request.route.index(dropoff_idx)

                # Calculate distance saved by removing
                removal_delta = 0

                # Pickup removal
                prev_pickup = None if pickup_pos == 0 else route_with_request.route[pickup_pos - 1]
                next_pickup = None if pickup_pos >= len(route_with_request.route) - 1 else route_with_request.route[pickup_pos + 1]

                old_dist_pickup = (route_with_request.get_distance_between(prev_pickup, pickup_idx) +
                                  route_with_request.get_distance_between(pickup_idx, next_pickup))
                new_dist_pickup = route_with_request.get_distance_between(prev_pickup, next_pickup)
                removal_delta += (new_dist_pickup - old_dist_pickup)

                # Dropoff removal (accounting for pickup already removed)
                adjusted_dropoff_pos = dropoff_pos - 1
                prev_dropoff = None if adjusted_dropoff_pos == 0 else route_with_request.route[adjusted_dropoff_pos - 1]
                next_dropoff = None if adjusted_dropoff_pos >= len(route_with_request.route) - 2 else route_with_request.route[adjusted_dropoff_pos + 1]

                old_dist_dropoff = (route_with_request.get_distance_between(prev_dropoff, dropoff_idx) +
                                   route_with_request.get_distance_between(dropoff_idx, next_dropoff))
                new_dist_dropoff = route_with_request.get_distance_between(prev_dropoff, next_dropoff)
                removal_delta += (new_dist_dropoff - old_dist_dropoff)

                # Most negative delta = highest cost saved by removing this request
                request_costs.append((removal_delta, request_id))

            except (ValueError, IndexError):
                # Skip if request not found properly
                continue

        if not request_costs:
            return destroyed

        # Sort by removal_delta (ascending - most negative = highest cost saved)
        request_costs.sort()
        to_remove = [req_id for _, req_id in request_costs[:q]]

        # Remove requests
        for request_id in to_remove:
            for route in destroyed.solution.routes:
                if request_id in route.served_requests:
                    route.remove_request(request_id)
                    destroyed.removed_requests.add(request_id)
                    break

        return destroyed


class LongestRouteRemovalOperator(DestroyOperator):
    """Remove requests from vehicle(s) with longest distance (targets fairness)."""

    def apply(self, solution: SCFPDPSolution) -> DestroyedSCFPDPSolution:
        destroyed = DestroyedSCFPDPSolution(solution)
        q = self._calculate_removal_count(solution)

        # Sort routes by distance (descending)
        sorted_routes = sorted(
            [route for route in destroyed.solution.routes if route.served_requests],
            key=lambda route: route.distance,
            reverse=True
        )

        if not sorted_routes:
            return destroyed

        removed_count = 0

        # Remove q requests from longest route(s)
        for route in sorted_routes:
            if removed_count >= q:
                break

            for request_id in list(route.served_requests):  # Copy to avoid modification during iteration
                if removed_count >= q:
                    break
                route.remove_request(request_id)
                destroyed.removed_requests.add(request_id)
                removed_count += 1

        return destroyed


# ===== REPAIR OPERATORS =====

class GreedyRepairOperator(RepairOperator):
    """Insert removed requests using FlexiblePickupAndDropoffConstructionHeuristic."""

    def apply(self, destroyed_solution: DestroyedSCFPDPSolution) -> SCFPDPSolution:
        # The constructor automatically detects already-served requests
        # construct() will only insert the removed ones
        constructor = FlexiblePickupAndDropoffConstructionHeuristic(destroyed_solution.solution)
        constructor.construct()

        return destroyed_solution.solution


class RandomGreedyRepairOperator(RepairOperator):
    """Insert removed requests at random feasible positions."""

    def apply(self, destroyed_solution: DestroyedSCFPDPSolution) -> SCFPDPSolution:
        for request_id in destroyed_solution.removed_requests:
            # Collect all feasible insertions
            feasible_insertions = []

            for route_idx, route in enumerate(destroyed_solution.solution.routes):
                for pickup_pos in range(len(route) + 1):
                    for dropoff_dist in range(1, len(route) - pickup_pos + 2):
                        dropoff_pos = pickup_pos + dropoff_dist

                        # Check feasibility
                        if route.can_take_request(request_id, pickup_pos, dropoff_pos):
                            feasible_insertions.append((route_idx, pickup_pos, dropoff_dist))

            # Insert at random feasible position
            if feasible_insertions:
                route_idx, pickup_pos, dropoff_dist = random.choice(feasible_insertions)
                destroyed_solution.solution.routes[route_idx].serve_request(
                    request_id, pickup_pos, dropoff_dist
                )

        return destroyed_solution.solution


class ObjectiveAwareRepairOperator(RepairOperator):
    """Insert removed requests by minimizing full objective (distance + fairness penalty)."""

    @staticmethod
    def _calculate_objective_delta(solution, route, request_id, pickup_pos, dropoff_dist):
        """
        Calculate objective change if request is inserted without actually modifying solution.
        Respects the configured fairness measure (jain, max_min, or gini).
        """
        # Current state
        old_route_distance = route.distance
        old_distances = [r.distance for r in solution.routes]
        old_total_distance = sum(old_distances)

        # Calculate distance delta for this insertion
        distance_delta = route.calculate_request_serving_delta(request_id, pickup_pos, dropoff_dist)
        new_route_distance = old_route_distance + distance_delta

        # New distances after insertion
        new_distances = old_distances.copy()
        route_idx = solution.routes.index(route)
        new_distances[route_idx] = new_route_distance
        new_total_distance = old_total_distance + distance_delta

        # Calculate fairness values
        K = solution.inst.n_K
        rho = solution.inst.rho

        old_sum_squared = sum(d**2 for d in old_distances)
        new_sum_squared = sum(d**2 for d in new_distances)

        if old_sum_squared == 0 or new_sum_squared == 0:
            return distance_delta

        old_fairness = old_total_distance**2 / (K * old_sum_squared)
        new_fairness = new_total_distance**2 / (K * new_sum_squared)

        old_objective = old_total_distance + rho * (1 - old_fairness)
        new_objective = new_total_distance + rho * (1 - new_fairness)

        return new_objective - old_objective

    def apply(self, destroyed_solution: DestroyedSCFPDPSolution) -> SCFPDPSolution:
        for request_id in destroyed_solution.removed_requests:
            best_delta = float('inf')
            best_route_idx = None
            best_pickup_pos = None
            best_dropoff_dist = None

            # Evaluate all feasible insertions by objective delta
            for route_idx, route in enumerate(destroyed_solution.solution.routes):
                for pickup_pos in range(len(route) + 1):
                    for dropoff_dist in range(1, len(route) - pickup_pos + 2):
                        dropoff_pos = pickup_pos + dropoff_dist

                        # Check feasibility
                        if not route.can_take_request(request_id, pickup_pos, dropoff_pos):
                            continue

                        # Calculate objective delta (no actual modification)
                        delta = self._calculate_objective_delta(
                            destroyed_solution.solution, route, request_id, pickup_pos, dropoff_dist
                        )

                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = route_idx
                            best_pickup_pos = pickup_pos
                            best_dropoff_dist = dropoff_dist

            # Insert at best position
            if best_route_idx is not None:
                destroyed_solution.solution.routes[best_route_idx].serve_request(
                    request_id, best_pickup_pos, best_dropoff_dist
                )

        return destroyed_solution.solution


# ===== OPERATOR ENUMS AND REGISTRY =====

class DestroyOperatorType(Enum):
    """Available destroy operators."""
    RANDOM = "Random"
    WORST_COST = "WorstCost"
    LONGEST_ROUTE = "LongestRoute"


class RepairOperatorType(Enum):
    """Available repair operators."""
    GREEDY = "Greedy"
    RANDOM_GREEDY = "RandomGreedy"
    OBJECTIVE_AWARE = "ObjectiveAware"


# Registry mapping operator types to their classes
DESTROY_OPERATOR_REGISTRY = {
    DestroyOperatorType.RANDOM: RandomRemovalOperator,
    DestroyOperatorType.WORST_COST: WorstCostRemovalOperator,
    DestroyOperatorType.LONGEST_ROUTE: LongestRouteRemovalOperator,
}

REPAIR_OPERATOR_REGISTRY = {
    RepairOperatorType.GREEDY: GreedyRepairOperator,
    RepairOperatorType.RANDOM_GREEDY: RandomGreedyRepairOperator,
    RepairOperatorType.OBJECTIVE_AWARE: ObjectiveAwareRepairOperator,
}


def create_all_destroy_operators(config: ALNSConfig) -> list[DestroyOperator]:
    """Create instances of all available destroy operators."""
    return [cls(op_type.value, config) for op_type, cls in DESTROY_OPERATOR_REGISTRY.items()]


def create_all_repair_operators(config: ALNSConfig) -> list[RepairOperator]:
    """Create instances of all available repair operators."""
    return [cls(op_type.value, config) for op_type, cls in REPAIR_OPERATOR_REGISTRY.items()]