from abc import ABC, abstractmethod
import random

import numpy as np
from line_profiler import profile
from sklearn.cluster import KMeans

from src.instance import SCFPDPInstance
from src.solution import SCFPDPSolution, Route


class ConstructionHeuristic(ABC):

    def __init__(self, initial_solution: SCFPDPSolution) -> None:
        self.instance: SCFPDPInstance = initial_solution.inst
        self.solution: SCFPDPSolution = initial_solution
        self.initial_served_requests: set[int] = initial_solution.get_all_served_requests()

    @abstractmethod
    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_position: int) -> int | None:
        raise NotImplementedError()

    @staticmethod
    def select_insertion_position(route: Route) -> int:
        """Insert in the end, just before the end depot"""
        return len(route)

    def _find_closest_pickup_location(self, from_location_idx: int, excluded_requests: set[int]) -> int | None:
        """
        Find the closest pickup location to from_location_idx out of the remaining ones
        """
        if len(excluded_requests) >= self.instance.n:
            return None

        # 1:n+1 to skip over depot
        relevant_pickup_locations = self.instance.distance_matrix[from_location_idx, 1:self.instance.n + 1].copy()
        # mark excluded pickup candidates as inf
        relevant_pickup_locations[list(excluded_requests)] = np.inf

        if np.all(relevant_pickup_locations == np.inf):
            return None

        # randomized tie-breaking can be applied here, argmin will return the index of the first minimum value
        closest_request = np.argmin(relevant_pickup_locations)
        return int(closest_request)

    def select_dropoff_distance(self, route: Route, request_id: int = None, pickup_position: int = None) -> int:
        """By default drop off directly after pickup"""
        return 1

    def select_next_request_and_position(self, route: Route, excluded_requests: set[int]) -> tuple[int, int] | None:
        """
        Select the next request and its pickup insertion position.

        Returns: (request_id, pickup_position) or None if no feasible request exists.
        Can be overridden by subclasses to jointly optimize request and position selection.
        """
        insertion_position = self.select_insertion_position(route)
        request_id = self._select_next_request(route, excluded_requests, insertion_position)
        if request_id is None:
            return None
        return (request_id, insertion_position)

    def construct(self) -> None:
        served_requests: set[int] = self.initial_served_requests.copy()
        full_routes: list[Route] = []

        while len(served_requests) < self.instance.gamma and len(full_routes) < len(self.solution.routes):
            for current_route in self.solution.routes:  # go over every route
                if current_route in full_routes:
                    continue

                result = self.select_next_request_and_position(current_route, served_requests)
                # If there is no feasible next request for this route, move on to the next route.
                if result is None:
                    full_routes.append(current_route)
                    continue

                next_request, next_insert_at = result
                dropoff_distance = self.select_dropoff_distance(current_route, next_request, next_insert_at)

                # In this greedy construction - append pickup and dropoff together as the last two stops before depot
                current_route.serve_request(next_request, next_insert_at, dropoff_distance)
                served_requests.add(next_request)

                if len(served_requests) >= self.instance.gamma:
                    break

        # validate to make sure the requirements were fulfilled
        try:
            self.solution.check()
        except ValueError:
            raise ValueError(f"Couldn't construct a satisfying solution. Best solution found: \n\n {self.solution}")


class GreedyConstructionHeuristic(ConstructionHeuristic):
    def _find_closest_fitting_pickup_location(self, route: Route, excluded_requests: set[int], from_location_idx: int) -> int | None:
        """
        Find the closest fitting request from from_location_idx out of the remaining ones
        """
        closest_request = self._find_closest_pickup_location(from_location_idx, excluded_requests)
        non_fitting_requests = set()

        while closest_request is not None:
            if route.can_take_request(closest_request, from_location_idx):
                return closest_request
            non_fitting_requests.add(closest_request)
            closest_request = self._find_closest_pickup_location(
                from_location_idx, excluded_requests | non_fitting_requests
            )

        return None

    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_position: int) -> int | None:
        """Greedily select the closest to the insertion position fitting pickup"""
        return self._find_closest_fitting_pickup_location(route, excluded_requests, insertion_position)


class RandomizedConstructionHeuristic(GreedyConstructionHeuristic):
    def __init__(self, initial_solution: SCFPDPSolution, top_random_pickups_to_consider: int = 10):
        super().__init__(initial_solution)
        self.top_random_pickups_to_consider = top_random_pickups_to_consider

    def get_rcl_of_next_requests(self, route: Route, excluded_requests: set[int], insertion_location_idx: int) -> list[int] | None:
        """Select request using RCL of length top_random_pickups_to_consider to insert at the insertion location."""
        capacity_at_insertion = route.get_capacity_at_position(insertion_location_idx)
        remaining_capacity = self.instance.C - capacity_at_insertion

        distances_to_pickups = self.instance.distance_matrix[insertion_location_idx, 1:self.instance.n + 1]

        candidates = []
        for request_id in range(self.instance.n):
            # request hasn't been served yet and it can be taken without exceeding capacity
            if request_id not in excluded_requests and self.instance.demands[request_id] <= remaining_capacity:
                candidates.append((distances_to_pickups[request_id], request_id))

        if not candidates:
            return None

        candidates.sort()
        rcl = candidates[:min(self.top_random_pickups_to_consider, len(candidates))]
        return [c[1] for c in rcl]

    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_location_idx: int) -> int | None:
        """Select request using RCL of length 10 to insert at the insertion location."""
        rcl = self.get_rcl_of_next_requests(route, excluded_requests, insertion_location_idx)
        return random.choice(rcl)


class FlexiblePickupAndDropoffConstructionHeuristic(GreedyConstructionHeuristic):
    """
    Enhanced greedy construction that finds optimal pickup AND dropoff positions.

    For each candidate request, evaluates ALL valid pickup positions in the route,
    and for each pickup position, evaluates ALL valid dropoff positions.
    Selects the request with minimum total insertion cost.
    """

    @profile
    def _find_best_insertion_cost(self, route: Route, request_id: int) -> tuple[int, int, int] | None:
        """
        Find the best pickup and dropoff positions for a request.

        Returns: (total_cost, best_pickup_pos, best_dropoff_dist) or None if infeasible.
        """
        best_cost = float('inf')
        best_pickup_pos = None
        best_dropoff_dist = None

        # Try all pickup positions (0 to len(route) inclusive)
        for pickup_pos in range(len(route) + 1):
            # Try all valid dropoff positions (after pickup)
            for dropoff_dist in range(1, len(route) - pickup_pos + 2):
                dropoff_pos = pickup_pos + dropoff_dist

                # Check capacity feasibility for this pickup-dropoff combination
                if not route.can_take_request(request_id, pickup_pos, dropoff_pos):
                    continue

                # Calculate delta in the cost when serving the request in suggested positions
                total_cost = route.calculate_request_serving_delta(request_id, pickup_pos, dropoff_pos)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_pickup_pos = pickup_pos
                    best_dropoff_dist = dropoff_dist

        if best_pickup_pos is None:
            return None
        return (best_cost, best_pickup_pos, best_dropoff_dist)

    def select_next_request_and_position(self, route: Route, excluded_requests: set[int]) -> tuple[int, int] | None:
        """
        Select request with minimum total insertion cost across all positions.

        Also stores the best dropoff distance for use in select_dropoff_distance().
        """
        best_request = None
        best_total_cost = float('inf')
        best_pickup_pos = None
        best_dropoff_dist = None

        for request_id in range(self.instance.n):
            if request_id in excluded_requests:
                continue

            result = self._find_best_insertion_cost(route, request_id)
            if result is None:
                continue

            cost, pickup_pos, dropoff_dist = result
            if cost < best_total_cost:
                best_total_cost = cost
                best_request = request_id
                best_pickup_pos = pickup_pos
                best_dropoff_dist = dropoff_dist

        if best_request is None:
            return None

        # Store for select_dropoff_distance to use
        self._cached_dropoff_distance = best_dropoff_dist
        return (best_request, best_pickup_pos)

    def select_dropoff_distance(self, route: Route, request_id: int = None, pickup_position: int = None) -> int:
        """Return the cached best dropoff distance computed during request selection."""
        return getattr(self, '_cached_dropoff_distance', 1)


class ClusterBasedConstructionHeuristic(GreedyConstructionHeuristic):
    """
    Cluster-based construction that pre-assigns requests to vehicles.

    Uses KMeans clustering on request centroids (midpoint of pickup-dropoff)
    to group geographically close requests. Each cluster is assigned to a vehicle,
    and requests are served within their assigned cluster.
    """

    def __init__(self, initial_solution: SCFPDPSolution):
        super().__init__(initial_solution)
        self.vehicle_requests: dict[int, set[int]] = {}  # vehicle_idx -> set of request_ids
        self._compute_clusters()

    def _compute_request_centroids(self) -> np.ndarray:
        """Compute centroid (midpoint) for each request."""
        centroids = []
        for request_id in range(self.instance.n):
            pickup_loc = self.instance.pickup_locations[request_id]
            dropoff_loc = self.instance.dropoff_locations[request_id]
            centroid_x = (pickup_loc.x + dropoff_loc.x) / 2
            centroid_y = (pickup_loc.y + dropoff_loc.y) / 2
            centroids.append([centroid_x, centroid_y])
        return np.array(centroids)

    def _compute_clusters(self) -> None:
        """Cluster requests and assign to vehicles."""
        n_vehicles = self.instance.n_K
        centroids = self._compute_request_centroids()

        # Initialize empty sets for each vehicle
        for vehicle_idx in range(n_vehicles):
            self.vehicle_requests[vehicle_idx] = set()

        # Use KMeans to cluster requests
        kmeans = KMeans(n_clusters=n_vehicles, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(centroids)

        # Assign each request to its cluster (vehicle)
        for request_id, cluster_id in enumerate(cluster_labels):
            self.vehicle_requests[cluster_id].add(request_id)

    def _select_next_request(self, route: Route, excluded_requests: set[int], insertion_position: int) -> int | None:
        """
        Select the closest fitting request that belongs to this vehicle's cluster.
        """
        vehicle_idx = self.solution.routes.index(route)

        # Get requests assigned to this vehicle that haven't been served
        cluster_requests = self.vehicle_requests[vehicle_idx] - excluded_requests

        if not cluster_requests:
            # If no requests left in cluster, treat the route of this vehicle as finalized
            return None

        # Find closest fitting request within the cluster
        excluded_non_cluster = excluded_requests | (set(range(self.instance.n)) - cluster_requests)
        return self._find_closest_fitting_pickup_location(route, excluded_non_cluster, insertion_position)


class HybridFlexibleClusteredConstructionHeuristic(FlexiblePickupAndDropoffConstructionHeuristic, ClusterBasedConstructionHeuristic):
    def select_next_request_and_position(self, route: Route, excluded_requests: set[int]) -> tuple[int, int] | None:
        """
        Select request with minimum total insertion cost, but only from this vehicle's cluster.
        """
        vehicle_idx = self.solution.routes.index(route)

        # Get requests assigned to this vehicle that haven't been served
        cluster_requests = self.vehicle_requests[vehicle_idx] - excluded_requests

        if not cluster_requests:
            # If no requests left in cluster, route is finalized
            return None

        best_request = None
        best_total_cost = float('inf')
        best_pickup_pos = None
        best_dropoff_dist = None

        # Only evaluate requests from this vehicle's cluster
        for request_id in cluster_requests:
            result = self._find_best_insertion_cost(route, request_id)
            if result is None:
                continue

            cost, pickup_pos, dropoff_dist = result
            if cost < best_total_cost:
                best_total_cost = cost
                best_request = request_id
                best_pickup_pos = pickup_pos
                best_dropoff_dist = dropoff_dist

        if best_request is None:
            return None

        # Store for select_dropoff_distance to use
        self._cached_dropoff_distance = best_dropoff_dist
        return (best_request, best_pickup_pos)


class RandomizedHybridConstructionHeuristic(HybridFlexibleClusteredConstructionHeuristic):
    """
    Randomized version of hybrid heuristic using Restricted Candidate List (RCL).

    Uses deterministic clustering (same as HybridFlexibleClusteredConstructionHeuristic)
    but randomly selects from the top-k best insertion positions instead of always
    picking the minimum cost. This produces diverse solutions for meta-heuristics
    while maintaining good quality.
    """

    def __init__(self, initial_solution: SCFPDPSolution, rcl_size: int = 10):
        super().__init__(initial_solution)
        self.rcl_size = rcl_size

    def select_next_request_and_position(self, route: Route, excluded_requests: set[int]) -> tuple[int, int] | None:
        """
        Select request randomly from RCL of top-k lowest insertion costs within cluster.
        """
        vehicle_idx = self.solution.routes.index(route)

        # Get requests assigned to this vehicle that haven't been served
        cluster_requests = self.vehicle_requests[vehicle_idx] - excluded_requests

        if not cluster_requests:
            # If no requests left in cluster, route is finalized
            return None

        # Evaluate all requests in cluster, collect (cost, request_id, pickup_pos, dropoff_dist)
        candidates = []
        for request_id in cluster_requests:
            result = self._find_best_insertion_cost(route, request_id)
            if result is None:
                continue

            cost, pickup_pos, dropoff_dist = result
            candidates.append((cost, request_id, pickup_pos, dropoff_dist))

        if not candidates:
            return None

        # Sort by cost and take top rcl_size
        candidates.sort()
        rcl = candidates[:min(self.rcl_size, len(candidates))]

        # Randomly select from RCL
        selected = random.choice(rcl)
        _, request_id, pickup_pos, dropoff_dist = selected

        # Store for select_dropoff_distance to use
        self._cached_dropoff_distance = dropoff_dist
        return (request_id, pickup_pos)


if __name__ == '__main__':
    test_instance = SCFPDPInstance('10/test_instance_small.txt')
    competition_instance = SCFPDPInstance('2000/competition/instance61_nreq2000_nveh40_gamma1829.txt')
    # competition_instance = SCFPDPInstance('1000/competition/instance61_nreq1000_nveh20_gamma879.txt')
    # competition_instance = SCFPDPInstance('100/competition/instance61_nreq100_nveh2_gamma91.txt')

    instance = competition_instance

    # _initial_solution = SCFPDPSolution(inst=instance)
    # GreedyConstructionHeuristic(_initial_solution).construct()
    # print("Greedy solution: ")
    # print(_initial_solution)
    #
    # _initial_solution_1 = SCFPDPSolution(inst=instance)
    # RandomizedConstructionHeuristic(_initial_solution_1).construct()
    # print("\n\nRandomized solution: ")
    # print(_initial_solution_1)
    #
    # _initial_solution_2 = SCFPDPSolution(inst=instance)
    # RandomizedHybridConstructionHeuristic(_initial_solution_2).construct()
    # print("\n\nRandomized solution 2: ")
    # print(_initial_solution_2)

    _initial_solution_3 = SCFPDPSolution(inst=instance)
    FlexiblePickupAndDropoffConstructionHeuristic(_initial_solution_3).construct()
    print("\n\nGreedy solution 2: ")
    print(_initial_solution_3)
    _initial_solution_3.write_to_file("greedy_construction")
    #
    # best_solution = _initial_solution if _initial_solution.calc_objective() < _initial_solution_1.calc_objective() else _initial_solution_1
    # best_algo = "greedy_construction" if _initial_solution.calc_objective() < _initial_solution_1.calc_objective() else "randomized_construction"
    #
    # print(f"Best solution ({best_algo}): ")
    # print(best_solution)

    # best_solution.write_to_file(algorithm=best_algo)
