import copy
from typing import Callable

from line_profiler import profile
from pymhlib.solution import Solution

from src.instance import SCFPDPInstance


class Route:
    def __init__(self, instance: SCFPDPInstance, delta_callback: Callable[[int, int], None]) -> None:
        self.instance = instance
        self.delta_callback = delta_callback
        self.route: list[int] = []
        self.distance: int = 0
        self.served_requests: list[int] = []
        self._capacities: list[int] = [0]  # capacity at each position (len = len(route) + 1)

    def __repr__(self):
        return f"Served requests: {{{', '.join(str(i) for i in self.served_requests)}}} along the route: {["depot"] + self.route + ["depot"]} (distance={self.distance:.2f}; capacity={self.get_carried_capacity()}/{self.instance.C})"

    def __len__(self) -> int:
        return len(self.route)

    def get_distance_between(self, loc1: int | None, loc2: int | None) -> int:
        """
        Get distance between two locations. None represents depot.
        loc1 and loc2 are location indices as they appear in self.route.
        """
        if loc1 is None:
            idx1 = 0  # depot
        else:
            idx1 = loc1 + 1  # location indices are offset by 1 in distance_matrix

        if loc2 is None:
            idx2 = 0  # depot
        else:
            idx2 = loc2 + 1

        return self.instance.distance_matrix[idx1][idx2]

    def recompute_route_distance(self) -> None:
        """Full recalculation of route distance. Used for verification and initialization."""
        old_distance = self.distance

        if len(self.route) == 0:
            self.distance = 0
            self.delta_callback(self.distance, old_distance)
            return

        from_depot_distance = self.instance.distance_matrix[0][self.route[0] + 1]
        to_depot_distance = self.instance.distance_matrix[0][self.route[-1] + 1]
        route_distance = 0
        for i, source_location in enumerate(self.route[:-1]):
            target_location = self.route[i + 1]
            route_distance += self.instance.distance_matrix[source_location + 1][target_location + 1]
        self.distance = from_depot_distance + route_distance + to_depot_distance
        self.delta_callback(self.distance, old_distance)

    def calculate_insertion_delta(self, location_idx: int, at: int) -> int:
        """
        Calculate distance delta for inserting location_idx at position at.
        Must be called BEFORE the actual insertion.

        Returns the change in distance: (new_edges_distance - old_edges_distance)
        """
        # Get neighbors before and after insertion position
        prev_loc = None if at == 0 else self.route[at - 1]
        next_loc = None if at >= len(self.route) else self.route[at]

        # Old edge: prev -> next
        old_distance = self.get_distance_between(prev_loc, next_loc)

        # New edges: prev -> location_idx -> next
        new_distance = (self.get_distance_between(prev_loc, location_idx) +
                        self.get_distance_between(location_idx, next_loc))

        return new_distance - old_distance

    def calculate_request_serving_delta(self, request_id: int, pickup_pos: int, dropoff_pos: int) -> int:
        """
        Calculate combined distance delta for inserting both pickup and dropoff.

        Args:
            request_id: The request to insert
            pickup_pos: Position to insert pickup (0 to len(route))
            dropoff_pos: Position to insert dropoff AFTER pickup insertion (pickup_pos+1 to len(route)+1)

        Returns the total change in distance for inserting both locations.
        """
        pickup_loc = request_id
        dropoff_loc = request_id + self.instance.n

        # Pickup delta (standard calculation on current route)
        pickup_prev = None if pickup_pos == 0 else self.route[pickup_pos - 1]
        pickup_next = None if pickup_pos >= len(self.route) else self.route[pickup_pos]

        pickup_old_edge = self.get_distance_between(pickup_prev, pickup_next)
        pickup_new_edges = (self.get_distance_between(pickup_prev, pickup_loc) +
                           self.get_distance_between(pickup_loc, pickup_next))
        pickup_delta = pickup_new_edges - pickup_old_edge

        # Dropoff delta (accounting for pickup already inserted)
        # After pickup insertion at pickup_pos:
        # - Indices < pickup_pos: unchanged
        # - Index pickup_pos: pickup_loc
        # - Indices >= pickup_pos in original: shifted by 1

        # Map dropoff_pos (post-pickup) to neighbors
        if dropoff_pos == pickup_pos + 1:
            # Dropoff immediately after pickup
            dropoff_prev = pickup_loc
            # Next is what was at pickup_pos in original (now at pickup_pos+1)
            dropoff_next = None if pickup_pos >= len(self.route) else self.route[pickup_pos]
        else:
            # dropoff_pos > pickup_pos + 1
            # Previous is at original index (dropoff_pos - 2) because of the shift
            original_prev_idx = dropoff_pos - 2
            if original_prev_idx < pickup_pos:
                dropoff_prev = self.route[original_prev_idx] if original_prev_idx >= 0 else None
            elif original_prev_idx == pickup_pos:
                dropoff_prev = pickup_loc
            else:
                dropoff_prev = self.route[original_prev_idx - 1]

            # Next is at original index (dropoff_pos - 1) because of the shift
            original_next_idx = dropoff_pos - 1
            if original_next_idx <= pickup_pos:
                dropoff_next = self.route[original_next_idx] if original_next_idx < len(self.route) else None
            else:
                actual_idx = original_next_idx - 1
                dropoff_next = self.route[actual_idx] if actual_idx < len(self.route) else None

        dropoff_old_edge = self.get_distance_between(dropoff_prev, dropoff_next)
        dropoff_new_edges = (self.get_distance_between(dropoff_prev, dropoff_loc) +
                            self.get_distance_between(dropoff_loc, dropoff_next))
        dropoff_delta = dropoff_new_edges - dropoff_old_edge

        return pickup_delta + dropoff_delta

    def _calculate_removal_delta(self, at: int) -> int:
        """
        Calculate distance delta for removing location at position at.
        Must be called BEFORE the actual removal.

        Returns the change in distance: (new_edge_distance - old_edges_distance)
        """
        location_idx = self.route[at]

        # Get neighbors
        prev_loc = None if at == 0 else self.route[at - 1]
        next_loc = None if at >= len(self.route) - 1 else self.route[at + 1]

        # Old edges: prev -> location -> next
        old_distance = (self.get_distance_between(prev_loc, location_idx) +
                        self.get_distance_between(location_idx, next_loc))

        # New edge: prev -> next
        new_distance = self.get_distance_between(prev_loc, next_loc)

        return new_distance - old_distance

    def insert_location(self, location_idx: int, at: int) -> None:
        if location_idx in self.route:
            raise ValueError(f"Request {location_idx} is already in the route")
        if location_idx in self.served_requests:
            raise ValueError(f"Request {location_idx} is already served")

        self.route.insert(at, location_idx)
        # serve the request if it's a pickup index
        if location_idx < self.instance.n:
            self.served_requests.append(location_idx)

    def serve_request(self, request_id: int, pickup_at: int, dropoff_distance_from_pickup: int) -> None:
        if pickup_at > len(self.route):
            raise ValueError(f"Pickup insertion index is out of range: {pickup_at}; route length: {len(self.route)}")
        if dropoff_distance_from_pickup <= 0:
            raise ValueError("Dropoff must be at least one position to the right from the pickup")
        dropoff_at = pickup_at + dropoff_distance_from_pickup
        if dropoff_at > len(self.route) + 1:
            raise ValueError(f"Dropoff insertion index is out of range: {dropoff_at}; route length: {len(self.route)}")

        demand = self.instance.demands[request_id]

        # TSP-like delta evaluation: calculate only affected edges
        old_distance = self.distance
        pickup_delta = self.calculate_insertion_delta(request_id, pickup_at)

        # Insert pickup (modifies route)
        self.insert_location(request_id, pickup_at)

        # Calculate dropoff delta after pickup insertion
        dropoff_delta = self.calculate_insertion_delta(request_id + self.instance.n, dropoff_at)

        # Insert dropoff
        self.insert_location(request_id + self.instance.n, dropoff_at)
        self._update_capacities_for_serve(pickup_at, dropoff_at, demand)
        self.check()

        # Update distance incrementally
        self.distance = old_distance + pickup_delta + dropoff_delta
        self.delta_callback(self.distance, old_distance)

    @profile
    def can_take_request(self, request_id: int, at_position: int, dropoff_position: int | None = None) -> bool:
        """
        Check if a request can be feasibly inserted at the given positions.

        Args:
            request_id: The request to insert
            at_position: Pickup insertion position
            dropoff_position: Dropoff insertion position (optional). If provided, checks capacity
                              at all positions between pickup and dropoff. If None, only checks
                              at the pickup position
        """
        # Cache attribute lookups to avoid repeated access in hot loop
        demand = self.instance.demands[request_id]
        max_cap = self.instance.C
        capacities = self._capacities

        if dropoff_position is None:
            return capacities[at_position] + demand <= max_cap

        # Check capacity at all positions from pickup to dropoff (inlined for performance)
        for pos in range(at_position, dropoff_position):
            if capacities[pos] + demand > max_cap:
                return False
        return True

    def get_carried_capacity(self) -> int:
        return self._capacities[-1] if self._capacities else 0

    @profile
    def _recompute_capacities(self) -> None:
        """Recompute the cached capacity array. O(n) operation."""
        self._capacities = [0]
        capacity = 0
        for location_idx in self.route:
            if location_idx < self.instance.n:  # pickup
                capacity += self.instance.demands[location_idx]
            else:  # dropoff
                capacity -= self.instance.demands[location_idx - self.instance.n]
            self._capacities.append(capacity)

    def _update_capacities_for_serve(self, pickup_at: int, dropoff_at: int, demand: int) -> None:
        """
        Delta update capacities after serving a request.
        Called AFTER route has been modified (both insertions done).
        pickup_at and dropoff_at are positions in the NEW route.
        """
        # Insert capacity after pickup (old capacity at that position + demand)
        cap_after_pickup = self._capacities[pickup_at] + demand
        self._capacities.insert(pickup_at + 1, cap_after_pickup)

        # Add demand to all capacities between pickup and dropoff (they now carry the item)
        # These are positions pickup_at+2 to dropoff_at in the array (after first insert)
        for i in range(pickup_at + 2, dropoff_at + 1):
            self._capacities[i] += demand

        # Insert capacity after dropoff (subtract demand since we dropped off)
        cap_after_dropoff = self._capacities[dropoff_at] - demand
        self._capacities.insert(dropoff_at + 1, cap_after_dropoff)

    def _update_capacities_for_remove(self, pickup_pos: int, dropoff_pos: int, demand: int) -> None:
        """
        Delta update capacities after removing a request.
        Called BEFORE route is modified. pickup_pos < dropoff_pos.
        """
        # Remove capacity after dropoff first (higher index)
        self._capacities.pop(dropoff_pos + 1)

        # Subtract demand from positions between pickup and dropoff
        for i in range(pickup_pos + 1, dropoff_pos):
            self._capacities[i] -= demand

        # Remove capacity after pickup
        self._capacities.pop(pickup_pos + 1)

    def get_capacity_at_position(self, position: int) -> int:
        return self._capacities[position]

    def _check_capacity_constraint_at_position(self, position: int) -> int:
        capacity = 0
        for location_idx in self.route[:position]:
            if location_idx < self.instance.n:  # pickup
                capacity += self.instance.demands[location_idx]
                if capacity > self.instance.C:
                    raise ValueError(f"Capacity constraint violation at {location_idx} is violated")
            else:  # dropoff
                capacity -= self.instance.demands[location_idx - self.instance.n]
        return capacity

    def check_capacity_constraint(self) -> None:
        total_capacity = self._check_capacity_constraint_at_position(len(self.route))
        if total_capacity != 0:
            raise ValueError(f"Capacity constraint is violated with total capacity {total_capacity} instead of 0")

    def n_served_requests(self) -> int:
        return len(self.served_requests)

    def check(self) -> None:
        if len(self.route) % 2 !=  0:
            raise ValueError(f"Route of odd length: {len(self.route)}; the length must be even")

        for request_id in self.served_requests:
            pickup_idx = request_id
            dropoff_idx = request_id + self.instance.n

            if pickup_idx not in self.route:
                raise ValueError(f"Request {request_id} is marked as served but pickup location {pickup_idx} not in route")

            if dropoff_idx not in self.route:
                raise ValueError(f"Request {request_id} is marked as served but dropoff location {dropoff_idx} not in route")

            pickup_pos = self.route.index(pickup_idx)
            dropoff_pos = self.route.index(dropoff_idx)

            if pickup_pos >= dropoff_pos:
                raise ValueError(f"Request {request_id}: dropoff at position {dropoff_pos} must come after pickup at position {pickup_pos}")

        self.check_capacity_constraint()

    def swap_locations(self, location_a, location_b) -> None:
        self.route[location_a], self.route[location_b] = self.route[location_b], self.route[location_a]
        self._recompute_capacities()
        self.check()
        self.recompute_route_distance()

    def move_from_to(self, move_from: int, move_to: int) -> None:
        moved_value = self.route.pop(move_from)
        self.route.insert(move_to, moved_value)
        self._recompute_capacities()
        self.check()
        self.recompute_route_distance()

    def relocate_request_to(self, request_id: int, route_to: 'Route', pos_to: int, dropoff_distance_from_pickup: int) -> None:
        self.remove_request(request_id)
        route_to.serve_request(request_id, pos_to, dropoff_distance_from_pickup)

    def remove_request(self, request_id: int) -> None:
        """Remove both pickup and dropoff locations for a request from this route."""
        pickup_idx = request_id
        dropoff_idx = request_id + self.instance.n

        pickup_pos = self.route.index(pickup_idx)
        dropoff_pos = self.route.index(dropoff_idx)
        demand = self.instance.demands[request_id]

        # TSP-like delta evaluation: calculate only affected edges
        old_distance = self.distance

        # Delta update capacities before route modification
        self._update_capacities_for_remove(pickup_pos, dropoff_pos, demand)

        # Calculate deltas before removal (remove in reverse order)
        # Remove dropoff first (higher index)
        dropoff_delta = self._calculate_removal_delta(dropoff_pos)
        self.route.pop(dropoff_pos)

        # Then remove pickup (index hasn't shifted because we removed from the end)
        pickup_delta = self._calculate_removal_delta(pickup_pos)
        self.route.pop(pickup_pos)

        if request_id in self.served_requests:
            self.served_requests.remove(request_id)

        self.check()

        # Update distance incrementally
        self.distance = old_distance + pickup_delta + dropoff_delta
        self.delta_callback(self.distance, old_distance)


class SCFPDPSolution(Solution):

    to_maximize = False

    def __init__(self, inst: SCFPDPInstance) -> None:
        super().__init__(inst)
        self.routes: list[Route] = [Route(inst, self.delta_evaluation) for _ in range(inst.n_K)]
        self.inst = inst  # this is overridden simply to help compiler with type hinting

        self._cached_total_distance: int = 0
        self._cached_sum_of_squares: int = 0

    def __repr__(self):
        is_valid = True
        error = 'N/A'
        try:
            self.check()
        except ValueError as e:
            is_valid = False
            error = e.args[0]
        objective_message = f'Objective: {self.calc_objective():.2f}'
        lines = [f"SCFPDPSolution({objective_message if is_valid else f'Invalid solution: "{error}";{objective_message}'})"]
        for vehicle_i, route in enumerate(self.routes):
            lines.append(f"  Vehicle {vehicle_i}: {route}")
        total_requests = sum(route.n_served_requests() for route in self.routes)
        lines.append(f"  Total requests served: {total_requests}/{self.inst.n} (min required: {self.inst.gamma})")
        return '\n'.join(lines)

    def copy_from(self, other: 'SCFPDPSolution'):
        self.routes = copy.deepcopy(other.routes)
        self._cached_total_distance = other._cached_total_distance
        self._cached_sum_of_squares = other._cached_sum_of_squares

        # Reconnect callbacks to this solution's method
        for route in self.routes:
            route.delta_callback = self.delta_evaluation

    def copy(self):
        sol = SCFPDPSolution(self.inst)
        sol.copy_from(self)
        return sol

    def invalidate(self):
        self._cached_total_distance = 0.0
        self._cached_sum_of_squares = 0.0

    def delta_evaluation(self, new_distance: int, old_distance: int) -> None:
        self._cached_total_distance += (new_distance - old_distance)
        self._cached_sum_of_squares += (new_distance ** 2 - old_distance ** 2)

    def calc_objective(self) -> float:
        if self._cached_sum_of_squares == 0:
            return 0
        fairness = self._cached_total_distance**2 / (self.inst.n_K * self._cached_sum_of_squares)
        return self._cached_total_distance + self.inst.rho * (1 - fairness)

    def initialize(self, k):
        from src.construction_heuristics import GreedyConstructionHeuristic
        self.invalidate()
        GreedyConstructionHeuristic(self).construct()

    def check(self):
        super().check()

        # number of routes is not greater than number of vehicles (not sure this check is required)
        if len(self.routes) > self.inst.n_K:
            raise ValueError(f"Expected up to {self.inst.n_K} routes but got {len(self.routes)}")

        all_served_requests = set()
        for route_i, route in enumerate(self.routes):
            try:
                route.check()
            except ValueError as e:
                raise ValueError(f"Route {route_i} is invalid: {e}") from e

            for request_id in route.served_requests:
                if request_id in all_served_requests:
                    raise ValueError(f"Request {request_id} appears in multiple routes")
                all_served_requests.add(request_id)

        n_total_served_requests = len(all_served_requests)
        if n_total_served_requests < self.inst.gamma:
            raise ValueError(f"Not enough requests served: {n_total_served_requests} < {self.inst.gamma} (gamma)")
        return None

    def get_all_served_requests(self) -> set[int]:
        return set().union(*[r.served_requests for r in self.routes])

    def is_complete(self) -> bool:
        # self.check()
        return len(self.get_all_served_requests()) == self.inst.gamma


if __name__ == '__main__':
    print(SCFPDPSolution(inst=SCFPDPInstance('10/test_instance_small.txt')))