import numpy as np
import math
from scipy.spatial.distance import cdist

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance_from(self, other: "Point") -> int:
        """ Euclidean distance between two points. """
        return math.ceil(math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class SCFPDPInstance:
    """An instance of the Selective Capacitated Fair Pickup and Delivery Problem.

    Attributes:
        n: number of customer requests
        n_K: number of vehicles
        C: vehicle capacity
        gamma: minimum number of requests to fulfill
        rho: fairness weight parameter
        demands: array of demand values for each request [c1, c2, ..., cn]
        depot_location: (x, y) coordinates of the depot
        pickup_locations: array of (x, y) coordinates for each pickup location
        dropoff_locations: array of (x, y) coordinates for each dropoff location
        distance_matrix: precomputed distance matrix for all locations
    """

    def __init__(self, file_name: str):
        """
        Args:
            file_name: path to the instance file
        """
        self.file_name = file_name

        self.n = None
        self.n_K = None
        self.C = None
        self.gamma = None
        self.rho = None

        self.demands: np.ndarray | None = None
        self.depot_location: Point | None = None
        self.pickup_locations: list[Point] | None = None
        self.dropoff_locations: list[Point] | None = None
        self.distance_matrix: np.ndarray[int] | None = None

        self._parse_file(file_name)
        self._compute_distances()

    def _parse_file(self, file_name: str):
        """
        File format:
            n n_K C gamma rho
            # demands
            c1 c2 ... cn
            # request locations
            x_depot y_depot
            x_pickup1 y_pickup1 x_pickup2 y_pickup2 ... x_pickupn y_pickupn
            x_dropoff1 y_dropoff1 x_dropoff2 y_dropoff2 ... x_dropoffn y_dropoffn
        """
        from pathlib import Path
        from src.utils import find_project_root

        file_path = Path(file_name)
        if not file_path.is_absolute():
            project_root = find_project_root()
            file_path = project_root / "scfpdp_instances" / file_name

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        params = lines[0].split()
        self.n = int(params[0])
        self.n_K = int(params[1])
        self.C = int(params[2])
        self.gamma = int(params[3])
        self.rho = float(params[4])

        demands_idx = lines.index('# demands') + 1
        self.demands = np.array([int(x) for x in lines[demands_idx].split()])

        depot_idx = lines.index('# request locations') + 1
        depot_coords = lines[depot_idx].split()
        self.depot_location = Point(int(depot_coords[0]), int(depot_coords[1]))

        pickup_start_idx = depot_idx + 1
        self.pickup_locations = np.array(
            [
                Point(int(lines[pickup_start_idx + i].split()[0]),
                      int(lines[pickup_start_idx + i].split()[1]))
                for i in range(self.n)
            ]
        )

        dropoff_start_idx = pickup_start_idx + self.n
        self.dropoff_locations = np.array(
            [
                Point(int(lines[dropoff_start_idx + i].split()[0]),
                      int(lines[dropoff_start_idx + i].split()[1]))
                for i in range(self.n)
            ]
        )

    def _compute_distances(self):
        """Compute the distance matrix for all locations using vectorized operations.

        Distance is Euclidean distance rounded up to next integer.

        Matrix layout:
            Index 0: depot
            Index 1 to n: pickup locations for requests 1 to n
            Index n+1 to 2n: drop-off locations for requests 1 to n

        Total size: (2n + 1) x (2n + 1)
        """
        all_locations = [self.depot_location] + list(self.pickup_locations) + list(self.dropoff_locations)

        # Convert Point objects to numpy coordinate array
        coords = np.array([[loc.x, loc.y] for loc in all_locations], dtype=np.float64)

        # Use scipy's cdist for optimized pairwise distance computation
        # cdist computes distances without creating large intermediate arrays
        distances = cdist(coords, coords, metric='euclidean')
        self.distance_matrix = np.ceil(distances)


    def __repr__(self):
        return (f"SCFPDPInstance(n_requests={self.n}, n_vehicles={self.n_K}, capacity={self.C}, "
                f"min_n_of_requests_to_serve={self.gamma}, fairness_weight={self.rho})")

if __name__ == '__main__':
    instance = SCFPDPInstance('10/test_instance_small.txt')
    print(instance)
    print(instance.distance_matrix)
    assert np.array_equal(instance.distance_matrix, instance.distance_matrix.T)


def scfpdp():
    return None