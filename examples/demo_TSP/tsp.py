# Copyright 2025 EvoBandits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


class TSP:
    """
    A Traveling Salesman Problem (TSP) solver using a distance matrix.
    """

    def __init__(self, n_cities: int, dist_matrix: np.ndarray) -> None:
        """
        Initialize the TSP.

        Args:
            n_cities (int): Number of cities in the TSP instance.
            dist_matrix (np.ndarray): 2D array representing the pairwise distances between cities.
        """
        self.n_cities: int = n_cities
        self.dist_matrix: np.ndarray = dist_matrix

    def calc_total_dist(self, route: np.ndarray) -> float:
        """
        Calculate the total distance of the given route.

        Note: This method does not check if a route is 'valid', i.e., whether it contains each
        city exactly once. This decision was made to prioritize performance, as validity checks
        can be computationally expensive for a large n_cities or repeated calculation.

        Args:
            route (np.ndarray): 1D array of city indices representing the visiting order.

        Returns:
            float: Total distance of the route, including the return to the starting city.
        """
        total_dist: float = np.sum(
            self.dist_matrix[route, np.roll(route, -1)], dtype=float
        )
        return total_dist


if __name__ == "__main__":
    """Usage example for the TSP"""

    from scipy.spatial.distance import cdist

    # Generate a number of random cities.
    n_cities = 5
    print(f"Usage Example for a TSP with {n_cities}:")

    # Calculate their symmetric, pairwise distances.
    rng = np.random.default_rng(seed=42)
    coordinates = rng.random(size=(n_cities, 2))
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    print(f"Pairwise Distances:\n{dist_matrix}:")

    # Generate a route
    route = np.random.permutation(n_cities)
    print(f"Random Route:\t{route}")

    # Initialize a TSP instance to calculate the cost of a random route.
    tsp = TSP(n_cities, dist_matrix)
    route_dist = tsp.calc_total_dist(route)
    print(f"Total distance:\t{route_dist}")
