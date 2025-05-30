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

# Optimization of a Genetic Algorithm (GA) that solves a Traveling Salesman Problem (TSP).
#
# Reference:
# https://scikit-opt.github.io/scikit-opt/#/en/more_ga?id=how-to-fix-start-point-and-end-point-with-ga-for-tsp

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from sko.GA import GA_TSP

### Setup the TSP ###
# grid with random coordinates with N_DIM cities
N_DIM = 20
CITY_COORDINATES = np.random.rand(N_DIM, 2)  # generate coordinate of points

# Routes will start at [0, 0] and end at [1, 1]
START = [[0, 0]]
END = [[1, 1]]
CITY_COORDINATES = np.concatenate([CITY_COORDINATES, START, END])

# Calculate the symmetric, euclidean distance matrix for CITY_COORDINATES
DIST_MATRIX = cdist(CITY_COORDINATES, CITY_COORDINATES, metric="euclidean")


def calc_total_distance(routine):
    """Calculate the total distance for a route in the TSP.
    This function is used as fitness function for the GA."""
    (num_points,) = routine.shape
    routine = np.concatenate([[num_points], routine, [num_points + 1]])
    return sum(
        [DIST_MATRIX[routine[i], routine[i + 1]] for i in range(num_points + 2 - 1)]
    )


if __name__ == "__main__":
    # Try to solve the TSP using a Genetic Algorithm from scikit-opt.
    # GA_TSP overloads mutation and crossover specifically for a route planning problem.
    ga_tsp = GA_TSP(
        func=calc_total_distance, n_dim=N_DIM, size_pop=50, max_iter=500, prob_mut=1
    )
    best_points, best_distance = ga_tsp.run()

    # Plot the resulting route and progression of the GA
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([[N_DIM], best_points, [N_DIM + 1]])
    best_points_coordinate = CITY_COORDINATES[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], "o-r")
    ax[1].plot(ga_tsp.generation_best_Y)
    plt.show()
