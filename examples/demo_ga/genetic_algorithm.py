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


from numba import njit
import numpy as np
from scipy.spatial.distance import cdist

# Seeding for reproducibility
SEED = 42
np.random.seed(SEED)

# Constants
N_CITIES = 100

# Initialize coordinates for the Traveling Salesman Problem (TSP)
# by generating random Cartesian coordinates within a 100x100 grid.
COORDINATES = np.random.rand(N_CITIES, 2) * 100

# Compute a symmetric Euclidean distance matrix based on the coordinates.
DIST_MATRIX = cdist(COORDINATES, COORDINATES, metric="euclidean")


# Genetic algorithm and components
@njit
def _calc_total_distance(tour, dist_matrix):
    """
    Fitness function for the genetic algorithm.

    Computes the total distance of a tour by summing the distances
    between consecutive cities, based on the provided `dist_matrix`.
    The tour includes the return leg to the starting city.
    """
    total_distance = 0
    for i in range(len(tour)):
        total_distance += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return total_distance


@njit
def _rank_population(population, dist_matrix):
    """
    Calculates the total distance for each tour in the `population`
    and ranks the population in ascending order of total distance.
    """
    distances = np.empty(len(population))
    for i in range(len(population)):
        distances[i] = _calc_total_distance(population[i], dist_matrix)

    ranked_indices = np.argsort(distances)
    ranked = []
    for i in ranked_indices:
        ranked.append(population[i])
    return ranked


@njit
def _apply_selection(population, elite_size, dist_matrix):
    """
    Applies elitist and roulette wheel selection to choose individuals
    from the given `population`.

    The top `elite_size` individuals are selected directly based
    on fitness, while the remaining individuals are selected
    probabilistically (roulette wheel selection).
    """
    ranked = _rank_population(population, dist_matrix)
    selected = ranked[:elite_size]

    # Roulette wheel selection based on inverse distance (higher fitness).
    fitness = np.empty(len(ranked))
    for i in range(len(ranked)):
        fitness[i] = 1 / _calc_total_distance(ranked[i], dist_matrix)
    cum_probs = np.cumsum(fitness) / np.sum(fitness)

    for _ in range(len(population) - elite_size):
        r = np.random.rand()
        for i in range(len(cum_probs)):
            if r <= cum_probs[i]:
                selected.append(ranked[i])
                break
    return selected


@njit
def _crossover(parent1, parent2, crossover_rate):
    """
    Performs ordered crossover between `parent1` and `parent2`
    with the specified `crossover_rate`.
    """
    if np.random.rand() >= crossover_rate:
        return parent1.copy()  # No crossover; return copy of parent1.

    # Select a random subsequence from parent1.
    start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
    seq1 = parent1[start:end]

    # Fill the remainder from parent2 in order, skipping duplicates.
    seq2 = np.array([city for city in parent2 if city not in seq1])
    child = np.concatenate((seq2[:start], seq1, seq2[start:]))
    return child


@njit
def _mutation(tour, mutation_rate):
    """
    Mutates a `tour` by swapping cities with the given `mutation_rate`.
    """
    tour_copy = tour.copy()
    for i in range(len(tour_copy)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(tour_copy))
            tour_copy[i], tour_copy[j] = tour_copy[j], tour_copy[i]
    return tour_copy


@njit
def genetic_algorithm(
    population_size: int,
    elite_size: int,
    crossover_rate: float,
    mutation_rate: float,
    generations: int = 200,
    dist_matrix: np.ndarray = DIST_MATRIX,
    seed: int = -1,
) -> float:
    """
    Runs a genetic algorithm to solve a TSP instance.

    Args:
        population_size: Number of tours per generation.
        elite_size: Number of best tours preserved across generations.
        mutation_rate: The probability to an individual city in a tour.
        crossover_rate: Probability to perform a crossover on a tour.
        generations: Number of iterations. Defaults to 200.
        dist_matrix: A precomputed distance matrix for the TSP.
            Defaults to a seeded instance for 100 cities.
        seed: A random seed to reproduce results.
            Defaults to -1, a sentinel value that disables seeding.

    Returns:
        The shortest distance found.
    """
    if seed < 0:
        np.random.seed(seed)

    # Initialize population with random permutations (tours).
    population = np.empty((population_size, N_CITIES), dtype=np.int32)
    for i in range(population_size):
        population[i] = np.random.permutation(N_CITIES)

    # Evolution loop
    for _ in range(generations):
        # Select parents using elitism and roulette wheel
        parents = _apply_selection(population, elite_size, dist_matrix)

        # Shuffle parents
        for i in range(len(parents) - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            parents[i], parents[j] = parents[j].copy(), parents[i].copy()

        # Generate new population
        new_population = np.empty_like(population)
        for idx in range(len(parents)):
            parent1 = parents[idx]
            parent2 = parents[(idx + 1) % len(parents)]
            child = _crossover(parent1, parent2, crossover_rate)
            new_population[idx] = _mutation(child, mutation_rate)
        population = new_population

    # Evaluate and return the best distance
    best = _rank_population(population, dist_matrix)[0]
    return _calc_total_distance(best, dist_matrix)


if __name__ == "__main__":
    # TODO integrate evobandits 0.0.6 release (seeding, renamings)
    from evobandits import EvoBandits, Study, FloatParam, CategoricalParam

    # Print the coordinates in a format compatible with online TSP solvers
    # such as http://kay-schoenberger.de/math/tsp/
    for i in range(len(COORDINATES)):
        print(f"{i} {COORDINATES[i][0]} {COORDINATES[i][1]}")

    # Define the solution space
    params = {
        "crossover_rate": FloatParam(0.0, 1.0, nsteps=20),
        "mutation_rate": FloatParam(0.0001, 1.0, nsteps=20, log=True),
        "elite_size": CategoricalParam(choices=[0, 10, 20]),
        "population_size": CategoricalParam(choices=[20, 40, 60]),
        "generations": CategoricalParam(choices=[100, 200, 300]),
    }

    # Run the optimization with EvoBandits
    n_trials = 500  # number of evaluations for tp4_func
    n_best = 1  # display only best result
    algorithm = EvoBandits()  # GMAB algorithm with default configuration

    study = Study(SEED, algorithm)
    results = study.optimize(genetic_algorithm, params, n_trials)

    print(f"Shortest distance found using the GA:\t{study.best_value}")
    print(f"Configuration for shortest distance:\t{study.best_params}")
