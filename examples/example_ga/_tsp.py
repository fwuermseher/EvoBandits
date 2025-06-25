
import os
from pathlib import Path

from numba import njit, prange
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


# Dataset and (near) optimal tour for a TSP instance with 91 cities.
# Total distance for the (near) optimal tour: 5.209008402820838.
#
# Generated using the online TSP solver available under:
# http://kay-schoenberger.de/math/tsp/ (last accessed: 23.06.2025).
_DATASET = """
1 0.226571 0.920208
2 0.173548 0.666576
3 0.463141 0.726396
4 0.950048 0.654267
5 0.98029 0.621904
6 0.0924908 0.628441
7 0.133865 0.564027
8 0.953627 0.712146
9 0.43007 0.932855
10 0.948039 0.748363
11 0.823157 0.933456
12 0.533437 0.64413
13 0.884641 0.705531
14 0.420899 0.942301
15 0.118399 0.612815
16 0.417606 0.115678
17 0.387101 0.376491
18 0.610481 0.222624
19 0.573267 0.652292
20 0.350366 0.905765
21 0.870916 0.737875
22 0.33877 0.671757
23 0.685357 0.982444
24 0.142192 0.697105
25 0.506606 0.61696
26 0.421007 0.106234
27 0.460827 0.644382
28 0.11654 0.525603
29 0.736953 0.138288
30 0.975475 0.624818
31 0.971237 0.543892
32 0.38997 0.68663
33 0.238885 0.206517
34 0.48365 0.623336
35 0.76367 0.966839
36 0.861239 0.693292
37 0.399184 0.300334
38 0.414877 0.408013
39 0.962292 0.80588
40 0.524409 0.701671
41 0.742496 0.245337
42 0.376045 0.631789
43 0.385346 0.098467
44 0.968311 0.719672
45 0.384829 0.863832
46 0.599514 0.221102
47 0.279295 0.290314
48 0.695641 0.259127
49 0.700251 0.223603
50 0.432597 0.603319
51 0.212764 0.924568
52 0.218393 0.663596
53 0.73793 0.915311
54 0.363093 0.0897026
55 0.742211 0.188025
56 0.546376 0.414004
57 0.981505 0.757802
58 0.168869 0.520973
59 0.555588 0.67264
60 0.285697 0.209828
61 0.429293 0.421913
62 0.546651 0.616212
63 0.612063 0.272478
64 0.226636 0.930833
65 0.380412 0.931925
66 0.788633 0.92208
67 0.64603 0.120722
68 0.444141 0.91521
69 0.5768 0.656541
70 0.579154 0.700848
71 0.471602 0.704903
72 0.732228 0.228863
73 0.146636 0.711516
74 0.138144 0.703469
75 0.210048 0.494679
76 0.642175 0.281504
77 0.181842 0.931918
78 0.531121 0.605864
79 0.243533 0.217772
80 0.847997 0.743102
81 0.451813 0.158381
82 0.818467 0.933958
83 0.400143 0.108261
84 0.568958 0.105209
85 0.403605 0.273656
86 0.696969 0.237763
87 0.935325 0.689895
88 0.502419 0.680108
89 0.779744 0.974716
90 0.710385 0.220587
91 0.109638 0.648065
"""

BEST_TOUR = [
    44, 19,  0, 50, 76, 63, 64, 13,  8, 67, 22, 52, 34, 88, 65, 81,
    10, 79, 20, 35, 12,  9, 38, 56, 43,  7, 86,  3, 29,  4, 30, 40,
    71, 47, 85, 48, 89, 54, 28, 66, 83, 45, 17, 62, 75, 55, 60, 37,
    16, 36, 84, 80, 15, 25, 82, 42, 53, 59, 32, 78, 46, 74, 57, 27,
     6, 14,  5, 90, 23, 73, 72,  1, 51, 21, 31, 41, 49, 26, 33, 24,
    77, 61, 11, 18, 68, 58, 69, 39, 87, 70,  2
]

BEST_COST = 5.209008402820838


# Extract the coordinates and the number of cities in the dataset.
CITIES = np.array(
    [
        [float(parts[1]), float(parts[2])]
        for line in _DATASET.strip().splitlines()
        if (parts := line.split())
    ]
)
N_CITIES = len(CITIES)

# Compute a symmetric Euclidean distance matrix from the coordinates.
DIST_MATRIX = cdist(CITIES, CITIES, metric="euclidean")


# Genetic Algorithm, components, and utility for plotting.
@njit
def _calc_total_distance(tour):
    """Calculate the the total distance of a tour for the TSP.

    Args:
        tour (np.ndarray): The order of cities visited in the tour.

    Returns:
        float: Total distance of the tour.
    """
    # Calculate the sum of distances for consecutive cities.
    total_distance = 0.0
    for i in range(N_CITIES - 1):
        total_distance += DIST_MATRIX[tour[i], tour[i + 1]]

    # Add the return leg back to the starting city.
    total_distance += DIST_MATRIX[tour[-1], tour[0]]

    return total_distance


@njit
def _ranking(pop):
    """Rank the population based on fitness

    Args:
        pop (np.ndarray): 2D array where each row represents a tour

    Returns:
        tuple: Ranked population and their corresponding fitness values
    """
    # Calculate the fitness value for each tour.
    pop_size = len(pop)
    fitness = np.empty(pop_size, dtype=np.float64)
    for i in prange(pop_size):
        fitness[i] = 1.0 / _calc_total_distance(pop[i])

    # Rank the population based on higher fitness (lower distance).
    ranked_indices = np.argsort(fitness)[::-1]
    ranked = pop[ranked_indices]

    return ranked, fitness[ranked_indices]


@njit
def _crossover(parent1, parent2, crossover_rate):
    """Perform an ordered crossover between two parent tours.

    Args:
        parent1 (np.ndarray): First parent tour.
        parent2 (np.ndarray): Second parent tour.
        crossover_rate (float): Probability of performing crossover.

    Returns:
        np.ndarray: The resulting child tour if crossover is performed,
            a copy of the first parent tour otherwise.
    """
    if np.random.rand() >= crossover_rate:
        return parent1.copy()

    # Select a random sequence from the first parent.
    start, end = sorted(np.random.choice(N_CITIES, 2, replace=False))
    child = np.full(N_CITIES, -1, dtype=parent1.dtype)
    child[start:end] = parent1[start:end]

    # Fill the remaining cities in order from the second parent.
    pos = end
    for city in parent2:
        if not _city_in_tour(city, child):
            while child[pos % N_CITIES] != -1:
                pos += 1
            child[pos % N_CITIES] = city

    return child


@njit
def _city_in_tour(city, tour):
    """Check if a city is present in a tour.

    Args:
        city (int): The index of the city to search for.
        tour (np.ndarray): The tour, or sequence of cities to search.

    Returns:
        bool: True if city found in tour, False otherwise.

    Note:
        While using `np.isin` would be more concise and readable,
        this function can be more performant and memory-efficient
        under Numba JIT compilation, as it avoids creating
        intermediate boolean arrays.
    """
    for i in range(len(tour)):
        if tour[i] == city:
            return True
    return False


@njit
def _mutation(tour, mutation_rate):
    """Perform a swap mutation on a tour.

    Args:
        tour (np.ndarray): Tour to mutate.
        mutation_rate (float): Probability of performing mutation.

    Returns:
        np.ndarray: Mutated tour
    """
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


@njit
def _selection(pop, fitness, tournament_size):
    """Select a parent from the population using tournament selection.

    Args:
        pop (np.ndarray): 2D array where each row represents a tour.
        fitness (np.ndarray): Fitness values for the population.
        tournament_size (int): Number of individuals for the tournament.

    Returns:
        np.ndarray: Selected parent tour.
    """
    pop_size = len(pop)
    selected_idx = -1
    best_fit = -1e9
    for _ in range(tournament_size):
        idx = np.random.randint(0, pop_size)
        if fitness[idx] > best_fit:
            best_fit = fitness[idx]
            selected_idx = idx
    return pop[selected_idx]


@njit
def genetic_algorithm(
    population_size,
    elite_size,
    tournament_size,
    crossover_rate,
    mutation_rate,
    generations=2000,
    seed=-1,
):
    """Execute the genetic algorithm (GA) to solve the TSP instance.

    Args:
        population_size (int): Number of tours per generation.
        elite_size (int): Number of best tours preserved over iterations.
        tournament_size (int): Size of the tournament for selection.
        crossover_rate (float): Probability of performing crossover.
        mutation_rate (float): Probability of performing mutation.
        generations (int): Number of iterations for the evolution.
            Defaults to 2000.
        seed (int): Random seed to reproduce results.
            Defaults to -1, a sentinel value that disables seeding.
        ret_tour (int): Flag to indicate if best tour should be returned.
            Defaults to 0 (False).

    Returns:
        tuple: Cost of the best tour found and the best tour itself.

    Note:
        This GA uses a precomputed, hard-coded distance matrix based
        on a fixed set of 91 cities.
    """
    # Initialize the rng to reproduce
    if seed != -1:
        np.random.seed(seed)

    # Initialize the population with random tours.
    pop = np.empty((population_size, N_CITIES), dtype=np.int32)
    for i in range(population_size):
        pop[i] = np.random.permutation(N_CITIES)

    # For each evolution generation:
    for _ in range(generations):
        # Compute fitness and rank the population
        pop, fitness = _ranking(pop)

        # Select elite individuals from the population
        offspring = np.empty_like(pop)
        offspring[:elite_size] = pop[:elite_size]

        # Select and modify individuals to fill the population
        for i in range(elite_size, population_size):
            parent1 = _selection(pop, fitness, tournament_size)
            parent2 = _selection(pop, fitness, tournament_size)
            child = _crossover(parent1, parent2, crossover_rate)
            child = _mutation(child, mutation_rate)
            offspring[i] = child

        pop = offspring

    # Apply final ranking to select the best individual tour.
    pop, _ = _ranking(pop)
    best_tour = pop[0]
    best_cost = _calc_total_distance(best_tour)

    return best_cost, best_tour


def _tourplot(tour, cities=CITIES, ax=None, line_color="blue"):
    """Plot a TSP tour on a 2D plane.

    Args:
        tour (np.ndarray): Indices of cities representing the tour.
        cities (np.ndarray): Cities to plot. Defaults to fixed `CITIES`.
        ax (matplotlib.axes.Axes): Axis to plot on. Defaults to new axis.
        line_color (str): Line color for the tour. Defaults to 'blue'.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter the cities
    ax.scatter(cities[:, 0], cities[:, 1], c="black")

    # Plot the tour path
    ordered_points = cities[tour]
    tour_points = np.vstack([ordered_points, ordered_points[0]])
    ax.plot(tour_points[:, 0], tour_points[:, 1], "-", color=line_color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")
    return ax


def patch_plotlib_tourplot():
    """Utility to extend pyplot and Axes with tourplot"""
    plt.tourplot = lambda tour, **kwargs: _tourplot(  # type: ignore
        tour, **kwargs
    )
    Axes.tourplot = lambda self, tour, **kwargs: _tourplot(  # type: ignore
        tour, ax=self, **kwargs
    )


def get_records_dir():
    """Utility to setup a directory for the analysis records."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    records_dir = Path(script_dir) / "records"
    os.makedirs(records_dir, exist_ok=True)
    return records_dir
