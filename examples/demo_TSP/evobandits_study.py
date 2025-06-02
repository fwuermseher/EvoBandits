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


if __name__ == "__main__":
    """Optimizes a Genetic Algorithm for solving TSPs using EvoBandits"""

    import evobandits as eb
    from scipy.spatial.distance import cdist

    from tsp import TSP  # Internal module
    from datasets import kro100C as dataset  # Internal module

    # Initialize TSP
    n_cities = dataset.N_CITIES
    coordinates = dataset.COORDINATES
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    tsp = TSP(n_cities, dist_matrix)

    # Define Study Objective
    from typing import Callable
    from sko.GA import GA_TSP

    def objective_ga(
        size_pop: int,
        max_iter: int,
        prob_mut: float,
        n_dim: int = n_cities,
        fitness_func: Callable = tsp.calc_total_dist,
    ) -> float:
        """
        Run a Genetic Algorithm (GA) to optimize a given fitness function.

        Args:
            fitness_func (Callable): Function to evaluate candidate solutions.
            n_dim (int): Problem dimension.
            size_pop (int): Population size for the GA.
            max_iter (int): Maximum number of generations (iterations) to run the GA.
            prob_mut (float): Probability of mutation for each individual per generation.

        Returns:
            float: Best (minimum) fitness value found by the GA.
        """
        ga = GA_TSP(fitness_func, n_dim, size_pop, max_iter, prob_mut)
        _, best_dist = ga.run()
        print("Completed Trial:", best_dist[0])
        return best_dist[0]

    # Model solution space
    params = {
        "size_pop": eb.CategoricalParam(
            list(range(2, 1001, 2))
        ),  # must be an even number
        "prob_mut": eb.FloatParam(0.0, 1.0, nsteps=1000),
        "max_iter": eb.IntParam(10, 1000),
    }

    # Run optimization
    study = eb.Study(seed=42)
    best_result = study.optimize(objective_ga, params, n_trials=20)
    print(best_result)

    # TODO: Integrate EvoBandits 0.0.6
