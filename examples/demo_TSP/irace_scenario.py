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


# rpy2 appears to be broken.
# Install pip install git+https://github.com/fwuermseher/rpy2.git@add-consolewrite-to-CALLBACK-INIT-PAIRS --no-cache-dir

from irace import (
    irace,
    ParameterSpace,
    Categorical,
    Integer,
    Scenario,
    Experiment,
)

import numpy as np
from scipy.spatial.distance import cdist
from sko.GA import GA_TSP

from tsp import TSP  # Internal module
from datasets import kro100C as dataset  # Internal module

if __name__ == "__main__":
    # Initialize TSP
    n_cities = dataset.N_CITIES
    coordinates = dataset.COORDINATES
    dist_matrix = cdist(coordinates, coordinates, metric="euclidean")
    tsp = TSP(n_cities, dist_matrix)

    def target_runner(experiment: Experiment, scenario: Scenario) -> float:
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
        ga = GA_TSP(
            func=tsp.calc_total_dist, n_dim=n_cities, **experiment.configuration
        )
        _, best_dist = ga.run()
        return best_dist[0]

    parameter_space = ParameterSpace(
        [
            Categorical("size_pop", list(range(2, 1001, 2))),
            Categorical("prob_mut", [round(x, 4) for x in np.arange(0.0, 1.0, 0.0001)]),
            Integer("max_iter", 10, 1000),
        ]
    )

    scenario = Scenario(
        max_experiments=500,  # matches n_trials
        verbose=1,
        seed=42,
    )
    result = irace(target_runner, parameter_space, scenario, return_df=True)

    print(result)
