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

# Optimization with EvoBandits based on Test Problem 4 (TP4) from Preil & Krapp, 2025.
#
# Reference:
# D. Preil and M. Krapp, "Genetic Multi-Armed Bandits: A Reinforcement Learning Inspired Approach
# for Simulation Optimization," in IEEE Transactions on Evolutionary Computation, vol. 29, no. 2,
# pp. 360-374, April 2025, doi: 10.1109/TEVC.2024.3524505.

import numpy as np
from evobandits import EvoBandits, IntParam, Study

# Seeding to ensure objective and EvoBandits instances are reproducible
SEED = 42
RNG = np.random.default_rng(SEED)

# Constants for TP4
BETA_1 = 300
BETA_2 = 500
GAMMA_1 = 0.001
GAMMA_2 = 0.005
EPS_1 = -38
EPS_2 = 56


def tp4_func(action_vector: list[int]) -> float:
    """Calculates the function value for TP4"""
    res = 0
    for val in action_vector:
        res += BETA_1 * np.exp(-GAMMA_1 * (val - EPS_1) ** 2) + BETA_2 * np.exp(
            -GAMMA_2 * (val - EPS_2) ** 2
        )
    res += RNG.normal(
        loc=0, scale=100 * len(action_vector)
    )  # add noise (100 * dimension)
    res = -res  # negate result to model a minimization problem
    return res


if __name__ == "__main__":
    # Print the known global optimum
    dimension = 5
    print(f"Optimization Problem:\tTP4_D{dimension}")

    opt_action_vector = [EPS_2] * dimension
    print(f"Optimal action_vector:\t{opt_action_vector}")

    opt_value = -500.4 * dimension
    print(f"Optimal function value:\t{opt_value:.4f}")

    # Define the solution space
    params = {"action_vector": IntParam(-100, 100, size=dimension)}

    # Run the optimization with EvoBandits:
    n_trials = 20_000  # number of evaluations for tp4_func
    n_best = 1  # display only best result
    algorithm = EvoBandits()  # GMAB algorithm with default configuration

    study = Study(SEED, algorithm)
    results = study.optimize(tp4_func, params, n_trials, n_best=n_best)
    print(f"Optimization result:\t{study.best_solution}")
