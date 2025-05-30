# Optimization with EvoBandits based on Test Problem 4 (TP4) from Preil & Krapp, 2025.
# 
# Reference:
# D. Preil and M. Krapp, "Genetic Multi-Armed Bandits: A Reinforcement Learning Inspired Approach
# for Simulation Optimization," in IEEE Transactions on Evolutionary Computation, vol. 29, no. 2, 
# pp. 360-374, April 2025, doi: 10.1109/TEVC.2024.3524505.

import numpy as np
from evobandits import EvoBandits

SEED = 42

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
    return -res # negate result for minimization

if __name__ == "__main__":
    # Print the known global optimum
    dimensions = 5
    print(f"Optimization Problem:\tTP4_D{dimensions}")

    opt_action_vector = [EPS_2] * dimensions
    print(f"Optimal action_vector:\t{opt_action_vector}")

    opt_value = tp4_func(opt_action_vector)
    print(f"Optimal function value:\t{opt_value:.4f}")

    # Direct optimization using an instance of EvoBandits
    bounds = [(-100, 100)] * dimensions # solution space
    n_trials = 20000 # number of evaluations for tp4_func
    n_best = 1 # display only best result
    algorithm = EvoBandits()
    results = algorithm.optimize(tp4_func, bounds, n_trials, n_best, SEED)
    print(f"Optimization result:\t{results[0].to_dict}")

    