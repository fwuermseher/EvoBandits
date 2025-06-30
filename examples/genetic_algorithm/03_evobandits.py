from tqdm import tqdm
import numpy as np
from evobandits import CategoricalParam, GMAB, Study

from _ga import (
    genetic_algorithm,
    GENERATIONS,
    POP_SIZE_OPT,
    ELITE_SPLIT_OPT,
    TOURNAMENT_SPLIT_OPT,
    CROSSOVER_RATE_OPT,
    MUTATION_RATE_OPT,
)
from _util import json_dump

# ---- Optimization of the GA using EvoBandits ---- #
# TODO: Number of Generations 300 -> 350 or 400 due to convergence?
# - Run this and irace, then compare with 02 script
N_RUNS = 10
SIM_BUDGET = 1000
EVAL_BUDGET = 500
SEED = 42
rng = np.random.default_rng(SEED)


# Objective: GA with fixed generations, returns cost only
def ga_objective(seed: int = -1, **params):
    best_cost, _ = genetic_algorithm(
        generations=GENERATIONS,
        seed=seed,
        **params,
    )
    return best_cost


# Solution Space
params = {
    "pop_size": CategoricalParam(POP_SIZE_OPT),
    "elite_split": CategoricalParam(ELITE_SPLIT_OPT),
    "tournament_split": CategoricalParam(TOURNAMENT_SPLIT_OPT),
    "crossover_rate": CategoricalParam(CROSSOVER_RATE_OPT),
    "mutation_rate": CategoricalParam(MUTATION_RATE_OPT),
}

# Simulation Optimization
results = {}
for i in tqdm(range(N_RUNS), desc="EvoBandits | Run"):
    print("\nRunning optimization ...")
    seed = rng.integers(0, 2**32 - 1, dtype=int)
    gmab = GMAB(mutation_span=0.2)
    study = Study(seed=seed, algorithm=gmab)
    study.optimize(ga_objective, params, n_trials=SIM_BUDGET)
    print(f"Config:\t{study.best_params}")
    print(f"Value:\t{study.best_value}")

    print("Estimating true value ...")
    evaluations = []
    for _ in range(EVAL_BUDGET):
        seed = rng.integers(0, 2**32 - 1, dtype=int)
        best_cost = ga_objective(seed=seed, **study.best_params)
        evaluations.append(best_cost)
    mean_evaluation = np.mean(evaluations)
    print(f"Est. true value:\t{mean_evaluation}")

    results[i] = {
        "mean_evaluation": mean_evaluation,
        "evaluations": evaluations,
        "best_solution": study.best_solution,
    }

json_dump(results, "03_evobandits.json")
