import numpy as np
from tqdm import tqdm

from _ga import (
    POP_SIZE_OPT,
    ELITE_SPLIT_OPT,
    TOURNAMENT_SPLIT_OPT,
    MUTATION_RATE_OPT,
    CROSSOVER_RATE_OPT,
    genetic_algorithm,
)
from _util import json_dump

# Define a best configuration and add sample configurations
# TODO: Use the elite configuration from evobandits optimization
# TODO: Use number of evaluations from evobandits as sample cnt?
N_CONFIGS = 10
N_SAMPLES = 20
SEED = 42
rng = np.random.default_rng(SEED)


def random_config(seed: int):
    rng = np.random.default_rng(seed)
    return {
        "pop_size": int(rng.choice(POP_SIZE_OPT)),
        "elite_split": float(rng.choice(ELITE_SPLIT_OPT)),
        "tournament_split": float(rng.choice(TOURNAMENT_SPLIT_OPT)),
        "crossover_rate": float(rng.choice(CROSSOVER_RATE_OPT)),
        "mutation_rate": float(rng.choice(MUTATION_RATE_OPT)),
    }


configurations = dict(
    {
        "Best": {
            "pop_size": 250,
            "elite_split": 0.10,
            "tournament_split": 0.05,
            "crossover_rate": 0.04,
            "mutation_rate": 0.77,
        },
    }
)
for i in range(1, N_CONFIGS):
    seed = rng.integers(0, 2**32 - 1, dtype=int)
    configurations.update({f"Random_{i}": random_config(seed)})


# Simulate the GA for each configuration and different no. of generations
# TODO: Only up to 600 generations needed, starting at 200 seems enough
generation_opt = list(range(300, 400, 10))

results = {
    "configurations": configurations,
    "generation_opt": generation_opt,
}

for name, config in configurations.items():
    results[name] = {}

    for gen in generation_opt:
        gen_results = []
        for _ in tqdm(range(N_SAMPLES), desc=f"{name} | Gen {gen}"):
            seed = rng.integers(0, 2**32 - 1, dtype=int)
            cost, _ = genetic_algorithm(
                generations=gen,
                pop_size=250,
                elite_split=0.10,
                tournament_split=0.05,
                crossover_rate=0.04,
                mutation_rate=config["mutation_rate"],
                seed=seed,
            )
            gen_results.append(cost)
        results[name][gen] = gen_results

json_dump(results, "02_ga_generations.json")
