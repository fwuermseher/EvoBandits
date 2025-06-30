from tqdm import tqdm
import numpy as np

from _ga import genetic_algorithm, GENERATIONS
from _util import json_dump


SAMPLES = 1000
SEED = 42
rng = np.random.default_rng(SEED)

# Repeatedly simulate the GA with a distinct configuration
# TODO: Use the elite configuration from evobandits optimization
results = []
for _ in tqdm(range(SAMPLES), desc="Collecting Samples"):
    seed = rng.integers(0, 2**32 - 1, dtype=int)
    best_cost, _ = genetic_algorithm(
        pop_size=250,
        elite_split=0.10,
        tournament_split=0.05,
        crossover_rate=0.04,
        mutation_rate=0.77,
        generations=GENERATIONS,
        seed=seed,
    )
    results.append(best_cost)

json_dump(results, "01_ga_repeats.json")
