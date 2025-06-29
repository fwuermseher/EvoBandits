from tqdm import tqdm

from _ga import genetic_algorithm, GENERATIONS
from _util import json_dump

# Repeatedly simulate the GA with a distinct configuration
# TODO: Use the elite configuration from evobandits optimization
SAMPLES = 1000
results = []
for i in tqdm(range(SAMPLES), desc="Collecting Samples"):
    best_cost, _ = genetic_algorithm(
        pop_size=250,
        elite_split=0.10,
        tournament_split=0.05,
        crossover_rate=0.04,
        mutation_rate=0.77,
        generations=GENERATIONS,
        seed=i,
    )
    results.append(best_cost)

json_dump(results, "01_ga_repeats.json")
