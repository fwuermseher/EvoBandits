from tqdm import tqdm
import json

from _ga import genetic_algorithm
from _util import RESULTS_DIR

# Repeatedly simulate the GA with a distinct configuration
# TODO: Use the elite configuration from evobandits optimization
samples = 1000
results = []
for i in tqdm(range(samples), desc="Collecting Samples"):
    best_cost, _ = genetic_algorithm(
        pop_size=250,
        elite_split=0.10,
        tournament_split=0.05,
        crossover_rate=0.04,
        mutation_rate=0.77,
        seed=i,
    )
    results.append(best_cost)

with open(RESULTS_DIR / "01_ga_repeats.json", "w") as f:
    json.dump(results, f, indent=2)
