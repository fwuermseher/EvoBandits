
# ---- Setup ---- #
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from _tsp import genetic_algorithm, get_records_dir

# ---- Simulation ---- #
# Repeatedly simulates the GA with a distinct configuration
samples = 1000
results = []
for i in tqdm(range(1, samples + 1), desc="Collecting GA samples:"):
    best_cost, _ = genetic_algorithm(
        population_size=500,
        elite_size=20,
        tournament_size=4,
        mutation_rate=0.50,
        crossover_rate=0.50,
        seed=i,
    )
    results.append(best_cost)

# ---- Plot 1 ---- #
# Distribution of simulation results
fig, axes = plt.subplots(
    1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 2]}
)
axes[0].grid(False)
axes[0].boxplot(results)
axes[0].set_ylabel("Total distance")

axes[1].grid()
axes[1].hist(results, bins=20, edgecolor="black", alpha=0.75)
axes[1].set_xlabel("Total distance")
axes[1].set_ylabel("Frequency")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(get_records_dir() / "01_ga_results_dist.pdf")

# ---- Plot 2 ---- #
# Running mean of simulation results
plt.figure(figsize=(10, 5))
plt.grid()

idx = np.arange(1, samples + 1)
plt.scatter(idx, results, label="Individual results", s=10)

means = np.cumsum(results) / idx
plt.plot(idx, means, label="Running mean", color="#e41a1c")

plt.xlabel("Number of runs")
plt.ylabel("Total distance")
plt.legend()
plt.savefig(get_records_dir() / "01_ga_running_means.pdf")

# ---- Save Results ---- #
summary = {
    "header": {
        "description": "Results from repeated runs of the GA.",
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
    "sample_count": samples,
    "results": results,
}
with open(get_records_dir() / "01_ga_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
