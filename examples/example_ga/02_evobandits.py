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


# ---- Setup ---- #
from datetime import datetime
from evobandits import EvoBandits, Study, CategoricalParam
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from _tsp import (
    genetic_algorithm,
    patch_plotlib_tourplot,
    get_records_dir,
    BEST_COST,
    BEST_TOUR,
)

# ---- Constants ---- #
N_RUNS = 20
SIM_BUDGET = 180
EVAL_BUDGET = 500
GENERATIONS = 1000

# ---- Simulation 1 ---- #
print("Starting Simulation 1: EvoBandits Demonstration")


# Set up the objective function for evobandits
def ga_objective(seed: int = -1, **params):
    best_cost, _ = genetic_algorithm(generations=GENERATIONS, seed=seed, **params)
    return best_cost  # returns only objective value


# Define the solution space
params = {
    "population_size": CategoricalParam([100, 250, 500]),
    "elite_size": CategoricalParam([0, 10, 20]),
    "tournament_size": CategoricalParam(list(range(0, 10))),
    "crossover_rate": CategoricalParam(
        np.round(np.arange(0.0, 1.01, 0.05), 2).tolist()
    ),
    "mutation_rate": CategoricalParam(np.round(np.arange(0.0, 1.01, 0.05), 2).tolist()),
}

# Configure and run the optimizer
algorithm = EvoBandits()  # GMAB algorithm with default configuration

study = Study(seed=42, algorithm=algorithm)
study.optimize(ga_objective, params, n_trials=BUDGET)

# Print the results
print(
    "Best result found by the genetic algorithm:\n"
    f"Configuration:\t{study.best_params}\n"
    f"Best distance:\t{study.best_value}\n"
    f"Mean distance:\t{study.mean_value}\n"
)

# ---- Plots for Simulation 1 ---- #
# Compare the best tour to an example create using the best GA config.
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
patch_plotlib_tourplot()
ax[0].tourplot(BEST_TOUR, line_color="#377eb8")
ax[0].set_title(f"Near-optimal solution for the TSP\nTotal cost: {BEST_COST:.4f}")

cost, tour = genetic_algorithm(seed=42, generations=GENERATIONS, **study.best_params)
ax[1].tourplot(tour, line_color="#377eb8")
ax[1].set_title(
    "Example solution using best genetic_algorithm configuration\n"
    f"Total cost: {cost:.4f}"
)
plt.savefig(get_records_dir() / "02.evobandits_tour_demo.pdf")


# ---- Simulation 2 ---- #
print("Starting Simulation 2: Track EvoBandits over multiple runs")

records = {}
for i in tqdm(range(1, N_RUNS + 1), desc="EvoBandits Run"):
    print("\nRun Simulation with EvoBandits...")
    algorithm = EvoBandits(population_size=1)
    study = Study(seed=i, algorithm=algorithm)
    study.optimize(ga_objective, params, n_trials=SIM_BUDGET)

    results = []
    for j in tqdm(range(1, EVAL_BUDGET + 1), desc="Estimating true value"):
        best_cost = ga_objective(seed=j, **study.best_params)
        results.append(best_cost)
    est_true_value = np.mean(results)
    records[f"{i}"] = {
        "est_true_value": est_true_value,
        "solution": study.best_solution,
    }

# ---- Plots for Simulation 2 ---- #
# Compare reported best value and estimated true value
best_values = [r["solution"]["value"] for r in records.values()]
true_values = [r["est_true_value"] for r in records.values()]

gspec = {"width_ratios": [1, 2]}
fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=gspec)

axes[0].grid(False)
axes[0].boxplot(
    [best_values, true_values],
    tick_labels=["Reported Value", "True Value"],
)
axes[0].set_ylabel("Total distance")
axes[0].axhline(BEST_COST, label="Best known", ls="--", color="#e41a1c")

axes[1].grid()
axes[1].hist(
    best_values,
    bins=10,
    label="Reported Value",
    color="#377eb8",
    alpha=0.6,
)
axes[1].hist(
    true_values,
    bins=10,
    label="True Value",
    color="#e41a1c",
    alpha=0.6,
)
axes[1].axvline(BEST_COST, label="Best known", ls="--", color="#e41a1c")
axes[1].set_xlabel("Total distance")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(get_records_dir() / "02_evobandits_results.pdf")


# ---- Save Results ---- #
summary = {
    "header": {
        "description": "Results from repeated runs of EvoBandits",
        "date": datetime.now().strftime("%Y-%m-%d"),
    },
    "n_runs": N_RUNS,
    "records": records,
}
with open(get_records_dir() / "02_evobandits_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
