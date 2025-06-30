import numpy as np

from _util import json_load, plt
from _tsp import OPT_COST

# ---- Lineplot: Results by Configuration and Generations ---- #
# TODO: Compare CIs vs. Stds, check if boxplots are better
# TODO: Consistent AXHLINE for OPT_COST in all plots (C2, label)
results = json_load("02_ga_generations.json")
plt.figure(figsize=(10, 6))

lines = []
configurations = results.pop("configurations")
generation_opt = results.pop("generation_opt")

for name, config in configurations.items():
    gen_results = results[name]
    means = [np.mean(gen_results[f"{gen}"]) for gen in generation_opt]
    stds = [np.std(gen_results[f"{gen}"]) for gen in generation_opt]

    if name == "Best":
        # Highlight this one
        plt.errorbar(
            generation_opt,
            means,
            yerr=stds,
            label=name,
            capsize=4,
            marker="o",
            color="tab:blue",
            linewidth=2.5,
        )
    else:
        # Gray lines with legend
        plt.plot(
            generation_opt,
            means,
            label=name,
            color="gray",
            linewidth=1.2,
            alpha=0.6,
            linestyle="--",
        )

plt.axhline(y=OPT_COST, color="C2")
plt.xlabel("Number of Generations")
plt.ylabel("Best Cost")
plt.grid(True)
plt.legend(title="Configuration")
plt.tight_layout()
plt.savefig("02_ga_generations.pdf")
