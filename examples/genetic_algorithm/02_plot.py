import numpy as np

from _util import json_load, plt

# ---- Lineplot: Results by Configuration and Generations ---- #
# TODO: Compare CIs vs. Stds, check if boxplots are better
results = json_load("02_ga_generations.json")
plt.figure(figsize=(10, 6))

lines = []
configurations = results.pop("configurations")
generation_opt = results.pop("generation_opt")

for name, config in configurations.items():
    gen_results = results[name]
    means = [np.mean(gen_results[f"{gen}"]) for gen in generation_opt]
    stds = [np.std(gen_results[f"{gen}"]) for gen in generation_opt]

    # cis = [1.96 * (np.std(gen_results[f"{gen}"]) / np.sqrt(len(gen_results[f"{gen}"]))) for gen in generation_opt]

    if name == "Best":
        # Highlight this one
        line = plt.errorbar(
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
        (line,) = plt.plot(
            generation_opt,
            means,
            label=name,
            color="gray",
            linewidth=1.2,
            alpha=0.6,
            linestyle="--",
        )
    lines.append(line)

plt.xlabel("Number of Generations")
plt.ylabel("Best Cost")
plt.grid(True)
plt.legend(title="Configuration")
plt.tight_layout()
plt.savefig("02_ga_generations.pdf")
