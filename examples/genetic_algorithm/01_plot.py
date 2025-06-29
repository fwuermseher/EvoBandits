import numpy as np

from _util import plt, json_load

# ---- Distribution of Results ---- #
results = json_load("01_ga_repeats.json")
gspec = {"width_ratios": [1, 2]}
fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=gspec)

axes[0].grid(False)
axes[0].boxplot(results)
axes[0].set_ylabel("Total distance")

axes[1].grid()
axes[1].hist(results, bins=20, alpha=0.75)
axes[1].set_xlabel("Total distance")
axes[1].set_ylabel("Frequency")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig("01_ga_results.pdf")

# ---- Running mean of Results ---- #
plt.figure(figsize=(10, 5))
plt.grid()

idx = np.arange(1, len(results) + 1)
plt.scatter(idx, results, label="Results", s=10)

means = np.cumsum(results) / idx
plt.plot(idx, means, label="Running mean", color="#dd8452")

plt.xlabel("Number of runs")
plt.ylabel("Total distance")
plt.legend()
plt.savefig("01_ga_running_mean.pdf")
