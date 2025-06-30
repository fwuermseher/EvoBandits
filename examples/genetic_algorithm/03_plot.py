from _util import plt, json_load
from _tsp import OPT_COST

results = json_load("03_evobandits.json")

# ---- Compare reported result and estimated true value --- #
best_values = [r["best_solution"]["value"] for r in results.values()]
mean_evaluations = [r["mean_evaluation"] for r in results.values()]

gspec = {"width_ratios": [1, 2]}
fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw=gspec)

axes[0].grid(False)
axes[0].boxplot(
    [best_values, mean_evaluations],
    tick_labels=["Reported Value", "True Value"],
)
axes[0].set_ylabel("Total distance")
axes[0].axhline(OPT_COST, label="Best known", ls="--", color="C2")

axes[1].grid()
axes[1].hist(
    best_values,
    bins=10,
    label="Reported Value",
    color="C0",
    alpha=0.6,
)
axes[1].hist(
    mean_evaluations,
    bins=10,
    label="True Value",
    color="C1",
    alpha=0.6,
)
axes[1].axvline(OPT_COST, label="Best known", ls="--", color="C2")
axes[1].set_xlabel("Total distance")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.legend()
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig("03_evobandits_results.pdf")
