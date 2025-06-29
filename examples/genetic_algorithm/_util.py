import os
import json
from pathlib import Path
from matplotlib.axes import Axes
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as np


# ---- File output utility ---- #
def _get_output_dir(name):
    """Utility to setup a directory for file outputs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / name
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def json_dump(results, filename):
    """Save results to local directory"""
    folder = _get_output_dir("results")
    with open(folder / filename, "w") as f:
        json.dump(results, f, indent=2)


def json_load(filename):
    """Open a file from the local results directory"""
    folder = _get_output_dir("results")
    with open(folder / filename, "r") as f:
        return json.load(f)


# ---- plotlib utility ---- #
def _savefig(self, filename):
    """Save the current figure to the results directory."""
    plots_folder = _get_output_dir("plots")
    self.savefig(plots_folder / filename)


def _tourplot(tour, cities, ax=None):
    """Plot a tour for the TSP instance on a 2D plane."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Plot the tour path
    tour_pts = cities[tour]
    tour_pts = np.vstack([tour_pts, tour_pts[0]])
    ax.plot(tour_pts[:, 0], tour_pts[:, 1], "-", zorder=1)

    # Scatter the cities
    ax.scatter(cities[:, 0], cities[:, 1], c="black", zorder=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")
    return ax


# Patch pyplot and Axes with `tourplot` and custom `savefig`
_plt.savefig = lambda self, filename: _savefig(self, filename)
_plt.tourplot = lambda tour, cities: _tourplot(  # type: ignore
    tour, cities
)
Axes.tourplot = lambda self, tour, cities: _tourplot(  # type: ignore
    tour, cities, ax=self,
)

# pyplot styling
_plt.style.use("petroff10")
_mpl.rcParams["font.family"] = "serif"
_mpl.rcParams["font.serif"] = [
    "Computer Modern Roman",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
]

# Expose patched plt
plt = _plt
