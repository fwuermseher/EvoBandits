import os
from pathlib import Path
from matplotlib.axes import Axes
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as np


# ---- File output utility ---- #
def _setup_output_dir(name):
    """Utility to setup a directory for file outputs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(script_dir) / name
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


RESULTS_DIR = _setup_output_dir("output")
PLOTS_DIR = _setup_output_dir("plots")


# ---- plotlib utility ---- #
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


# Patch pyplot and Axes with `tourplot`
_plt.tourplot = lambda tour, cities: _tourplot(  # type: ignore
    tour, cities
)
Axes.tourplot = lambda self, tour, cities: _tourplot(  # type: ignore
    tour, cities, ax=self,
)

# pyplot styling
_plt.style.use("seaborn-v0_8-whitegrid")
_mpl.rcParams["font.family"] = "serif"
_mpl.rcParams["font.serif"] = [
    "Computer Modern Roman",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
]

# Expose patched plt
plt = _plt
