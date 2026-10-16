"""
visualization.py
================

Matplotlib plots for cities, tours and benchmark results.

Every function accepts an optional ``save_path`` and returns the Matplotlib figure.
Colours come from the Okabe-Ito colour-blind-safe palette and each algorithm keeps
the same colour in every chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALGO_COLORS = {
    "Brute Force": "#000000",
    "Held-Karp DP": "#0072B2",
    "Branch and Bound": "#56B4E9",
    "Nearest Neighbor": "#D55E00",
    "MST Preorder": "#E69F00",
    "Christofides": "#009E73",
}
ALGO_MARKERS = {
    "Brute Force": "x", "Held-Karp DP": "o", "Branch and Bound": "s",
    "Nearest Neighbor": "^", "MST Preorder": "D", "Christofides": "v",
}


def _save(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_cities(coords, title="City locations", annotate=True, save_path=None, ax=None):
    """Scatter plot of the city coordinates (city 0 - the start - is highlighted)."""
    coords = np.asarray(coords)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.scatter(coords[1:, 0], coords[1:, 1], s=40, color="#0072B2", zorder=3, label="City")
    ax.scatter(coords[0, 0], coords[0, 1], s=110, color="#D55E00", marker="*", zorder=4,
               label="Start city (0)")
    if annotate and len(coords) <= 60:
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x coordinate (distance units)")
    ax.set_ylabel("y coordinate (distance units)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if own_fig:
        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_tour(coords, tour, title="Tour", cost=None, color="#0072B2", annotate=True,
              save_path=None, ax=None):
    """Draw a closed tour with arrows showing the direction of travel."""
    coords = np.asarray(coords)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    for a, b in zip(tour[:-1], tour[1:]):
        ax.annotate("", xy=coords[b], xytext=coords[a],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3,
                                "shrinkA": 3, "shrinkB": 3}, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], s=30, color="#333333", zorder=3)
    ax.scatter(coords[tour[0], 0], coords[tour[0], 1], s=120, color="#D55E00", marker="*",
               zorder=4, label=f"Start city ({tour[0]})")
    if annotate and len(coords) <= 30:
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    full_title = title if cost is None else f"{title}\nlength = {cost:.2f}"
    ax.set_title(full_title)
    ax.set_xlabel("x coordinate (distance units)")
    ax.set_ylabel("y coordinate (distance units)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if own_fig:
        fig.tight_layout()
        _save(fig, save_path)
    return fig


def plot_tour_grid(coords, results, title="Tours found by each algorithm", ncols=3, save_path=None):
    """Small-multiples figure: one panel per algorithm.

    ``results`` maps algorithm name -> (tour, cost).
    """
    names = list(results)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        tour, cost = results[name]
        plot_tour(coords, tour, title=name, cost=cost, color=ALGO_COLORS.get(name, "#0072B2"),
                  annotate=len(coords) <= 15, ax=ax)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_route_distance_comparison(df, title="Route length by algorithm", save_path=None):
    """Bar chart of tour length per algorithm; a dashed line marks the optimum if known."""
    d = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ALGO_COLORS.get(a, "#777777") for a in d["algorithm"]]
    bars = ax.bar(d["algorithm"], d["tour_cost"], color=colors)
    for bar, value in zip(bars, d["tour_cost"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center",
                va="bottom", fontsize=9)
    if "optimal_cost" in d and d["optimal_cost"].notna().any():
        opt = float(d["optimal_cost"].dropna().iloc[0])
        ax.axhline(opt, color="black", linestyle="--", linewidth=1, label=f"Optimal = {opt:.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Tour length (distance units)")
    ax.set_ylim(0, d["tour_cost"].max() * 1.12)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_execution_time_comparison(df, title="Execution time by algorithm", save_path=None):
    """Bar chart of execution time per algorithm (logarithmic axis)."""
    d = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ALGO_COLORS.get(a, "#777777") for a in d["algorithm"]]
    bars = ax.bar(d["algorithm"], d["time_s"], color=colors)
    for bar, value in zip(bars, d["time_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}s" if value < 1 else f"{value:.2f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Execution time (seconds, log scale)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_scalability(scal_df, title="Scalability: execution time and memory vs number of cities",
                     save_path=None):
    """Three panels: time vs n, peak memory vs n, tour length relative to the lower bound."""
    d = scal_df[scal_df["status"] == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, g in d.groupby("algorithm", sort=False):
        kw = {"color": ALGO_COLORS.get(name), "marker": ALGO_MARKERS.get(name, "o"),
              "label": name, "markersize": 5, "linewidth": 1.5}
        axes[0].plot(g["n"], g["time_s"], **kw)
        gm = g.dropna(subset=["peak_memory_kb"])
        axes[1].plot(gm["n"], gm["peak_memory_kb"], **kw)
        gr = g.dropna(subset=["ratio_to_lower_bound"])
        axes[2].plot(gr["n"], gr["ratio_to_lower_bound"], **kw)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_title("Execution time")
    axes[0].set_xlabel("Number of cities n (log scale)")
    axes[0].set_ylabel("Time (seconds, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_title("Peak extra memory (tracemalloc)")
    axes[1].set_xlabel("Number of cities n (log scale)")
    axes[1].set_ylabel("Peak memory (KiB, log scale)")
    axes[2].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[2].set_title("Tour length / 1-tree lower bound")
    axes[2].set_xlabel("Number of cities n")
    axes[2].set_ylabel("Ratio (1.0 = lower bound; optimum lies in between)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_gap_by_size(gap_df, title="Optimality gap of the approximation algorithms", save_path=None):
    """Mean optimality gap (with min-max range) per instance size for the approximation methods."""
    d = gap_df[gap_df["algorithm"].isin(["Nearest Neighbor", "MST Preorder", "Christofides"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, g in d.groupby("algorithm", sort=False):
        stats = g.groupby("n")["gap_pct"].agg(["mean", "min", "max"])
        ax.plot(stats.index, stats["mean"], color=ALGO_COLORS[name], marker=ALGO_MARKERS[name],
                label=f"{name} (mean)")
        ax.fill_between(stats.index, stats["min"], stats["max"], color=ALGO_COLORS[name], alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Number of cities n")
    ax.set_ylabel("Gap to optimal tour (%) - line = mean, band = min to max")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_large_dataset_comparison(df, title="Approximation algorithms on a larger dataset", save_path=None):
    """Bar chart of ``tour length / lower bound`` for datasets without a known optimum."""
    d = df[df["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ALGO_COLORS.get(a, "#777777") for a in d["algorithm"]]
    bars = ax.bar(d["algorithm"], d["ratio_to_lower_bound"], color=colors)
    for bar, value in zip(bars, d["ratio_to_lower_bound"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1-tree lower bound")
    ax.set_ylim(0.9, d["ratio_to_lower_bound"].max() * 1.06)
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Tour length / lower bound")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
