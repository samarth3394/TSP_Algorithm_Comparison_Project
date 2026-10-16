"""
benchmarking.py
===============

Runs the six algorithms on identical distance matrices and records, for each run:

* tour cost (as reported) and independently recomputed cost,
* validity of the tour,
* wall-clock time (``time.perf_counter``, median of several repeats for fast runs),
* peak extra memory (``tracemalloc``) - measured in a *separate* run so that the
  tracing overhead does not distort the timings.

Nothing here is simulated: every number in the result tables comes from actually
running the algorithms.
"""

from __future__ import annotations

import time
import tracemalloc
from statistics import median

import numpy as np
import pandas as pd

from .approximation_algorithms import christofides_tsp, mst_preorder_tsp, nearest_neighbor_tsp
from .data_generator import generate_cities
from .distance_utils import euclidean_distance_matrix, is_symmetric, validate_distance_matrix
from .exact_algorithms import (
    branch_and_bound_tsp,
    brute_force_tsp,
    held_karp_tsp,
    one_tree_lower_bound,
)
from .validation import is_valid_tour, recompute_tour_cost

# Algorithm registry -------------------------------------------------------
# "max_n" is the largest instance the *benchmark* runs for the algorithm.  The
# functions themselves have their own (looser) safety limits.
ALGORITHMS = {
    "Brute Force": {"func": brute_force_tsp, "type": "Exact", "max_n": 11},
    "Held-Karp DP": {"func": held_karp_tsp, "type": "Exact", "max_n": 18},
    "Branch and Bound": {"func": branch_and_bound_tsp, "type": "Exact", "max_n": 20},
    "Nearest Neighbor": {"func": nearest_neighbor_tsp, "type": "Heuristic", "max_n": None},
    "MST Preorder": {"func": mst_preorder_tsp, "type": "2-approximation (metric)", "max_n": None},
    "Christofides": {"func": christofides_tsp, "type": "1.5-approximation (metric)", "max_n": None},
}
EXACT_NAMES = [k for k, v in ALGORITHMS.items() if v["type"] == "Exact"]
APPROX_NAMES = [k for k in ALGORITHMS if k not in EXACT_NAMES]

BNB_TIME_LIMIT = 120.0          # seconds; a run that exceeds it is recorded as "timeout"
MEMORY_TIME_CUTOFF = 2.0        # tracemalloc slows code down a lot, so memory is only measured for runs faster than this (seconds)


def _call(name, D):
    """Run one algorithm and return (tour, cost); extra kwargs only where needed."""
    func = ALGORITHMS[name]["func"]
    if name == "Branch and Bound":
        return func(D, max_n=ALGORITHMS[name]["max_n"], time_limit=BNB_TIME_LIMIT)
    return func(D)


def run_algorithm(name: str, D, repeats: int = 3, measure_memory: bool = True) -> dict:
    """Run algorithm ``name`` on ``D`` and return a dict with the measurements.

    ``status`` is one of ``ok``, ``skipped (size limit)`` or ``timeout``.
    """
    spec = ALGORITHMS[name]
    n = len(D)
    row = {"algorithm": name, "type": spec["type"], "n": n, "status": "ok", "tour": None,
           "tour_cost": np.nan, "recomputed_cost": np.nan, "valid_tour": False,
           "time_s": np.nan, "peak_memory_kb": np.nan}
    if spec["max_n"] is not None and n > spec["max_n"]:
        row["status"] = "skipped (size limit)"
        return row

    # --- timing (no memory tracing active) ---
    times, tour, cost = [], None, None
    try:
        for r in range(repeats):
            t0 = time.perf_counter()
            tour, cost = _call(name, D)
            times.append(time.perf_counter() - t0)
            if times[-1] > 2.0:          # slow run: one measurement is enough
                break
    except TimeoutError:
        row["status"] = "timeout"
        return row
    row["time_s"] = median(times)
    row["tour"], row["tour_cost"] = tour, float(cost)
    row["valid_tour"] = bool(is_valid_tour(tour, n))
    row["recomputed_cost"] = recompute_tour_cost(np.asarray(D).tolist(), tour)

    # --- peak memory in a separate traced run ---
    if measure_memory and times[0] <= MEMORY_TIME_CUTOFF:
        tracemalloc.start()
        try:
            _call(name, D)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        row["peak_memory_kb"] = peak / 1024.0
    return row


def one_instance(D, algorithms=None, repeats: int = 3, measure_memory: bool = True) -> pd.DataFrame:
    """Run several algorithms on the same matrix; add optimum, gap and ratio columns.

    * ``optimal_cost`` is the cost found by the exact algorithms (if any ran);
      ``exact_agree`` says whether all exact algorithms that ran agree on it.
    * ``gap_pct``  = 100 * (cost - optimal) / optimal.
    * ``lower_bound`` is the 1-tree bound (symmetric instances) and
      ``ratio_to_lower_bound`` = cost / lower bound (an *upper bound* on cost/OPT).
    """
    D = validate_distance_matrix(D, require_symmetric=False)
    algorithms = list(algorithms or ALGORITHMS)
    rows = [run_algorithm(a, D, repeats=repeats, measure_memory=measure_memory) for a in algorithms]
    df = pd.DataFrame(rows)

    exact_ok = df[(df["type"] == "Exact") & (df["status"] == "ok")]
    if len(exact_ok):
        optimum = float(exact_ok["tour_cost"].min())
        agree = bool(np.allclose(exact_ok["tour_cost"], optimum, rtol=1e-9, atol=1e-9))
        df["optimal_cost"] = optimum
        df["exact_agree"] = agree
        df["gap_pct"] = 100.0 * (df["tour_cost"] - optimum) / optimum if optimum > 0 else 0.0
    else:
        df["optimal_cost"] = np.nan
        df["exact_agree"] = np.nan
        df["gap_pct"] = np.nan
    if is_symmetric(D) and len(D) >= 3:
        lb = one_tree_lower_bound(D)
        df["lower_bound"] = lb
        df["ratio_to_lower_bound"] = df["tour_cost"] / lb if lb > 0 else np.nan
    else:
        df["lower_bound"] = np.nan
        df["ratio_to_lower_bound"] = np.nan
    return df


def run_optimality_gap_experiment(sizes=(6, 8, 10, 12, 14), seeds=range(1, 11)) -> pd.DataFrame:
    """Many random instances per size; Held-Karp provides the optimum.

    Brute force (n <= 10) and branch and bound (n <= 12) are run too and compared with
    Held-Karp as an independent correctness check (``exact_methods_agree``).
    The approximation guarantees are checked per instance (``within_guarantee``).
    """
    rows = []
    for n in sizes:
        for seed in seeds:
            coords = generate_cities(n, seed=1000 * n + seed)[["x", "y"]].to_numpy()
            D = euclidean_distance_matrix(coords)
            hk = run_algorithm("Held-Karp DP", D, repeats=1, measure_memory=False)
            optimum = hk["tour_cost"]
            agree = True
            for name, limit in (("Brute Force", 10), ("Branch and Bound", 12)):
                if n <= limit:
                    other = run_algorithm(name, D, repeats=1, measure_memory=False)
                    agree = agree and abs(other["tour_cost"] - optimum) <= 1e-9 * max(1.0, optimum)
            for name in ["Held-Karp DP"] + APPROX_NAMES:
                res = hk if name == "Held-Karp DP" else run_algorithm(name, D, repeats=3, measure_memory=False)
                ratio = res["tour_cost"] / optimum
                guarantee = {"MST Preorder": 2.0, "Christofides": 1.5}.get(name)
                rows.append({
                    "n": n, "seed": 1000 * n + seed, "algorithm": name,
                    "tour_cost": res["tour_cost"], "optimal_cost": optimum,
                    "gap_pct": 100.0 * (ratio - 1.0), "ratio_to_optimal": ratio,
                    "guarantee_factor": guarantee if guarantee else np.nan,
                    "within_guarantee": (ratio <= guarantee + 1e-9) if guarantee else np.nan,
                    "valid_tour": res["valid_tour"], "time_s": res["time_s"],
                    "exact_methods_agree": agree,
                })
    return pd.DataFrame(rows)


def run_scalability_experiment(large_coords, sizes=None, repeats: int = 3) -> pd.DataFrame:
    """Run every algorithm on the first ``n`` cities of the large dataset for growing ``n``.

    Exact algorithms stop at their benchmark size limit (``ALGORITHMS[...]['max_n']``);
    those sizes are not run and therefore do not appear as rows.
    """
    sizes = list(sizes or [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 30, 50, 75, 100, 150, 200])
    rows = []
    for n in sizes:
        D = euclidean_distance_matrix(large_coords[:n])
        df = one_instance(D, repeats=repeats, measure_memory=True)
        df = df[df["status"] != "skipped (size limit)"]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out.drop(columns=["tour"])
