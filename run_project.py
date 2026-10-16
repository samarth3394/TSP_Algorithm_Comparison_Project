#!/usr/bin/env python3
"""
run_project.py - regenerate every dataset, result table and figure of the project.

Usage
-----
    python run_project.py            # full experiment suite (a few minutes)
    python run_project.py --quick    # reduced sizes/seeds, finishes in well under a minute
    python run_project.py --tests    # additionally run the unit tests first

Outputs
-------
    datasets/*.csv
    results/algorithm_comparison.csv, scalability_results.csv, optimality_gap_results.csv, tours.csv
    results/figures/*.png and results/figures/route_visualizations/*.png

All numbers come from actually running the algorithms; nothing is hard-coded.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw to files; no window needed
import matplotlib.pyplot as plt            # noqa: E402
import pandas as pd                        # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import benchmarking as bm                                             # noqa: E402
from src import visualization as viz                                           # noqa: E402
from src.data_generator import generate_all_datasets                           # noqa: E402
from src.distance_utils import (                                               # noqa: E402
    coords_from_dataframe,
    euclidean_distance_matrix,
    is_symmetric,
    load_cities_csv,
    satisfies_triangle_inequality,
    validate_distance_matrix,
)

DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
ROUTES = FIGURES / "route_visualizations"


def slug(text):
    return text.lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smaller experiments for a fast smoke run")
    parser.add_argument("--tests", action="store_true", help="run the unit tests before the experiments")
    args = parser.parse_args()
    start_time = time.perf_counter()

    if args.tests:
        print("== Running unit tests ==")
        code = subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests")])
        if code != 0:
            sys.exit("Unit tests failed - stopping.")

    # 1. datasets -----------------------------------------------------------------
    print("== 1/5 Generating datasets ==")
    generate_all_datasets(DATASETS)
    cities = {name: load_cities_csv(DATASETS / f"cities_{name}.csv") for name in ("small", "medium", "large")}
    coords = {k: coords_from_dataframe(v) for k, v in cities.items()}
    matrices = {k: validate_distance_matrix(euclidean_distance_matrix(c)) for k, c in coords.items()}
    for k, D in matrices.items():
        print(f"   {k:<6} n={len(D):<4} symmetric={is_symmetric(D)} metric={satisfies_triangle_inequality(D)}")
        viz.plot_cities(coords[k], title=f"{k.capitalize()} dataset: {len(D)} cities",
                        save_path=ROUTES / f"cities_{k}.png")
        plt.close("all")

    # 2. head-to-head comparison --------------------------------------------------
    print("== 2/5 Algorithm comparison (small: all six algorithms; medium/large: polynomial-time ones) ==")
    comparison_parts, tours = [], []
    small_tours = {}
    for key in ("small", "medium", "large"):
        names = list(bm.ALGORITHMS) if key == "small" else bm.APPROX_NAMES
        df = bm.one_instance(matrices[key], algorithms=names, repeats=3, measure_memory=True)
        for _, r in df.iterrows():
            if r["status"] == "ok":
                tours.append({"dataset": key, "algorithm": r["algorithm"], "tour_cost": r["tour_cost"],
                              "tour": " ".join(map(str, r["tour"]))})
                if key == "small":
                    small_tours[r["algorithm"]] = (r["tour"], r["tour_cost"])
                else:
                    viz.plot_tour(coords[key], r["tour"], title=f"{r['algorithm']} - {key} dataset",
                                  cost=r["tour_cost"], color=viz.ALGO_COLORS[r["algorithm"]],
                                  annotate=False, save_path=ROUTES / f"{key}_{slug(r['algorithm'])}.png")
                    plt.close("all")
        df.insert(0, "dataset", key)
        comparison_parts.append(df.drop(columns=["tour"]))
    comparison = pd.concat(comparison_parts, ignore_index=True)
    comparison.to_csv(RESULTS / "algorithm_comparison.csv", index=False)
    pd.DataFrame(tours).to_csv(RESULTS / "tours.csv", index=False)

    for name, (tour, cost) in small_tours.items():
        viz.plot_tour(coords["small"], tour, title=f"{name} - small dataset", cost=cost,
                      color=viz.ALGO_COLORS[name], save_path=ROUTES / f"small_{slug(name)}.png")
        plt.close("all")
    viz.plot_tour_grid(coords["small"], small_tours, title="All algorithms on the small dataset (11 cities)",
                       save_path=ROUTES / "small_all_algorithms.png")
    small_df = comparison[comparison["dataset"] == "small"]
    viz.plot_route_distance_comparison(small_df, title="Route length by algorithm (small dataset, 11 cities)",
                                       save_path=FIGURES / "route_distance_comparison.png")
    viz.plot_execution_time_comparison(small_df, title="Execution time by algorithm (small dataset, 11 cities)",
                                       save_path=FIGURES / "execution_time_comparison.png")
    for key in ("medium", "large"):
        d = comparison[comparison["dataset"] == key]
        viz.plot_large_dataset_comparison(d, title=f"Tour length / lower bound ({key} dataset, {len(matrices[key])} cities)",
                                          save_path=FIGURES / f"ratio_to_lower_bound_{key}.png")
    plt.close("all")

    # 3. optimality gaps ----------------------------------------------------------
    print("== 3/5 Optimality-gap experiment (many random instances, Held-Karp optimum) ==")
    sizes = (6, 8, 10) if args.quick else (6, 8, 10, 12, 14)
    seeds = range(1, 4) if args.quick else range(1, 11)
    gaps = bm.run_optimality_gap_experiment(sizes=sizes, seeds=seeds)
    gaps.to_csv(RESULTS / "optimality_gap_results.csv", index=False)
    viz.plot_gap_by_size(gaps, save_path=FIGURES / "optimality_gap_by_size.png")
    plt.close("all")

    # 4. scalability --------------------------------------------------------------
    print("== 4/5 Scalability experiment (prefixes of the large dataset) ==")
    scal_sizes = [5, 8, 10, 11, 12, 14, 16, 18, 20, 30, 50, 100] if args.quick else None
    scal = bm.run_scalability_experiment(coords["large"], sizes=scal_sizes, repeats=3)
    scal.to_csv(RESULTS / "scalability_results.csv", index=False)
    viz.plot_scalability(scal, save_path=FIGURES / "scalability_comparison.png")
    plt.close("all")

    # 5. integrity summary --------------------------------------------------------
    print("== 5/5 Integrity checks ==")
    ok_rows = comparison[comparison["status"] == "ok"]
    assert ok_rows["valid_tour"].all(), "an algorithm returned an invalid tour"
    assert ((ok_rows["tour_cost"] - ok_rows["recomputed_cost"]).abs() < 1e-6).all(), "cost mismatch"
    assert gaps["valid_tour"].all() and scal.loc[scal["status"] == "ok", "valid_tour"].all()
    assert gaps["exact_methods_agree"].all(), "exact algorithms disagree"
    guarantee_rows = gaps.dropna(subset=["within_guarantee"])
    assert guarantee_rows["within_guarantee"].all(), "an approximation guarantee was violated"
    print("   all tours valid; recomputed costs match; exact algorithms agree on every instance;")
    print(f"   approximation factors respected on all {len(guarantee_rows)} guarantee checks.")
    print(f"\nDone in {time.perf_counter() - start_time:.1f} s.  Results in: {RESULTS}")


if __name__ == "__main__":
    main()
