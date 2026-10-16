# Sample Outputs

Real excerpts from the final run (`results/*.csv`, notebook outputs and a fresh unit-test run). Values will differ slightly on other
machines (timings) but tour lengths are deterministic.

## Unit tests

```text
$ python -m unittest discover -s tests
Ran 66 tests in 2.131 s - OK
```

## Tours on the small dataset (11 cities, `results/tours.csv`)

```text
Brute Force       length   265.15   tour: 0 9 1 3 6 5 2 4 7 8 10 0
Held-Karp DP      length   265.15   tour: 0 10 8 7 4 2 5 6 3 1 9 0
Branch and Bound  length   265.15   tour: 0 10 8 7 4 2 5 6 3 1 9 0
Nearest Neighbor  length   281.68   tour: 0 10 9 1 3 6 5 2 4 7 8 0
MST Preorder      length   368.78   tour: 0 10 7 8 4 9 1 3 6 5 2 0
Christofides      length   265.15   tour: 0 9 1 3 6 5 2 4 7 8 10 0
```

## Small dataset comparison (`results/algorithm_comparison.csv`)

| Algorithm | Type | Tour length | Gap to optimum (%) | Time (s) | Peak memory (KiB) |
|---|---|---|---|---|---|
| Brute Force | Exact | 265.15 | 0.00 | 1.56 | 6.5 |
| Held-Karp DP | Exact | 265.15 | 0.00 | 0.0150 | 107.2 |
| Branch and Bound | Exact | 265.15 | 0.00 | 0.0043 | 5.9 |
| Nearest Neighbor | Heuristic | 281.68 | 6.24 | 0.0001 | 2.6 |
| MST Preorder | 2-approximation (metric) | 368.78 | 39.08 | 0.0002 | 4.9 |
| Christofides | 1.5-approximation (metric) | 265.15 | 0.00 | 0.0009 | 34.7 |

## Medium and large datasets

| Dataset | Algorithm | Tour length | Length / lower bound | Time (s) | Peak memory (KiB) |
|---|---|---|---|---|---|
| medium | Nearest Neighbor | 698.74 | 1.383 | 0.0002 | 3.4 |
| medium | MST Preorder | 786.06 | 1.556 | 0.0013 | 63.0 |
| medium | Christofides | 653.12 | 1.293 | 0.0088 | 271.3 |
| large | Nearest Neighbor | 1344.76 | 1.363 | 0.0009 | 39.3 |
| large | MST Preorder | 1429.17 | 1.448 | 0.0159 | 938.6 |
| large | Christofides | 1238.03 | 1.255 | 0.4345 | 3115.2 |

## Optimality gaps over 50 random instances (`results/optimality_gap_results.csv`)

| Algorithm | Mean gap (%) | Median gap (%) | Worst gap (%) |
|---|---|---|---|
| Christofides | 5.17 | 4.53 | 20.58 |
| MST Preorder | 15.13 | 16.02 | 32.06 |
| Nearest Neighbor | 8.17 | 5.93 | 47.46 |

| Algorithm | Largest observed tour / optimum | Guaranteed factor |
|---|---|---|
| Christofides | 1.206 | 1.5 |
| MST Preorder | 1.321 | 2.0 |

## Scalability: execution time in seconds (`results/scalability_results.csv`)

| n | Brute Force | Held-Karp DP | Branch and Bound | Nearest Neighbor | MST Preorder | Christofides |
|---|---|---|---|---|---|---|
| 5 | 0.0001 | 0.0003 | 0.0001 | 0.0001 | 0.0002 | 0.0005 |
| 8 | 0.0018 | 0.0020 | 0.0008 | 0.0001 | 0.0002 | 0.0008 |
| 10 | 0.1494 | 0.0073 | 0.0041 | 0.0001 | 0.0002 | 0.0012 |
| 11 | 1.64 | 0.0157 | 0.0156 | 0.0002 | 0.0003 | 0.0014 |
| 12 | - | 0.0304 | 0.0247 | 0.0001 | 0.0004 | 0.0012 |
| 14 | - | 0.1325 | 0.0698 | 0.0001 | 0.0003 | 0.0009 |
| 16 | - | 0.5493 | 0.3107 | 0.0001 | 0.0004 | 0.0012 |
| 17 | - | 1.06 | 0.5708 | 0.0002 | 0.0004 | 0.0010 |
| 18 | - | 2.06 | 2.65 | 0.0001 | 0.0004 | 0.0018 |
| 20 | - | - | 7.44 | 0.0001 | 0.0004 | 0.0023 |
| 30 | - | - | - | 0.0003 | 0.0011 | 0.0096 |
| 50 | - | - | - | 0.0002 | 0.0011 | 0.0079 |
| 100 | - | - | - | 0.0004 | 0.0033 | 0.0354 |
| 150 | - | - | - | 0.0006 | 0.0080 | 0.1256 |
| 200 | - | - | - | 0.0008 | 0.0163 | 0.4207 |

## Generated figures

* `results/figures/route_distance_comparison.png` - tour length per algorithm (small dataset)
* `results/figures/execution_time_comparison.png` - execution time per algorithm (small dataset, log scale)
* `results/figures/scalability_comparison.png` - time, memory and quality versus n
* `results/figures/optimality_gap_by_size.png` - mean and range of the gap versus n
* `results/figures/ratio_to_lower_bound_medium.png`, `ratio_to_lower_bound_large.png`
* `results/figures/route_visualizations/` - city plots and the tour drawn by each algorithm
