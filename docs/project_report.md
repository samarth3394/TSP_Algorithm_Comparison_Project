# Different Exact and Approximation Algorithms for the Travelling Salesperson Problem (TSP)

**Project report** - all numerical results in this report are taken from the result files in `results/` produced by the
code in this project (final run). Timings are from one machine and will differ on yours.

## Abstract

The Travelling Salesperson Problem (TSP) asks for the shortest closed tour through a set of cities. It is NP-hard, so
exact methods are limited to small instances, while polynomial-time approximation algorithms scale to hundreds of cities at
the price of solution quality. This project implements three exact algorithms (Brute Force, Held-Karp dynamic programming,
Branch and Bound) and three polynomial-time algorithms (Nearest Neighbor, an MST-preorder 2-approximation, and Christofides'
1.5-approximation) in Python, generates reproducible synthetic Euclidean datasets (11, 50 and 200 cities), validates every
tour and independently recomputes its length, and compares the algorithms by tour length, optimality gap, execution time,
memory and scalability. All three exact algorithms returned identical optimal lengths on every instance on which they were
compared. Over 50 random instances (6, 8, 10, 12, 14 cities) the mean gap to the optimum was
5.17 % for Christofides, 8.17 % for Nearest Neighbor and
15.13 % for MST-preorder; the 2x and 1.5x guarantees were never violated on these metric instances.
The exact algorithms were practical up to roughly 11 (Brute Force), 18 (Held-Karp) and 20 (Branch and Bound) cities in
this implementation, whereas the three polynomial-time algorithms handled 200 cities in under 0.5 s.

## 1. Introduction

A salesperson must visit `n` cities exactly once and return to the start; the cost of travelling between cities is given by a
distance matrix. The problem models vehicle routing, delivery planning, circuit-board drilling, warehouse order picking,
telescope scheduling, genome assembly and robot path planning. Its decision version is NP-complete (Karp, 1972), so no
algorithm is known that finds optimal tours in polynomial time. This motivates a spectrum of methods: exact algorithms with
exponential running time, fast heuristics without guarantees, and approximation algorithms with provable guarantees.

### Problem statement and formulation
Let `V = {0, ..., n-1}` be the cities and `D = (d_ij)` the cost matrix. Find a permutation `pi` of `V` with `pi(0) = 0` minimising
`L(pi) = sum_k d(pi(k), pi(k+1)) + d(pi(n-1), 0)`. An equivalent integer program uses binary variables `x_ij` with
degree constraints (each city has exactly one outgoing and one incoming edge) and sub-tour elimination constraints.

### Assumptions
Symmetric costs `d_ij = d_ji`, `d_ij >= 0`, `d_ii = 0`; for the approximation guarantees also the triangle inequality
(*metric* TSP); a complete graph; the tour starts and ends at city 0. Exact algorithms and Nearest Neighbor also accept
asymmetric matrices.

## 2. Objectives

1. Implement Brute Force, Held-Karp, Branch and Bound, Nearest Neighbor, MST-preorder and Christofides as reusable functions.
2. Verify correctness: valid tours, independently recomputed lengths, agreement of the exact algorithms, tests of edge cases and invalid inputs.
3. Compare solution quality (optimality gap, ratio to a lower bound), time and memory on identical inputs.
4. Study scalability as the number of cities grows, and state precisely when each approximation guarantee holds.

## 3. Background and related work

* **Complexity.** Karp (1972) listed the Hamiltonian-cycle problem among his 21 NP-complete problems, from which the NP-hardness of the TSP follows. Without the triangle inequality no polynomial-time algorithm can approximate the TSP within any constant factor unless P = NP (Sahni & Gonzalez, 1976); the metric TSP is APX-hard (Papadimitriou & Yannakakis, 1993).
* **Exact methods.** Bellman (1962) and Held & Karp (1962) independently gave the `O(2^n n^2)` dynamic program. Little, Murty, Sweeney & Karel (1963) introduced branch and bound for the TSP. Modern exact solvers (branch-and-cut, e.g. Concorde) solve instances with thousands of cities using linear-programming relaxations (Applegate, Bixby, Chvatal & Cook, 2006).
* **Lower bounds.** Held & Karp (1970) related the TSP to minimum spanning trees and introduced the 1-tree bound used here to evaluate instances whose optimum is unknown.
* **Heuristics.** Nearest Neighbor was analysed by Rosenkrantz, Stearns & Lewis (1977): no constant ratio in general and `Theta(log n)` worst-case ratio on metric instances.
* **Approximation.** The MST-preorder method gives a factor 2 for metric instances (see e.g. Cormen et al., *Introduction to Algorithms*, Ch. 35). Christofides (1976) improved this to 1.5 using minimum-weight matching (Edmonds, 1965). Karlin, Klein & Oveis Gharan (2021) obtained a ratio slightly below 1.5. For Euclidean instances polynomial-time approximation schemes exist (Arora, 1998).

## 4. Methodology

### 4.1 Algorithms
Full descriptions, pseudo-code and correctness arguments are in `docs/algorithm_explanations.md`; complexities are in `docs/complexity_analysis.md`.

| Algorithm | Kind | Time | Space | Guarantee |
|---|---|---|---|---|
| Brute Force | exact | O(n!) | O(n) | optimal |
| Held-Karp DP | exact | O(2^n n^2) | O(2^n n) | optimal |
| Branch and Bound | exact | O(n!) worst case | O(n) | optimal |
| Nearest Neighbor | heuristic | O(n^2) | O(n) | no constant factor |
| MST preorder | approximation | O(n^2) | O(n) | <= 2 OPT (symmetric metric) |
| Christofides | approximation | O(n^3) | O(n^2) | <= 1.5 OPT (symmetric metric) |

The Branch-and-Bound lower bound is *cost so far + cheapest edge leaving the current city + the cheapest usable edge leaving every
unvisited city*; it is admissible, i.e. never larger than the cost of any completion (checked by a unit test against exhaustive enumeration).
For instances without a known optimum, the **1-tree lower bound** (best over all vertices `v` of: MST of the other cities + the two cheapest edges at `v`) is used; the ratio
`tour / lower bound` is then an *upper bound* on the true ratio `tour / OPT`.

### 4.2 Datasets
Cities are uniformly random in `[0, 100]^2` with fixed seeds (`numpy.random.default_rng`) and rounded to two decimals:
`cities_small.csv` (11 cities, seed 42), `cities_medium.csv` (50, seed 2024), `cities_large.csv` (200, seed 7); plus a custom
6 x 6 Manhattan-distance matrix (`sample_distance_matrix.csv`). Distances are Euclidean, so the matrices are symmetric and metric
(checked in the notebook). Details: `datasets/dataset_README.md`.

### 4.3 Validation
`validate_tour` checks that a tour has `n+1` entries, begins and ends at the same city and visits every city exactly once.
`recompute_tour_cost` recomputes the length with an independent plain-Python loop. Matrices are validated for squareness, finiteness, non-negativity, zero diagonal and (where required) symmetry. The test-suite (66 tests) additionally checks edge cases (1, 2, 3 cities, duplicate cities), invalid matrices, size limits, and cross-checks distance matrices and MST weights against SciPy.

## 5. Implementation

Python, NumPy, pandas, Matplotlib, NetworkX (minimum-weight matching and Eulerian circuit in Christofides) and SciPy (used in the tests to cross-check
our distance matrix and MST). Code layout: `src/data_generator.py`, `distance_utils.py`, `exact_algorithms.py`, `approximation_algorithms.py`,
`validation.py`, `visualization.py`, `benchmarking.py`; `TSP_Algorithm_Comparison.ipynb` is the main working document; `run_project.py` regenerates all outputs.
All algorithms take the same distance matrix and return `(tour, cost)` with a closed tour starting at city 0.

## 6. Experimental setup

* **Timing:** `time.perf_counter`, median of up to 3 repeats (a single run once a run takes over 2 s).
* **Memory:** peak of `tracemalloc` during one separate call (Python and NumPy allocations); recorded only for runs shorter than 2 s because tracing slows execution.
* **Gap:** `100 (L - OPT) / OPT`, where OPT comes from the exact algorithms. For the optimality-gap experiment Held-Karp gives OPT and is cross-checked by Brute Force (n <= 10) and Branch and Bound (n <= 12).
* **Experiments:** (a) all six algorithms on the 11-city dataset; (b) the three polynomial-time algorithms on the 50- and 200-city datasets; (c) 50 random instances (6, 8, 10, 12, 14 cities, 10 seeds each); (d) scalability on the first `n` cities of the large dataset.
* **Size limits used in the benchmark:** Brute Force n <= 11, Held-Karp n <= 18, Branch and Bound n <= 20 (120 s time limit); the polynomial-time algorithms run up to n = 200.
* **Unit tests:** `Ran 66 tests in 2.131 s - OK`.

## 7. Results

### 7.1 Small dataset (11 cities), all six algorithms
Optimal tour length (all three exact algorithms agree): **265.15**.

| Algorithm | Type | Tour length | Gap to optimum (%) | Time (s) | Peak memory (KiB) |
|---|---|---|---|---|---|
| Brute Force | Exact | 265.15 | 0.00 | 1.56 | 6.5 |
| Held-Karp DP | Exact | 265.15 | 0.00 | 0.0150 | 107.2 |
| Branch and Bound | Exact | 265.15 | 0.00 | 0.0043 | 5.9 |
| Nearest Neighbor | Heuristic | 281.68 | 6.24 | 0.0001 | 2.6 |
| MST Preorder | 2-approximation (metric) | 368.78 | 39.08 | 0.0002 | 4.9 |
| Christofides | 1.5-approximation (metric) | 265.15 | 0.00 | 0.0009 | 34.7 |

### 7.2 Medium and large datasets (optimum unknown)
Ratio = tour length / 1-tree lower bound (an upper bound on tour / OPT).

| Dataset | Algorithm | Tour length | Length / lower bound | Time (s) | Peak memory (KiB) |
|---|---|---|---|---|---|
| medium | Nearest Neighbor | 698.74 | 1.383 | 0.0002 | 3.4 |
| medium | MST Preorder | 786.06 | 1.556 | 0.0013 | 63.0 |
| medium | Christofides | 653.12 | 1.293 | 0.0088 | 271.3 |
| large | Nearest Neighbor | 1344.76 | 1.363 | 0.0009 | 39.3 |
| large | MST Preorder | 1429.17 | 1.448 | 0.0159 | 938.6 |
| large | Christofides | 1238.03 | 1.255 | 0.4345 | 3115.2 |

### 7.3 Optimality gap on 50 random instances
Gap to the Held-Karp optimum. All exact algorithms agreed and all tours were valid on every instance.

| Algorithm | Mean gap (%) | Median gap (%) | Worst gap (%) |
|---|---|---|---|
| Christofides | 5.17 | 4.53 | 20.58 |
| MST Preorder | 15.13 | 16.02 | 32.06 |
| Nearest Neighbor | 8.17 | 5.93 | 47.46 |

Mean gap by number of cities:

| n | Christofides | MST Preorder | Nearest Neighbor |
|---|---|---|---|
| 6 | 1.96 | 4.33 | 2.18 |
| 8 | 1.92 | 16.39 | 5.49 |
| 10 | 7.23 | 9.36 | 6.39 |
| 12 | 7.94 | 22.15 | 13.02 |
| 14 | 6.77 | 23.44 | 13.77 |

Guarantee check (100 algorithm-instance checks, 0 violated):

| Algorithm | Largest observed tour / optimum | Guaranteed factor |
|---|---|---|
| Christofides | 1.206 | 1.5 |
| MST Preorder | 1.321 | 2.0 |

### 7.4 Scalability
Execution time in seconds (the first `n` cities of the large dataset; "-" = not run because of the size limit):

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

Peak extra memory in KiB (`-` = not run or not measured because the run took more than 2 s):

| n | Brute Force | Held-Karp DP | Branch and Bound | Nearest Neighbor | MST Preorder | Christofides |
|---|---|---|---|---|---|---|
| 5 | 2.9 | 5.7 | 3.0 | 2.4 | 2.6 | 20.0 |
| 8 | 4.1 | 14.6 | 3.8 | 2.4 | 3.5 | 29.0 |
| 10 | 5.2 | 51.7 | 5.2 | 2.5 | 4.4 | 46.4 |
| 11 | 6.5 | 107.2 | 5.8 | 2.5 | 4.9 | 48.8 |
| 12 | - | 227.7 | 7.6 | 2.5 | 5.5 | 49.1 |
| 14 | - | 1048.9 | 10.1 | 2.6 | 6.7 | 44.1 |
| 16 | - | 4810.2 | 13.0 | 2.6 | 8.2 | 46.9 |
| 17 | - | 10251.0 | 14.9 | 2.6 | 9.0 | 48.6 |
| 18 | - | - | - | 2.7 | 9.9 | 58.6 |
| 20 | - | - | - | 2.8 | 11.8 | 89.7 |
| 30 | - | - | - | 3.0 | 24.0 | 152.5 |
| 50 | - | - | - | 3.5 | 63.1 | 204.8 |
| 100 | - | - | - | 10.0 | 235.4 | 603.7 |
| 150 | - | - | - | 22.2 | 528.4 | 1379.5 |
| 200 | - | - | - | 39.3 | 938.6 | 3117.8 |

Branch and Bound versus brute-force enumeration (same instances):

| n | B&B nodes expanded | B&B nodes pruned | Brute-force tours (n-1)!/2 | B&B time (s) |
|---|---|---|---|---|
| 8 | 278 | 206 | 2520 | 0.0008 |
| 10 | 1329 | 1072 | 1.814e+05 | 0.0040 |
| 12 | 6707 | 5618 | 1.996e+07 | 0.0241 |
| 14 | 1.487e+04 | 1.286e+04 | 3.114e+09 | 0.0685 |
| 16 | 4.987e+04 | 4.366e+04 | 6.538e+11 | 0.2843 |

Figures: `results/figures/route_distance_comparison.png`, `execution_time_comparison.png`, `scalability_comparison.png`,
`optimality_gap_by_size.png`, `ratio_to_lower_bound_medium.png`, `ratio_to_lower_bound_large.png`, and the tour drawings in
`results/figures/route_visualizations/`.

## 8. Discussion

* **Exact algorithms.** All three returned the same optimum wherever they were compared. Brute Force needed 1.64 s for the 11-city
  instance, whereas Held-Karp needed 0.0157 s and Branch and Bound 0.0156 s: factorial growth versus exponential growth versus effective pruning.
  Held-Karp's time doubles roughly with every added city and its table memory (10251 KiB at n = 17) grows exponentially, whereas Branch and Bound
  needed only 15 KiB at n = 17. Branch and Bound was faster than Held-Karp for every size up to n = 17 on this nested family of instances (a near tie at n = 11) but slower at n = 18
  (2.65 s versus 2.06 s), and needed 7.4 s at n = 20; since its worst case is factorial and its running time depends on the instance, this should not be generalised.
* **Approximation algorithms.** They are many orders of magnitude cheaper: at n = 200 they needed 0.0008 s (Nearest Neighbor),
  0.0163 s (MST-preorder) and 0.42 s (Christofides, whose cubic-time matching step dominates).
* **Quality.** On the random instances Christofides had the lowest mean gap, Nearest Neighbor was in between and MST-preorder had the highest.
  On the 50- and 200-city datasets Christofides likewise had the smallest ratio to the lower bound. This is an empirical ordering on random uniform
  Euclidean instances, not a theorem: Nearest Neighbor has no constant worst-case guarantee, and its worst observed gap here (47.5 %) is larger than
  Christofides' (20.6 %). On the 11-city dataset Christofides happened to find the optimal tour; this is instance-specific.
* **Guarantees versus practice.** The largest observed ratios (MST-preorder 1.321, Christofides 1.206) are far below the worst-case
  factors 2 and 1.5, in line with the guarantees being worst-case statements.
* **Choosing a method.** Use an exact method when `n` is small and optimality matters; use Christofides when a guaranteed quality is needed for larger `n`; use Nearest Neighbor when speed dominates; improve any tour with local search (2-opt / Lin-Kernighan; not implemented here).

## 9. Limitations

* Only random uniform Euclidean instances were studied; clustered, real-road or asymmetric instances may behave differently.
* Timings are from a single machine and a pure-Python/NumPy implementation; memory is `tracemalloc`'s peak for one call, not process memory, and is missing for slower runs.
* The scalability study uses one nested family of instances per size; Branch and Bound in particular is instance dependent.
* Sample sizes of the gap experiment (50 instances) give indicative means and ranges, not rigorous statistics.
* No local-search improvement, metaheuristics or integer-programming solvers are included.
* The shipped notebook outputs were produced by a Python runner rather than a Jupyter kernel (Jupyter could not be installed in the build environment); the notebook should be re-run on the reader's machine.

## 10. Conclusion

Six TSP algorithms were implemented, tested (66 unit tests passing) and compared on identical inputs. Exact algorithms give optimal tours but are limited to roughly
10-20 cities because of factorial or exponential time/memory. Nearest Neighbor, MST-preorder and Christofides scale to hundreds of cities in well under a second
(Christofides 0.42 s at n = 200) with quality that - for symmetric metric instances - is guaranteed to be within factor 2 (MST-preorder) and 1.5 (Christofides) of the optimum
and was, in our experiments, much closer: Christofides gave the best average quality among the polynomial-time methods. The project thus illustrates the central trade-off between solution quality,
guarantees and computational effort in combinatorial optimisation.

## References

1. Karp, R. M. (1972). Reducibility among combinatorial problems. In R. E. Miller & J. W. Thatcher (Eds.), *Complexity of Computer Computations* (pp. 85-103). Plenum Press.
2. Bellman, R. (1962). Dynamic programming treatment of the travelling salesman problem. *Journal of the ACM*, 9(1), 61-63.
3. Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of the Society for Industrial and Applied Mathematics*, 10(1), 196-210.
4. Held, M., & Karp, R. M. (1970). The traveling-salesman problem and minimum spanning trees. *Operations Research*, 18(6), 1138-1162.
5. Little, J. D. C., Murty, K. G., Sweeney, D. W., & Karel, C. (1963). An algorithm for the traveling salesman problem. *Operations Research*, 11(6), 972-989.
6. Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977). An analysis of several heuristics for the traveling salesman problem. *SIAM Journal on Computing*, 6(3), 563-581.
7. Christofides, N. (1976). *Worst-case analysis of a new heuristic for the travelling salesman problem* (Report 388). Graduate School of Industrial Administration, Carnegie Mellon University.
8. Sahni, S., & Gonzalez, T. (1976). P-complete approximation problems. *Journal of the ACM*, 23(3), 555-565.
9. Papadimitriou, C. H., & Yannakakis, M. (1993). The traveling salesman problem with distances one and two. *Mathematics of Operations Research*, 18(1), 1-11.
10. Karlin, A. R., Klein, N., & Oveis Gharan, S. (2021). A (slightly) improved approximation algorithm for metric TSP. In *Proceedings of the 53rd Annual ACM Symposium on Theory of Computing (STOC '21)* (pp. 32-45).
11. Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *Journal of the ACM*, 45(5), 753-782.
12. Edmonds, J. (1965). Paths, trees, and flowers. *Canadian Journal of Mathematics*, 17, 449-467.
13. Prim, R. C. (1957). Shortest connection networks and some generalizations. *Bell System Technical Journal*, 36(6), 1389-1401.
14. Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.
15. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (chapter "Approximation Algorithms", section on the travelling-salesman problem). MIT Press.
16. Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy 2008)* (pp. 11-15).
17. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.
18. Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261-272.
