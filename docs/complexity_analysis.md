# Complexity Analysis

`n` = number of cities. Time and space are for the implementations in `src/`.

| Algorithm | Time | Space | Solution quality |
|-----------|------|-------|------------------|
| Brute Force | `O((n-1)! * n)` (about `(n-1)!/2` tours for symmetric `D`) | `O(n)` | optimal |
| Held-Karp DP | `O(2^n * n^2)` | `O(2^n * n)` | optimal |
| Branch and Bound | `O(n!)` worst case; each node costs `O(k^2)` for `k` unvisited cities; typically far fewer nodes | `O(n)` recursion depth (+ `O(n^2)` for the matrix copy) | optimal |
| Nearest Neighbor | `O(n^2)` | `O(n)` | no constant guarantee; worst-case ratio `Theta(log n)` on metric instances |
| MST preorder | `O(n^2)` (Prim on a dense graph) | `O(n)` | `<= 2 * OPT` (symmetric metric) |
| Christofides | `O(n^3)` | `O(n^2)` | `<= 1.5 * OPT` (symmetric metric) |
| 1-tree lower bound | `O(n^3)` (n Prim runs) | `O(n^2)` | `<= OPT` (symmetric) |

(The input matrix itself needs `O(n^2)` memory for every algorithm; the "Space" column counts the extra working memory.)

## Derivations

**Brute force.** There are `(n-1)!` permutations of the cities other than the start; evaluating each costs `O(n)`.
For a symmetric matrix a tour and its reverse are equal, halving the work but not changing the order of growth.

**Held-Karp.** There are `2^(n-1)` subsets of cities `1..n-1` and, for each, up to `n-1` possible end cities: `O(n 2^n)`
states. Each state minimises over up to `n-1` predecessors: `O(n)` work per state, total `O(n^2 2^n)`. The table stores
one value (and one parent pointer) per state: `O(n 2^n)` space. Compared with `n!`, `2^n n^2` is dramatically smaller
(for `n = 20`: about `4 * 10^8` versus `1.2 * 10^17`) but still exponential.

**Branch and Bound.** The search tree contains at most `sum_k (n-1)!/(n-1-k)!` nodes, i.e. the same factorial worst case as
brute force, plus bound computation. Pruning is effective when the bound is close to the true cost, which is why it is
often much faster in practice - but no polynomial bound exists for it. The recursion depth is `n`, giving `O(n)` stack
space (the implementation copies short lists along the path, which does not change this order).

**Nearest Neighbor.** `n-1` steps, each scanning at most `n` candidates: `O(n^2)`. State: a visited array and the tour: `O(n)`.

**MST preorder.** Prim's algorithm on a dense graph: `n` iterations, each `O(n)`: `O(n^2)`. Preorder DFS: `O(n)`.

**Christofides.** MST `O(n^2)`; the number of odd-degree vertices is at most `n`; a minimum-weight perfect matching on `m <= n`
vertices with a blossom algorithm takes `O(m^3)`, which dominates; Euler circuit `O(n)` edges; shortcutting `O(n)`.
The complete graph over the odd vertices stores `O(n^2)` edges.

## Practical size limits used in this project

| Algorithm | Limit in benchmark | Reason |
|-----------|-------------------:|--------|
| Brute Force | `n <= 11` | grows by a factor `n` per added city; the 11-city run already takes seconds |
| Held-Karp | `n <= 18` | doubling per added city in time **and** memory |
| Branch and Bound | `n <= 20`, plus a 120 s time limit | worst-case factorial; instance dependent |
| Nearest Neighbor, MST preorder, Christofides | `n <= 200` in the experiments | polynomial; limited only by the chosen dataset |

The functions themselves raise `ValueError` for inputs above their `max_n` argument, so a very large instance cannot be
started by accident; the limit can be raised deliberately.
