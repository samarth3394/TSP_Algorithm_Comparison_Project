# Algorithm Explanations

Conventions: `D` is the `n x n` distance matrix, cities are `0 .. n-1`, city `0` is the start, and a *tour* is a
closed list such as `[0, 3, 1, 2, 0]`. Source code: `src/exact_algorithms.py` and `src/approximation_algorithms.py`.

---

## 1. Brute Force (exact)

**Idea.** Enumerate every possible order of the other `n-1` cities, compute the length of each tour, keep the shortest.

```text
best = infinity
for each permutation P of cities 1..n-1:
    if matrix is symmetric and P[0] > P[-1]: skip      # reverse of a tour already seen
    length = D[0][P[0]] + sum(D[P[i]][P[i+1]]) + D[P[-1]][0]
    if length < best: best = length; best_tour = [0] + P + [0]
return best_tour, best
```

* Number of tours examined: `(n-1)!` (or `(n-1)!/2` for symmetric matrices).
* Always optimal, trivially correct, and therefore used as the reference implementation in the tests.
* Infeasible beyond about 11-12 cities in pure Python (`max_n = 11` default).

---

## 2. Held-Karp dynamic programming (exact)

**Idea.** An optimal tour has *optimal sub-structure*: the best path that starts at city 0, visits a set `S` of
cities and ends at `j` consists of the best path through `S \ {j}` ending at some `k`, plus the edge `k -> j`.
Many orders of visiting the same set lead to the same state, so we store each state once.

```text
dp[{j}][j] = D[0][j]                                   for j = 1..n-1
dp[S][j]   = min over k in S, k != j of  dp[S \ {j}][k] + D[k][j]
answer     = min over j of  dp[all cities][j] + D[j][0]
```

* A subset `S` of cities `1..n-1` is stored as a bit mask, so the table has `2^(n-1) * (n-1)` entries.
* A `parent` table remembers the best `k` for each state; the tour is rebuilt by walking the parents backwards.
* The implementation vectorises the inner minimum with NumPy, which makes it fast enough for n up to ~18-20 in a few seconds.
* Works for asymmetric matrices too.

---

## 3. Branch and Bound (exact)

**Idea.** Build the tour city by city (depth-first). Before extending a partial tour, compute a **lower bound** on the
cost of any complete tour that starts with it. If that bound is at least as large as the best complete tour found so far,
no extension can improve the answer, so the whole branch is discarded.

**Lower bound used.** For a partial path ending at `cur` with cost `c`, and unvisited cities `R`, any completion
`cur -> u1 -> ... -> uk -> 0` leaves `cur` and every city of `R` exactly once, and each leaving edge costs at least the
cheapest edge available to that city:

```text
bound = c
      + min over u in R of D[cur][u]
      + sum over u in R of  min over v in (R ∪ {0}) \ {u} of D[u][v]
```

The bound is **admissible** (never larger than the real cost of the best completion), so pruning with it never removes the
optimal tour. This is verified by a unit test that compares the bound with exhaustive search for every partial path of
small instances (symmetric and asymmetric).

**Other details.** The initial upper bound is the Nearest Neighbor tour; children are visited closest-first so good tours
are found early; the search can be stopped with a `time_limit`; a `stats` dictionary reports expanded / pruned nodes.
Worst case is still factorial, but typical random instances need far fewer nodes than brute force.

---

## 4. Nearest Neighbor (heuristic)

```text
current = start; tour = [start]
repeat n-1 times:
    next = the unvisited city with the smallest D[current][next]
    append next; current = next
append start
```

Very fast (`O(n^2)`) and simple; its measured quality on this project's random Euclidean instances is reported in
`results/` and the project report, but it has **no universal constant approximation factor**: on general instances it can be arbitrarily bad, and even on metric instances its
worst-case ratio grows logarithmically with `n` (Rosenkrantz, Stearns & Lewis, 1977).

---

## 5. MST-based preorder approximation (2-approximation for metric TSP)

```text
T = minimum spanning tree of the complete graph          (Prim, O(n^2))
tour = cities of T in depth-first PREORDER, starting at the start city
append start
```

**Why `<= 2 * OPT` (symmetric + triangle inequality).**

1. Deleting one edge from the optimal tour leaves a spanning tree, so `w(T) <= OPT`.
2. Walking around the *doubled* tree (every edge twice) visits all cities and has length `2 w(T) <= 2 OPT`.
3. Preorder skips cities that were already visited. With the triangle inequality a direct edge is never longer than the
   detour it replaces, so shortcutting does not increase the length. Hence `L(preorder tour) <= 2 w(T) <= 2 OPT`.

Without the triangle inequality, step 3 fails and no constant guarantee holds.

---

## 6. Christofides algorithm (1.5-approximation for metric TSP)

```text
1. T  = minimum spanning tree
2. O  = vertices with odd degree in T                      (|O| is always even)
3. M  = minimum-weight perfect matching on the complete graph over O   (exact blossom algorithm, NetworkX)
4. H  = T ∪ M  (a multigraph: every vertex now has even degree, and it is connected)
5. E  = an Eulerian circuit of H                             (visits every edge once)
6. tour = E with repeated cities skipped (shortcutting)
```

**Why `<= 1.5 * OPT` (symmetric + triangle inequality).**

* `w(T) <= OPT` as above.
* The optimal tour, shortcut to the odd vertices `O`, is a cycle through `O` that splits into two perfect matchings of `O`;
  the cheaper of the two weighs at most `OPT / 2`, so `w(M) <= OPT / 2`.
* The Euler circuit has length `w(T) + w(M) <= 1.5 OPT`, and shortcutting does not make it longer (triangle inequality).

The code returns these quantities with `return_details=True`, and the unit tests check `w(T) <= OPT`, `w(M) <= OPT/2`,
`tour <= w(T) + w(M)`, and that the matching weight equals the weight found by exhaustive search on the odd vertices.
(The best known worst-case ratio for metric TSP is slightly below 1.5 - Karlin, Klein & Oveis Gharan, 2021 - but
Christofides remains the classical reference algorithm.)

---

## 7. 1-tree lower bound (used for the medium and large datasets)

For a vertex `v`: remove `v`, compute the MST of the remaining cities, add the two cheapest edges incident to `v`; take the
maximum over all `v`. Removing `v` from an optimal tour leaves a spanning path (at least as heavy as the MST) and the tour
uses two edges at `v` (at least as heavy as the two cheapest), so the value never exceeds `OPT`. Therefore
`tour / lower_bound >= tour / OPT`: the reported ratio is an upper bound on the true ratio to the optimum.
