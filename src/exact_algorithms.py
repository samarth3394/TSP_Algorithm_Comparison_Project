"""
exact_algorithms.py
===================

Exact (optimal) algorithms for the TSP.  All three accept symmetric **and**
asymmetric distance matrices and return ``(tour, cost)`` with a closed tour that
starts and ends at city 0.

Because the problem is NP-hard, every function has a ``max_n`` safety limit and
raises ``ValueError`` for larger instances instead of running for hours.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .approximation_algorithms import nearest_neighbor_tsp
from .distance_utils import is_symmetric, prim_mst, validate_distance_matrix

EPS = 1e-12


# ----------------------------------------------------------------------------
# 1. Brute force
# ----------------------------------------------------------------------------
def brute_force_tsp(D, max_n: int = 11):
    """Try every possible tour and keep the cheapest.

    City 0 is fixed as the start, so (n-1)! orderings exist.  For a symmetric
    matrix a tour and its reverse have the same cost, so only one of each pair
    is evaluated ((n-1)!/2 tours).

    Time  : O(n!)   (precisely O((n-1)! * n))      Space : O(n)
    """
    D = validate_distance_matrix(D, require_symmetric=False)
    n = D.shape[0]
    if n > max_n:
        raise ValueError(f"brute force limited to n <= {max_n} (got n = {n}); it needs (n-1)! tours")
    if n == 1:
        return [0, 0], 0.0
    dist = D.tolist()                      # plain lists are faster than NumPy for scalar access
    symmetric = is_symmetric(D)
    best_cost, best_perm = float("inf"), None
    for perm in itertools.permutations(range(1, n)):
        if symmetric and perm[0] > perm[-1]:
            continue                       # reverse of a tour already examined
        cost = dist[0][perm[0]]
        for a, b in zip(perm, perm[1:]):
            cost += dist[a][b]
        cost += dist[perm[-1]][0]
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return [0, *best_perm, 0], float(best_cost)


# ----------------------------------------------------------------------------
# 2. Held-Karp dynamic programming
# ----------------------------------------------------------------------------
def held_karp_tsp(D, max_n: int = 20):
    """Held-Karp dynamic programming (Bellman 1962; Held & Karp 1962).

    ``dp[S, j]`` = cost of the cheapest path that starts at city 0, visits exactly
    the cities in subset ``S`` (a bitmask over cities 1..n-1) and ends at ``j`` in ``S``.

        dp[{j}, j]  = D[0, j]
        dp[S, j]    = min over k in S\\{j} of  dp[S\\{j}, k] + D[k, j]
        OPT         = min over j of  dp[all, j] + D[j, 0]

    Time  : O(2^n * n^2)       Space : O(2^n * n)
    """
    D = validate_distance_matrix(D, require_symmetric=False)
    n = D.shape[0]
    if n > max_n:
        raise ValueError(f"Held-Karp limited to n <= {max_n} (got n = {n}); memory grows as 2^n * n")
    if n == 1:
        return [0, 0], 0.0
    m = n - 1                              # cities 1..n-1 are stored as bits 0..m-1
    full = 1 << m
    C = D[1:, 1:]                          # C[k, j] = cost from city k+1 to city j+1
    dp = np.full((full, m), np.inf)
    parent = np.full((full, m), -1, dtype=np.int16)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    bit_index = np.arange(m)
    for mask in range(1, full):
        if mask & (mask - 1) == 0:         # single city: base case already filled
            continue
        js = np.flatnonzero((mask >> bit_index) & 1)   # cities contained in the subset
        prevs = mask ^ (1 << js)                        # the subset without j
        cand = dp[prevs] + C[:, js].T                   # cand[a, k] = dp[prev_a, k] + C[k, js[a]]
        ks = cand.argmin(axis=1)
        dp[mask, js] = cand[np.arange(len(js)), ks]
        parent[mask, js] = ks
    closing = dp[full - 1] + D[1:, 0]
    last = int(np.argmin(closing))
    best_cost = float(closing[last])
    # walk the parent pointers backwards to rebuild the tour
    order, mask, j = [], full - 1, last
    while mask:
        order.append(j + 1)
        k = int(parent[mask, j])
        mask ^= 1 << j
        j = k
    return [0, *order[::-1], 0], best_cost


# ----------------------------------------------------------------------------
# 3. Branch and bound
# ----------------------------------------------------------------------------
def _lower_bound(dist, cur, cost, remaining):
    """Valid lower bound on the cost of any complete tour extending the partial path.

    The rest of the tour is a path  cur -> u1 -> ... -> uk -> 0  that leaves ``cur``
    and every remaining city exactly once.  Each of those k+1 leaving edges costs at
    least the cheapest edge available to that city, so

        bound = cost_so_far
              + min over u in remaining of D[cur, u]
              + sum over u in remaining of ( min over v in remaining U {0}, v != u of D[u, v] ).

    Every real completion pays at least this much, so the bound is never too large
    (it is *admissible*), and pruning with it can never discard the optimum.
    """
    first = min(dist[cur][u] for u in remaining)
    total = cost + first
    for u in remaining:
        row = dist[u]
        cheapest = row[0]
        for v in remaining:
            if v != u and row[v] < cheapest:
                cheapest = row[v]
        total += cheapest
    return total


def branch_and_bound_tsp(D, max_n: int = 18, time_limit: float | None = None, stats: dict | None = None):
    """Depth-first branch and bound.

    * The initial upper bound comes from the Nearest Neighbor tour.
    * Branches are explored closest-city-first so good tours are found early.
    * A branch is cut when ``lower_bound >= best cost found so far``.

    Parameters
    ----------
    max_n : int
        Safety limit on the number of cities.
    time_limit : float or None
        Seconds after which ``TimeoutError`` is raised (the benchmark records such
        runs as "timeout" rather than reporting a possibly non-optimal tour).
    stats : dict or None
        If a dict is given it is filled with ``nodes_expanded`` and ``nodes_pruned``.

    Time  : O(n!) worst case, usually far less     Space : O(n) recursion depth
    """
    D = validate_distance_matrix(D, require_symmetric=False)
    n = D.shape[0]
    if n > max_n:
        raise ValueError(f"branch and bound limited to n <= {max_n} (got n = {n})")
    if n == 1:
        if stats is not None:
            stats.update(nodes_expanded=0, nodes_pruned=0)
        return [0, 0], 0.0
    dist = D.tolist()
    nn_tour, nn_cost = nearest_neighbor_tsp(D, start=0)
    best = {"cost": nn_cost, "tour": nn_tour}
    counters = {"expanded": 0, "pruned": 0}
    deadline = None if time_limit is None else time.perf_counter() + time_limit

    def search(cur, cost, path, remaining):
        counters["expanded"] += 1
        if deadline is not None and counters["expanded"] % 1000 == 0 and time.perf_counter() > deadline:
            raise TimeoutError(f"branch and bound exceeded the {time_limit} s time limit")
        if not remaining:
            total = cost + dist[cur][0]
            if total < best["cost"] - EPS:
                best["cost"], best["tour"] = total, path + [0]
            return
        if _lower_bound(dist, cur, cost, remaining) >= best["cost"] - EPS:
            counters["pruned"] += 1
            return
        for u in sorted(remaining, key=lambda c: dist[cur][c]):
            new_cost = cost + dist[cur][u]
            if new_cost >= best["cost"] - EPS:     # distances are >= 0, so cost can only grow
                counters["pruned"] += 1
                continue
            search(u, new_cost, path + [u], [c for c in remaining if c != u])

    try:
        search(0, 0.0, [0], list(range(1, n)))
    finally:
        if stats is not None:
            stats.update(nodes_expanded=counters["expanded"], nodes_pruned=counters["pruned"])
    return list(best["tour"]), float(best["cost"])


# ----------------------------------------------------------------------------
# Lower bound for instances too large to solve exactly
# ----------------------------------------------------------------------------
def one_tree_lower_bound(D) -> float:
    """Held-Karp style **1-tree** lower bound on the optimal tour (symmetric TSP).

    For a vertex v: remove v, take the MST of the remaining cities and add the two
    cheapest edges at v.  Any optimal tour minus v is a spanning path (>= MST) and
    uses two edges at v (>= the two cheapest), so this value never exceeds OPT.  The
    best (largest) value over all choices of v is returned.  It is used to report
    "tour cost / lower bound" for the medium and large datasets, where the true
    optimum is unknown.

    Time: O(n^3) (n Prim runs of O(n^2)).
    """
    D = validate_distance_matrix(D, require_symmetric=True)
    n = D.shape[0]
    if n == 1:
        return 0.0
    if n == 2:
        return float(2 * D[0, 1])
    best = 0.0
    for v in range(n):
        others = [i for i in range(n) if i != v]
        _, mst_weight = prim_mst(D[np.ix_(others, others)])
        two_cheapest = np.sort(D[v, others])[:2].sum()
        best = max(best, mst_weight + float(two_cheapest))
    return best
