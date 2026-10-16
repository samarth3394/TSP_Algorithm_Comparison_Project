"""
approximation_algorithms.py
===========================

Polynomial-time heuristics / approximation algorithms for the TSP.

All functions take a distance matrix ``D`` and return ``(tour, cost)`` where
``tour`` is a closed list of city indices (see ``distance_utils``).

Guarantees (what is and is not claimed)
---------------------------------------
* **MST preorder** - cost <= 2 * OPT, *only* for symmetric instances that satisfy
  the triangle inequality (metric TSP).
* **Christofides** - cost <= 1.5 * OPT, *only* for symmetric metric instances.
* **Nearest Neighbor** - no constant-factor guarantee.  For metric instances its
  worst-case ratio grows logarithmically with n (Rosenkrantz, Stearns & Lewis,
  1977); for general (non-metric) instances it can be arbitrarily bad.

Euclidean distance matrices are metric, so the guarantees hold for all
coordinate-based datasets of this project.  For custom matrices the functions
issue a warning if the triangle inequality is violated.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import networkx as nx
import numpy as np

from .distance_utils import (
    prim_mst,
    satisfies_triangle_inequality,
    tour_length,
    validate_distance_matrix,
)


def _warn_if_not_metric(D: np.ndarray, name: str) -> None:
    if not satisfies_triangle_inequality(D):
        warnings.warn(
            f"{name}: the distance matrix violates the triangle inequality, so the "
            "approximation guarantee does not apply (the returned tour is still valid).",
            RuntimeWarning,
            stacklevel=3,
        )


# ----------------------------------------------------------------------------
# 1. Nearest Neighbor heuristic
# ----------------------------------------------------------------------------
def nearest_neighbor_tsp(D, start: int = 0):
    """Nearest Neighbor heuristic: always travel to the closest unvisited city.

    Works for symmetric *and* asymmetric matrices (it follows ``D[current, next]``).

    Time  : O(n^2)       Space : O(n)
    Guarantee: none that is a universal constant (see module docstring).

    Parameters
    ----------
    D : array-like (n, n)
    start : int
        City where the tour begins and ends.

    Returns
    -------
    tour : list[int]   closed tour of length n + 1
    cost : float
    """
    D = validate_distance_matrix(D, require_symmetric=False)
    n = D.shape[0]
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n - 1}], got {start}")
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dist = np.where(visited, np.inf, D[current])
        nxt = int(np.argmin(dist))          # ties -> lowest index (deterministic)
        visited[nxt] = True
        tour.append(nxt)
        current = nxt
    tour.append(start)
    return tour, tour_length(D, tour)


# ----------------------------------------------------------------------------
# 2. MST-based preorder approximation (2-approximation for metric TSP)
# ----------------------------------------------------------------------------
def mst_preorder_tsp(D, start: int = 0, warn_non_metric: bool = True):
    """MST + preorder walk (the classic 2-approximation).

    Steps: (1) build a minimum spanning tree, (2) walk the tree depth-first from
    ``start`` and record cities in *preorder* (first visit), (3) return to start.
    Preorder skips repeated cities = "shortcutting" a doubled-MST Euler tour.

    Guarantee: cost <= 2 * OPT **if D is symmetric and satisfies the triangle
    inequality**.  Requires a symmetric matrix.

    Time  : O(n^2)   (Prim on a dense graph)     Space : O(n)
    """
    D = validate_distance_matrix(D, require_symmetric=True)
    n = D.shape[0]
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n - 1}], got {start}")
    if warn_non_metric:
        _warn_if_not_metric(D, "mst_preorder_tsp")
    if n == 1:
        return [start, start], 0.0

    edges, _ = prim_mst(D)
    adjacency = {i: [] for i in range(n)}
    for u, v, w in edges:
        adjacency[u].append((w, v))
        adjacency[v].append((w, u))
    for i in adjacency:                      # visit closer children first (deterministic)
        adjacency[i].sort()

    order, seen = [], set()
    stack = [start]
    while stack:                             # iterative DFS -> preorder
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for _, v in reversed(adjacency[u]):  # reversed so the closest child is popped first
            if v not in seen:
                stack.append(v)
    tour = order + [start]
    return tour, tour_length(D, tour)


# ----------------------------------------------------------------------------
# 3. Christofides algorithm (1.5-approximation for metric TSP)
# ----------------------------------------------------------------------------
def christofides_tsp(D, start: int = 0, return_details: bool = False, warn_non_metric: bool = True):
    """Christofides' algorithm.

    1. Build a minimum spanning tree T.
    2. Let O be the set of odd-degree vertices of T (always an even number).
    3. Find a minimum-weight perfect matching M on the complete graph over O
       (exact, via NetworkX's blossom implementation).
    4. Combine T and M into a connected multigraph in which every vertex has
       even degree, and find an Eulerian circuit.
    5. Shortcut the circuit (skip already-visited cities) to get a Hamiltonian tour.

    Guarantee: cost <= 1.5 * OPT **if D is symmetric and satisfies the triangle
    inequality**.  Requires a symmetric matrix.

    Time  : O(n^3)  (dominated by minimum-weight matching on up to n vertices)
    Space : O(n^2)

    Parameters
    ----------
    return_details : bool
        If True a third value is returned: a dict with the MST weight, matching
        weight, number of odd vertices and Euler-circuit length.  These let you
        check the proof's inequality  cost <= MST + matching  directly.
    """
    D = validate_distance_matrix(D, require_symmetric=True)
    n = D.shape[0]
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n - 1}], got {start}")
    if warn_non_metric:
        _warn_if_not_metric(D, "christofides_tsp")
    if n == 1:
        result = ([start, start], 0.0)
        details = {"mst_weight": 0.0, "matching_weight": 0.0, "n_odd_vertices": 0, "euler_length": 0.0}
        return (*result, details) if return_details else result
    if n == 2:
        tour = [start, 1 - start, start]
        cost = tour_length(D, tour)
        details = {"mst_weight": float(D[0, 1]), "matching_weight": 0.0,
                   "n_odd_vertices": 2, "euler_length": cost}
        return (tour, cost, details) if return_details else (tour, cost)

    # 1. minimum spanning tree
    mst_edges, mst_weight = prim_mst(D)

    # 2. odd-degree vertices
    degree = np.zeros(n, dtype=int)
    for u, v, _ in mst_edges:
        degree[u] += 1
        degree[v] += 1
    odd = [i for i in range(n) if degree[i] % 2 == 1]
    assert len(odd) % 2 == 0, "handshake lemma violated - cannot happen"

    # 3. minimum-weight perfect matching on the odd vertices
    matching_edges = []
    matching_weight = 0.0
    if odd:
        complete = nx.Graph()
        complete.add_nodes_from(odd)
        for i, j in combinations(odd, 2):
            complete.add_edge(i, j, weight=float(D[i, j]))
        matching = nx.min_weight_matching(complete, weight="weight")
        if 2 * len(matching) != len(odd):
            raise RuntimeError("matching is not perfect")  # complete graph on an even set always has one
        for i, j in matching:
            matching_edges.append((int(i), int(j)))
            matching_weight += float(D[i, j])

    # 4. Eulerian circuit of MST + matching (a multigraph: an edge may appear twice)
    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(range(n))
    for u, v, w in mst_edges:
        multigraph.add_edge(u, v, weight=w)
    for u, v in matching_edges:
        multigraph.add_edge(u, v, weight=float(D[u, v]))
    assert all(d % 2 == 0 for _, d in multigraph.degree()), "all degrees must be even"
    assert nx.is_connected(multigraph), "multigraph must be connected"
    circuit = list(nx.eulerian_circuit(multigraph, source=start))
    euler_walk = [u for u, _ in circuit] + [start]
    euler_length = float(sum(D[a, b] for a, b in zip(euler_walk[:-1], euler_walk[1:])))

    # 5. shortcut repeated vertices
    seen, tour = set(), []
    for v in euler_walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    tour.append(start)

    cost = tour_length(D, tour)
    if return_details:
        details = {
            "mst_weight": float(mst_weight),
            "matching_weight": float(matching_weight),
            "n_odd_vertices": len(odd),
            "euler_length": euler_length,
        }
        return tour, cost, details
    return tour, cost
