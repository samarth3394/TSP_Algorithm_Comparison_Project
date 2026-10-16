"""
distance_utils.py
=================

Helper functions for building, loading and checking distance matrices.

A distance matrix ``D`` is an ``n x n`` NumPy array where ``D[i, j]`` is the cost
of travelling directly from city ``i`` to city ``j``.

Tour convention used everywhere in this project
-----------------------------------------------
A *tour* is a **closed** list of city indices that starts and ends at the same
city, e.g. ``[0, 3, 1, 2, 0]`` for four cities.  A tour on ``n`` cities
therefore has ``n + 1`` entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Building distance matrices
# ----------------------------------------------------------------------------
def euclidean_distance_matrix(coords) -> np.ndarray:
    """Return the Euclidean distance matrix for an ``(n, d)`` array of points.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        One row per city (for this project ``d = 2``: x and y).

    Returns
    -------
    numpy.ndarray, shape (n, n)
        Symmetric matrix with a zero diagonal.
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 1:
        raise ValueError("coords must be a 2-D array with at least one row")
    if not np.all(np.isfinite(pts)):
        raise ValueError("coords contains NaN or infinite values")
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist


# ----------------------------------------------------------------------------
# Checking distance matrices
# ----------------------------------------------------------------------------
def is_symmetric(D, tol: float = 1e-9) -> bool:
    """True if ``D[i, j] == D[j, i]`` for all pairs (within ``tol``)."""
    D = np.asarray(D, dtype=float)
    return bool(np.allclose(D, D.T, atol=tol, rtol=0.0))


def satisfies_triangle_inequality(D, tol: float = 1e-9) -> bool:
    """True if ``D[i, j] <= D[i, k] + D[k, j]`` for all i, j, k (within ``tol``).

    This is the *metric* condition required by the approximation guarantees of
    the MST-preorder (2x) and Christofides (1.5x) algorithms.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    for k in range(n):
        # shortest route i -> k -> j must never beat the direct edge i -> j
        if np.any(D > D[:, k][:, None] + D[k, :][None, :] + tol):
            return False
    return True


def validate_distance_matrix(D, require_symmetric: bool = True, tol: float = 1e-9) -> np.ndarray:
    """Validate a distance matrix and return it as a float ``numpy.ndarray``.

    Checks performed
    ----------------
    * square 2-D shape with at least one city
    * only finite numbers (no NaN / inf)
    * no negative entries
    * zero diagonal
    * symmetry (only if ``require_symmetric`` is True)

    Raises
    ------
    ValueError
        With a message describing the first problem found.
    """
    try:
        arr = np.asarray(D, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"distance matrix is not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("distance matrix must contain at least one city")
    if not np.all(np.isfinite(arr)):
        raise ValueError("distance matrix contains NaN or infinite values")
    if np.any(arr < 0):
        raise ValueError("distance matrix contains negative distances")
    if not np.allclose(np.diag(arr), 0.0, atol=tol):
        raise ValueError("distance matrix diagonal must be zero")
    if require_symmetric and not is_symmetric(arr, tol):
        raise ValueError("distance matrix must be symmetric for this algorithm")
    return arr


# ----------------------------------------------------------------------------
# Tour helpers
# ----------------------------------------------------------------------------
def tour_length(D, tour) -> float:
    """Total length of a closed tour (consecutive pairs of ``tour``)."""
    D = np.asarray(D, dtype=float)
    return float(sum(D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


# ----------------------------------------------------------------------------
# Minimum spanning tree (Prim, O(n^2)) - shared by several algorithms
# ----------------------------------------------------------------------------
def prim_mst(D):
    """Minimum spanning tree of the complete graph given by ``D`` (Prim, O(n^2)).

    Returns
    -------
    edges : list of (u, v, weight)
    total_weight : float
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    key[0] = 0.0
    edges = []
    total = 0.0
    for _ in range(n):
        candidate = np.where(in_tree, np.inf, key)
        u = int(np.argmin(candidate))
        in_tree[u] = True
        if parent[u] >= 0:
            w = float(D[parent[u], u])
            edges.append((int(parent[u]), u, w))
            total += w
        better = (~in_tree) & (D[u] < key)
        key[better] = D[u][better]
        parent[better] = u
    return edges, total


# ----------------------------------------------------------------------------
# File input
# ----------------------------------------------------------------------------
def load_cities_csv(path) -> pd.DataFrame:
    """Load a city-coordinate CSV with columns ``city_id, x, y``."""
    df = pd.read_csv(path)
    missing = {"city_id", "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return df


def coords_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Return the ``(n, 2)`` coordinate array from a cities DataFrame."""
    return df[["x", "y"]].to_numpy(dtype=float)


def load_distance_matrix_csv(path):
    """Load a labelled distance-matrix CSV.

    Expected format: first row = city labels (after an empty/``city`` corner
    cell), first column = the same labels, remaining cells = distances.

    Returns
    -------
    D : numpy.ndarray, shape (n, n)
    labels : list of str
    """
    df = pd.read_csv(path, index_col=0)
    labels = [str(c) for c in df.columns]
    if [str(i) for i in df.index] != labels:
        raise ValueError("row labels and column labels of the matrix CSV must match")
    return df.to_numpy(dtype=float), labels
