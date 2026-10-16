"""
validation.py
=============

Independent checks for TSP results.

``recompute_tour_cost`` deliberately re-implements the cost calculation with a
plain Python loop (it does not call ``distance_utils.tour_length``) so that it can
be used as an independent cross-check of what an algorithm reports.
"""

from __future__ import annotations

from typing import Sequence


class TourValidationError(ValueError):
    """Raised when a tour is not a valid closed Hamiltonian cycle."""


def validate_tour(tour: Sequence[int], n: int, start: int | None = None) -> bool:
    """Check that ``tour`` visits each of ``n`` cities exactly once and returns to the start.

    A valid tour is a closed list of ``n + 1`` entries: the first and last entry
    are equal, and the first ``n`` entries are a permutation of ``0..n-1``.

    Raises
    ------
    TourValidationError
        If any rule is broken.  Returns ``True`` otherwise.
    """
    if n < 1:
        raise TourValidationError("n must be at least 1")
    tour = list(tour)
    if len(tour) != n + 1:
        raise TourValidationError(f"tour must have n + 1 = {n + 1} entries, got {len(tour)}")
    if tour[0] != tour[-1]:
        raise TourValidationError("tour must return to its starting city")
    if start is not None and tour[0] != start:
        raise TourValidationError(f"tour must start at city {start}, starts at {tour[0]}")
    body = tour[:-1]
    if any((not isinstance(c, (int,)) and not hasattr(c, "__index__")) for c in body):
        raise TourValidationError("tour entries must be integers")
    if sorted(int(c) for c in body) != list(range(n)):
        raise TourValidationError("tour must visit every city exactly once")
    return True


def is_valid_tour(tour: Sequence[int], n: int, start: int | None = None) -> bool:
    """Boolean version of :func:`validate_tour` (never raises)."""
    try:
        return validate_tour(tour, n, start)
    except TourValidationError:
        return False


def recompute_tour_cost(D, tour: Sequence[int]) -> float:
    """Recalculate the cost of a closed tour edge by edge with plain Python."""
    total = 0.0
    for i in range(len(tour) - 1):
        a, b = int(tour[i]), int(tour[i + 1])
        total += float(D[a][b])
    return total


def verify_result(D, tour: Sequence[int], reported_cost: float, rel_tol: float = 1e-9) -> bool:
    """True if the tour is valid **and** its recomputed cost matches ``reported_cost``."""
    n = len(D)
    if not is_valid_tour(tour, n):
        return False
    recomputed = recompute_tour_cost(D, tour)
    return abs(recomputed - reported_cost) <= rel_tol * max(1.0, abs(recomputed))


def canonical_tour(tour: Sequence[int]) -> tuple:
    """Rotate/reflect a closed tour so equivalent tours compare equal.

    The canonical form starts at city 0 and, of the two directions, picks the
    one whose second city has the smaller index.  (Reflection is only an
    equivalence for *symmetric* instances.)
    """
    body = list(tour[:-1])
    k = body.index(0)
    body = body[k:] + body[:k]
    reversed_body = [body[0]] + body[:0:-1]
    best = min(body, reversed_body)
    return tuple(best + [best[0]])
