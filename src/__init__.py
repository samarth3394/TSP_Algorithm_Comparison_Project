"""TSP algorithm comparison project: exact and approximation algorithms."""

from .approximation_algorithms import christofides_tsp, mst_preorder_tsp, nearest_neighbor_tsp
from .distance_utils import (
    euclidean_distance_matrix,
    is_symmetric,
    satisfies_triangle_inequality,
    tour_length,
    validate_distance_matrix,
)
from .exact_algorithms import (
    branch_and_bound_tsp,
    brute_force_tsp,
    held_karp_tsp,
    one_tree_lower_bound,
)
from .validation import is_valid_tour, recompute_tour_cost, validate_tour, verify_result

__version__ = "1.0.0"

__all__ = [
    "brute_force_tsp", "held_karp_tsp", "branch_and_bound_tsp", "one_tree_lower_bound",
    "nearest_neighbor_tsp", "mst_preorder_tsp", "christofides_tsp",
    "euclidean_distance_matrix", "validate_distance_matrix", "is_symmetric",
    "satisfies_triangle_inequality", "tour_length",
    "validate_tour", "is_valid_tour", "recompute_tour_cost", "verify_result",
]
