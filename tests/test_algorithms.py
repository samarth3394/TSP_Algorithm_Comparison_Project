"""Correctness tests for the six TSP algorithms.

Run with either::

    python -m unittest discover -s tests -v
    python -m pytest tests            # if pytest is installed
"""

import itertools
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.approximation_algorithms import christofides_tsp, mst_preorder_tsp, nearest_neighbor_tsp  # noqa: E402
from src.data_generator import generate_cities, generate_sample_distance_matrix  # noqa: E402
from src.distance_utils import euclidean_distance_matrix  # noqa: E402
from src.exact_algorithms import (  # noqa: E402
    _lower_bound,
    branch_and_bound_tsp,
    brute_force_tsp,
    held_karp_tsp,
    one_tree_lower_bound,
)
from src.validation import canonical_tour, recompute_tour_cost, validate_tour  # noqa: E402

TOL = 1e-9
EXACT = [brute_force_tsp, held_karp_tsp, branch_and_bound_tsp]


def random_instance(n, seed):
    return euclidean_distance_matrix(generate_cities(n, seed=seed)[["x", "y"]].to_numpy())


class TestExactAlgorithmsAgree(unittest.TestCase):
    def test_all_exact_algorithms_give_same_optimal_cost(self):
        for n in range(2, 10):
            for seed in range(5):
                D = random_instance(n, seed)
                costs = []
                for algo in EXACT:
                    tour, cost = algo(D)
                    validate_tour(tour, n)
                    self.assertAlmostEqual(cost, recompute_tour_cost(D, tour), places=9)
                    costs.append(cost)
                with self.subTest(n=n, seed=seed):
                    self.assertAlmostEqual(costs[0], costs[1], delta=TOL * max(1, costs[0]))
                    self.assertAlmostEqual(costs[0], costs[2], delta=TOL * max(1, costs[0]))

    def test_equivalent_tours_are_recognised(self):
        """Different exact methods may return a rotated/reversed copy of the same tour."""
        D = random_instance(8, seed=21)
        tours = {canonical_tour(algo(D)[0]) for algo in EXACT}
        self.assertEqual(len(tours), 1)  # unique optimum for random real coordinates

    def test_known_optimum_unit_square(self):
        coords = [[0, 0], [1, 1], [1, 0], [0, 1]]      # corners given in a scrambled order
        D = euclidean_distance_matrix(coords)
        for algo in EXACT:
            with self.subTest(algo=algo.__name__):
                self.assertAlmostEqual(algo(D)[1], 4.0, places=9)

    def test_known_optimum_points_on_a_line(self):
        D = euclidean_distance_matrix([[0, 0], [3, 0], [1, 0], [7, 0], [5, 0]])
        for algo in EXACT:
            with self.subTest(algo=algo.__name__):
                self.assertAlmostEqual(algo(D)[1], 14.0, places=9)   # there and back: 2 * 7

    def test_asymmetric_matrix_exact_algorithms_agree(self):
        rng = np.random.default_rng(0)
        for n in (3, 5, 7):
            D = rng.integers(1, 50, size=(n, n)).astype(float)
            np.fill_diagonal(D, 0)
            costs = [algo(D)[1] for algo in EXACT]
            with self.subTest(n=n):
                self.assertAlmostEqual(costs[0], costs[1], places=9)
                self.assertAlmostEqual(costs[0], costs[2], places=9)

    def test_asymmetric_direction_matters(self):
        # cheap clockwise cycle 0->1->2->3->0 (cost 4); the reverse direction is expensive.
        D = np.full((4, 4), 10.0)
        np.fill_diagonal(D, 0)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            D[a, b] = 1.0
        for algo in EXACT:
            tour, cost = algo(D)
            with self.subTest(algo=algo.__name__):
                self.assertAlmostEqual(cost, 4.0)
                self.assertEqual(tour, [0, 1, 2, 3, 0])

    def test_duplicate_cities_zero_distances(self):
        D = euclidean_distance_matrix([[0, 0], [0, 0], [5, 0], [5, 0], [2, 3]])
        costs = [algo(D)[1] for algo in EXACT]
        self.assertAlmostEqual(costs[0], costs[1])
        self.assertAlmostEqual(costs[0], costs[2])

    def test_integer_matrix_sample_from_csv_generator(self):
        D = generate_sample_distance_matrix().to_numpy()
        costs = [algo(D)[1] for algo in EXACT]
        self.assertEqual(len(set(round(c, 9) for c in costs)), 1)


class TestEdgeCases(unittest.TestCase):
    def test_single_city(self):
        D = np.zeros((1, 1))
        for algo in EXACT + [nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp]:
            with self.subTest(algo=algo.__name__):
                self.assertEqual(algo(D), ([0, 0], 0.0))

    def test_two_cities(self):
        D = np.array([[0.0, 7.0], [7.0, 0.0]])
        for algo in EXACT + [nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp]:
            with self.subTest(algo=algo.__name__):
                tour, cost = algo(D)
                self.assertEqual(tour, [0, 1, 0])
                self.assertAlmostEqual(cost, 14.0)

    def test_three_cities_any_tour_is_optimal(self):
        D = euclidean_distance_matrix([[0, 0], [4, 0], [0, 3]])
        for algo in EXACT + [nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp]:
            with self.subTest(algo=algo.__name__):
                self.assertAlmostEqual(algo(D)[1], 12.0)

    def test_size_limits_raise_value_error(self):
        D = random_instance(8, seed=1)
        with self.assertRaises(ValueError):
            brute_force_tsp(D, max_n=7)
        with self.assertRaises(ValueError):
            held_karp_tsp(D, max_n=7)
        with self.assertRaises(ValueError):
            branch_and_bound_tsp(D, max_n=7)

    def test_invalid_start_city_rejected(self):
        D = random_instance(5, seed=1)
        for algo in (nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp):
            with self.subTest(algo=algo.__name__):
                with self.assertRaises(ValueError):
                    algo(D, start=5)
                with self.assertRaises(ValueError):
                    algo(D, start=-1)

    def test_approximations_start_where_requested(self):
        D = random_instance(9, seed=2)
        for algo in (nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp):
            tour, cost = algo(D, start=4)
            with self.subTest(algo=algo.__name__):
                validate_tour(tour, 9, start=4)
                self.assertAlmostEqual(cost, recompute_tour_cost(D, tour))

    def test_branch_and_bound_timeout_and_stats(self):
        D = random_instance(15, seed=3)
        with self.assertRaises(TimeoutError):
            branch_and_bound_tsp(D, time_limit=1e-9)
        stats = {}
        branch_and_bound_tsp(random_instance(8, seed=3), stats=stats)
        self.assertGreater(stats["nodes_expanded"], 0)
        self.assertIn("nodes_pruned", stats)


class TestBranchAndBoundBound(unittest.TestCase):
    def test_lower_bound_never_exceeds_best_completion(self):
        """The bound must be admissible: <= cost of the best real completion."""
        for D in (random_instance(7, seed=4), random_instance(7, seed=5)):
            dist = D.tolist()
            n = len(D)
            for prefix_len in (1, 2, 3, 4):
                for prefix in itertools.permutations(range(1, n), prefix_len - 1):
                    path = [0, *prefix]
                    cost = sum(dist[a][b] for a, b in zip(path, path[1:]))
                    remaining = [c for c in range(1, n) if c not in path]
                    best = min(
                        cost + dist[path[-1]][rest[0]]
                        + sum(dist[a][b] for a, b in zip(rest, rest[1:])) + dist[rest[-1]][0]
                        for rest in itertools.permutations(remaining)
                    )
                    bound = _lower_bound(dist, path[-1], cost, remaining)
                    self.assertLessEqual(bound, best + 1e-9)

    def test_lower_bound_admissible_for_asymmetric_matrix(self):
        rng = np.random.default_rng(8)
        D = rng.integers(1, 30, size=(6, 6)).astype(float)
        np.fill_diagonal(D, 0)
        dist = D.tolist()
        for prefix in itertools.permutations(range(1, 6), 2):
            path = [0, *prefix]
            cost = sum(dist[a][b] for a, b in zip(path, path[1:]))
            remaining = [c for c in range(1, 6) if c not in path]
            best = min(
                cost + dist[path[-1]][rest[0]]
                + sum(dist[a][b] for a, b in zip(rest, rest[1:])) + dist[rest[-1]][0]
                for rest in itertools.permutations(remaining)
            )
            self.assertLessEqual(_lower_bound(dist, path[-1], cost, remaining), best + 1e-9)

    def test_one_tree_bound_never_exceeds_optimum(self):
        for n in (3, 5, 8, 10):
            for seed in range(4):
                D = random_instance(n, seed)
                self.assertLessEqual(one_tree_lower_bound(D), held_karp_tsp(D)[1] + 1e-9)

    def test_one_tree_bound_small_cases(self):
        self.assertEqual(one_tree_lower_bound(np.zeros((1, 1))), 0.0)
        self.assertAlmostEqual(one_tree_lower_bound(np.array([[0, 3.0], [3.0, 0]])), 6.0)


class TestApproximationGuarantees(unittest.TestCase):
    def test_costs_never_below_optimum(self):
        for n in (5, 8, 10):
            for seed in range(5):
                D = random_instance(n, seed)
                opt = held_karp_tsp(D)[1]
                for algo in (nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp):
                    with self.subTest(n=n, seed=seed, algo=algo.__name__):
                        self.assertGreaterEqual(algo(D)[1], opt - 1e-9)

    def test_mst_preorder_within_factor_two_on_metric_instances(self):
        for n in (5, 8, 10, 12):
            for seed in range(8):
                D = random_instance(n, seed)
                opt = held_karp_tsp(D)[1]
                with self.subTest(n=n, seed=seed):
                    self.assertLessEqual(mst_preorder_tsp(D)[1], 2.0 * opt + 1e-9)

    def test_christofides_within_factor_one_and_a_half_on_metric_instances(self):
        for n in (5, 8, 10, 12):
            for seed in range(8):
                D = random_instance(n, seed)
                opt = held_karp_tsp(D)[1]
                with self.subTest(n=n, seed=seed):
                    self.assertLessEqual(christofides_tsp(D)[1], 1.5 * opt + 1e-9)

    def test_christofides_proof_inequalities(self):
        """MST <= OPT, matching <= OPT/2, and shortcut tour <= Euler walk = MST + matching."""
        for seed in range(6):
            D = random_instance(11, seed)
            opt = held_karp_tsp(D)[1]
            tour, cost, d = christofides_tsp(D, return_details=True)
            with self.subTest(seed=seed):
                self.assertLessEqual(d["mst_weight"], opt + 1e-9)
                self.assertLessEqual(d["matching_weight"], 0.5 * opt + 1e-9)
                self.assertAlmostEqual(d["euler_length"], d["mst_weight"] + d["matching_weight"], places=9)
                self.assertLessEqual(cost, d["euler_length"] + 1e-9)
                self.assertEqual(d["n_odd_vertices"] % 2, 0)

    def test_christofides_matching_is_minimum_weight(self):
        """Compare the matching weight with exhaustive search on the odd-degree vertices."""
        from src.distance_utils import prim_mst
        D = random_instance(12, seed=9)
        edges, _ = prim_mst(D)
        deg = np.zeros(len(D), dtype=int)
        for u, v, _ in edges:
            deg[u] += 1
            deg[v] += 1
        odd = [i for i in range(len(D)) if deg[i] % 2]

        def best_matching(vertices):
            if not vertices:
                return 0.0
            first, rest = vertices[0], vertices[1:]
            return min(D[first, r] + best_matching([x for x in rest if x != r]) for r in rest)

        _, _, details = christofides_tsp(D, return_details=True)
        self.assertAlmostEqual(details["matching_weight"], best_matching(odd), places=9)

    def test_mst_preorder_cost_at_most_twice_mst(self):
        from src.distance_utils import prim_mst
        D = random_instance(40, seed=2)
        _, mst_w = prim_mst(D)
        self.assertLessEqual(mst_preorder_tsp(D)[1], 2 * mst_w + 1e-9)

    def test_approximations_reject_asymmetric_matrix(self):
        D = np.array([[0, 1, 5], [2, 0, 1], [3, 4, 0]], dtype=float)
        with self.assertRaises(ValueError):
            mst_preorder_tsp(D)
        with self.assertRaises(ValueError):
            christofides_tsp(D)
        tour, cost = nearest_neighbor_tsp(D)       # Nearest Neighbor follows D[i, j] directly
        validate_tour(tour, 3)
        self.assertAlmostEqual(cost, recompute_tour_cost(D, tour))

    def test_non_metric_symmetric_matrix_warns_but_returns_valid_tour(self):
        D = np.array([[0, 1, 10, 4], [1, 0, 1, 9], [10, 1, 0, 1], [4, 9, 1, 0]], dtype=float)
        for algo in (mst_preorder_tsp, christofides_tsp):
            with self.subTest(algo=algo.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    tour, cost = algo(D)
                self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
                validate_tour(tour, 4)
                self.assertAlmostEqual(cost, recompute_tour_cost(D, tour))

    def test_metric_matrix_does_not_warn(self):
        D = random_instance(10, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mst_preorder_tsp(D)
            christofides_tsp(D)

    def test_nearest_neighbor_is_deterministic_and_greedy(self):
        D = euclidean_distance_matrix([[0, 0], [10, 0], [1, 0], [5, 0]])
        tour, _ = nearest_neighbor_tsp(D)
        self.assertEqual(tour, [0, 2, 3, 1, 0])


if __name__ == "__main__":
    unittest.main()
