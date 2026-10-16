"""Tests for tour validation, matrix validation and dataset utilities.

Run with either::

    python -m unittest discover -s tests -v
    python -m pytest tests            # if pytest is installed
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.approximation_algorithms import christofides_tsp, mst_preorder_tsp, nearest_neighbor_tsp  # noqa: E402
from src.data_generator import generate_all_datasets, generate_cities  # noqa: E402
from src.distance_utils import (  # noqa: E402
    euclidean_distance_matrix,
    is_symmetric,
    load_cities_csv,
    load_distance_matrix_csv,
    satisfies_triangle_inequality,
    tour_length,
    validate_distance_matrix,
)
from src.exact_algorithms import branch_and_bound_tsp, brute_force_tsp, held_karp_tsp  # noqa: E402
from src.validation import (  # noqa: E402
    TourValidationError,
    canonical_tour,
    is_valid_tour,
    recompute_tour_cost,
    validate_tour,
    verify_result,
)

ALL_ALGORITHMS = [brute_force_tsp, held_karp_tsp, branch_and_bound_tsp,
                  nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp]


class TestValidateTour(unittest.TestCase):
    def test_valid_tour_accepted(self):
        self.assertTrue(validate_tour([0, 2, 1, 3, 0], 4))

    def test_wrong_length_rejected(self):
        with self.assertRaises(TourValidationError):
            validate_tour([0, 1, 2, 0], 4)

    def test_not_returning_to_start_rejected(self):
        with self.assertRaises(TourValidationError):
            validate_tour([0, 1, 2, 3, 1], 4)

    def test_repeated_city_rejected(self):
        with self.assertRaises(TourValidationError):
            validate_tour([0, 1, 1, 3, 0], 4)

    def test_missing_city_rejected(self):
        with self.assertRaises(TourValidationError):
            validate_tour([0, 1, 2, 5, 0], 4)

    def test_start_city_enforced(self):
        self.assertTrue(validate_tour([2, 0, 1, 2], 3, start=2))
        with self.assertRaises(TourValidationError):
            validate_tour([2, 0, 1, 2], 3, start=0)

    def test_single_city_tour(self):
        self.assertTrue(validate_tour([0, 0], 1))

    def test_is_valid_tour_never_raises(self):
        self.assertFalse(is_valid_tour([0, 1], 5))
        self.assertFalse(is_valid_tour([], 3))
        self.assertTrue(is_valid_tour([0, 1, 2, 0], 3))


class TestCostRecomputation(unittest.TestCase):
    def setUp(self):
        self.D = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=float)

    def test_recompute_matches_hand_calculation(self):
        self.assertAlmostEqual(recompute_tour_cost(self.D, [0, 1, 2, 0]), 1 + 2 + 4)

    def test_recompute_agrees_with_tour_length(self):
        tour = [0, 2, 1, 0]
        self.assertAlmostEqual(recompute_tour_cost(self.D, tour), tour_length(self.D, tour))

    def test_verify_result_accepts_correct_and_rejects_wrong_cost(self):
        self.assertTrue(verify_result(self.D, [0, 1, 2, 0], 7.0))
        self.assertFalse(verify_result(self.D, [0, 1, 2, 0], 6.0))

    def test_verify_result_rejects_invalid_tour(self):
        self.assertFalse(verify_result(self.D, [0, 1, 0], 2.0))

    def test_canonical_tour_identifies_rotations_and_reflections(self):
        a = canonical_tour([1, 2, 3, 0, 1])
        b = canonical_tour([0, 3, 2, 1, 0])
        c = canonical_tour([2, 1, 0, 3, 2])
        self.assertEqual(a, b)
        self.assertEqual(a, c)


class TestMatrixValidation(unittest.TestCase):
    def test_valid_matrix_returned_as_float_array(self):
        D = validate_distance_matrix([[0, 2], [2, 0]])
        self.assertEqual(D.dtype, float)

    def test_non_square_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix(np.zeros((3, 4)))

    def test_non_two_dimensional_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix([0, 1, 2])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix(np.zeros((0, 0)))

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix([[0, -1], [-1, 0]])

    def test_nonzero_diagonal_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix([[1, 2], [2, 0]])

    def test_nan_and_inf_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix([[0, np.nan], [np.nan, 0]])
        with self.assertRaises(ValueError):
            validate_distance_matrix([[0, np.inf], [np.inf, 0]])

    def test_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            validate_distance_matrix([["a", "b"], ["c", "d"]])

    def test_asymmetric_rejected_only_when_symmetry_required(self):
        asym = [[0, 1, 2], [5, 0, 1], [2, 3, 0]]
        with self.assertRaises(ValueError):
            validate_distance_matrix(asym, require_symmetric=True)
        validate_distance_matrix(asym, require_symmetric=False)  # no error

    def test_every_algorithm_rejects_invalid_matrix(self):
        bad = np.array([[0, -1, 2], [-1, 0, 1], [2, 1, 0]], dtype=float)
        for algo in ALL_ALGORITHMS:
            with self.subTest(algo=algo.__name__):
                with self.assertRaises(ValueError):
                    algo(bad)

    def test_euclidean_matrix_properties(self):
        coords = generate_cities(25, seed=3)[["x", "y"]].to_numpy()
        D = euclidean_distance_matrix(coords)
        self.assertEqual(D.shape, (25, 25))
        self.assertTrue(is_symmetric(D))
        self.assertTrue(np.all(np.diag(D) == 0))
        self.assertTrue(np.all(D >= 0))
        self.assertTrue(satisfies_triangle_inequality(D))

    def test_triangle_inequality_violation_detected(self):
        D = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]], dtype=float)
        self.assertFalse(satisfies_triangle_inequality(D))

    def test_euclidean_matrix_rejects_bad_coordinates(self):
        with self.assertRaises(ValueError):
            euclidean_distance_matrix([1, 2, 3])
        with self.assertRaises(ValueError):
            euclidean_distance_matrix([[0, 0], [np.nan, 1]])


class TestAllAlgorithmsReturnValidTours(unittest.TestCase):
    def test_valid_tours_and_correct_costs_on_random_instances(self):
        for n in (3, 4, 5, 7, 9):
            for seed in range(3):
                D = euclidean_distance_matrix(generate_cities(n, seed=seed)[["x", "y"]].to_numpy())
                for algo in ALL_ALGORITHMS:
                    with self.subTest(n=n, seed=seed, algo=algo.__name__):
                        tour, cost = algo(D)
                        self.assertTrue(validate_tour(tour, n))
                        self.assertEqual(tour[0], 0)
                        self.assertAlmostEqual(cost, recompute_tour_cost(D, tour), places=9)
                        self.assertTrue(verify_result(D, tour, cost))

    def test_larger_instance_for_polynomial_algorithms(self):
        D = euclidean_distance_matrix(generate_cities(60, seed=11)[["x", "y"]].to_numpy())
        for algo in (nearest_neighbor_tsp, mst_preorder_tsp, christofides_tsp):
            with self.subTest(algo=algo.__name__):
                tour, cost = algo(D)
                self.assertTrue(verify_result(D, tour, cost))

    def test_nearest_neighbor_other_start_city(self):
        D = euclidean_distance_matrix(generate_cities(8, seed=5)[["x", "y"]].to_numpy())
        tour, cost = nearest_neighbor_tsp(D, start=3)
        self.assertTrue(validate_tour(tour, 8, start=3))
        self.assertAlmostEqual(cost, recompute_tour_cost(D, tour))


class TestDatasets(unittest.TestCase):
    def test_generation_is_reproducible(self):
        a = generate_cities(20, seed=99)
        b = generate_cities(20, seed=99)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(generate_cities(20, seed=100)))

    def test_generate_cities_rejects_non_positive_n(self):
        with self.assertRaises(ValueError):
            generate_cities(0, seed=1)

    def test_generated_files_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_all_datasets(tmp)
            for stem, n in (("cities_small", 11), ("cities_medium", 50), ("cities_large", 200)):
                df = load_cities_csv(Path(tmp) / f"{stem}.csv")
                self.assertEqual(len(df), n)
                self.assertTrue(((df[["x", "y"]] >= 0) & (df[["x", "y"]] <= 100)).all().all())
            D, labels = load_distance_matrix_csv(Path(tmp) / "sample_distance_matrix.csv")
            self.assertEqual(D.shape, (6, 6))
            self.assertEqual(len(labels), 6)
            validate_distance_matrix(D)
            self.assertTrue(satisfies_triangle_inequality(D))

    def test_committed_datasets_match_generator(self):
        """The CSV files shipped with the project equal a fresh regeneration."""
        shipped = ROOT / "datasets"
        if not (shipped / "cities_small.csv").exists():
            self.skipTest("datasets not generated yet")
        with tempfile.TemporaryDirectory() as tmp:
            fresh = generate_all_datasets(tmp)
            for stem in ("cities_small", "cities_medium", "cities_large"):
                shipped_df = load_cities_csv(shipped / f"{stem}.csv")
                np.testing.assert_allclose(shipped_df[["x", "y"]].to_numpy(),
                                           fresh[stem][["x", "y"]].to_numpy())

    def test_cities_csv_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                load_cities_csv(path)

    def test_matrix_csv_label_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_matrix.csv"
            path.write_text("city,A,B\nA,0,1\nZ,1,0\n")
            with self.assertRaises(ValueError):
                load_distance_matrix_csv(path)


class TestAgainstSciPy(unittest.TestCase):
    """Independent cross-checks of our own building blocks using SciPy."""

    def test_euclidean_matrix_matches_scipy_pdist(self):
        from scipy.spatial.distance import pdist, squareform
        coords = generate_cities(40, seed=8)[["x", "y"]].to_numpy()
        np.testing.assert_allclose(euclidean_distance_matrix(coords), squareform(pdist(coords)), atol=1e-9)

    def test_prim_mst_weight_matches_scipy(self):
        from scipy.sparse.csgraph import minimum_spanning_tree
        from src.distance_utils import prim_mst
        for seed in range(5):
            D = euclidean_distance_matrix(generate_cities(35, seed=seed)[["x", "y"]].to_numpy())
            _, ours = prim_mst(D)
            theirs = minimum_spanning_tree(D).sum()
            self.assertAlmostEqual(ours, theirs, places=9)


if __name__ == "__main__":
    unittest.main()
