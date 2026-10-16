"""
data_generator.py
=================

Reproducible synthetic data for the project.

Everything is generated from fixed random seeds with ``numpy.random.default_rng``,
so running the generator again always reproduces the same CSV files.

Datasets
--------
============  ======  ====  ==========================================
file          cities  seed  purpose
============  ======  ====  ==========================================
cities_small      11    42  all six algorithms (exact vs approx)
cities_medium     50  2024  approximation algorithms (exact is infeasible)
cities_large     200     7  scalability of the polynomial-time algorithms
============  ======  ====  ==========================================

Cities are drawn uniformly at random in the square [0, 100] x [0, 100]
(abstract distance units; think of them as kilometres) and rounded to 2 decimals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .distance_utils import is_symmetric, satisfies_triangle_inequality

DATASET_SPECS = {
    "cities_small": {"n": 11, "seed": 42},
    "cities_medium": {"n": 50, "seed": 2024},
    "cities_large": {"n": 200, "seed": 7},
}

AREA_SIZE = 100.0


def generate_cities(n: int, seed: int, area: float = AREA_SIZE) -> pd.DataFrame:
    """Return a DataFrame ``city_id, x, y`` with ``n`` random cities in ``[0, area]^2``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    xy = np.round(rng.uniform(0.0, area, size=(n, 2)), 2)
    return pd.DataFrame({"city_id": np.arange(n), "x": xy[:, 0], "y": xy[:, 1]})


# City labels and integer grid coordinates of the hand-made sample matrix.
SAMPLE_MATRIX_COORDS = {
    "Depot": (0, 0),
    "North": (2, 7),
    "East": (8, 3),
    "South": (4, -4),
    "West": (-5, 2),
    "Market": (6, 8),
}


def generate_sample_distance_matrix() -> pd.DataFrame:
    """A small *custom* distance matrix that is **not** Euclidean.

    Six places on a street grid; the distance between two places is the
    **Manhattan (taxi-cab) distance** ``|dx| + |dy|`` in km.  Manhattan distances
    are symmetric, non-negative, have a zero diagonal and satisfy the triangle
    inequality, so all six algorithms accept this matrix.
    """
    labels = list(SAMPLE_MATRIX_COORDS)
    pts = np.array([SAMPLE_MATRIX_COORDS[k] for k in labels], dtype=float)
    manhattan = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=-1)
    return pd.DataFrame(manhattan, index=labels, columns=labels)


def generate_all_datasets(dataset_dir) -> dict:
    """Write the three city CSVs and the sample matrix CSV into ``dataset_dir``.

    Returns a dict mapping file stem -> DataFrame.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for stem, spec in DATASET_SPECS.items():
        df = generate_cities(spec["n"], spec["seed"])
        df.to_csv(dataset_dir / f"{stem}.csv", index=False)
        out[stem] = df
    matrix = generate_sample_distance_matrix()
    assert is_symmetric(matrix.to_numpy()) and satisfies_triangle_inequality(matrix.to_numpy())
    matrix.to_csv(dataset_dir / "sample_distance_matrix.csv", index_label="city")
    out["sample_distance_matrix"] = matrix
    return out


if __name__ == "__main__":      # python -m src.data_generator
    here = Path(__file__).resolve().parent.parent / "datasets"
    generate_all_datasets(here)
    print(f"Datasets written to {here}")
