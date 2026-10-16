# Dataset README

All datasets are **synthetic** and generated from scratch by `src/data_generator.py` using fixed
random seeds (`numpy.random.default_rng(seed)`). Re-running the generator reproduces exactly the same
files. No real-world measurements are included, and none were invented as "experimental results":
the files only contain *inputs* for the algorithms.

Regenerate everything with:

```bash
python -m src.data_generator        # from the project root
# or simply
python run_project.py
```

## City-coordinate datasets

Cities are drawn uniformly at random in the square **[0, 100] x [0, 100]** and rounded to 2 decimals.
The unit is abstract; you may read it as kilometres. Distances are **Euclidean**, so every
distance matrix built from these files is symmetric, has a zero diagonal and satisfies the triangle
inequality (a *metric* instance).

| File | Cities | Seed | Purpose |
|------|-------:|-----:|---------|
| `cities_small.csv`  |  11 |   42 | All six algorithms, including brute force; optimum known |
| `cities_medium.csv` |  50 | 2024 | Heuristics / approximations; exact algorithms are infeasible; compared with a lower bound |
| `cities_large.csv`  | 200 |    7 | Scalability of the polynomial-time algorithms; the first *n* rows are also used as nested instances of size *n* for the scalability experiment |

### Columns (all three files)

| Column | Type | Meaning |
|--------|------|---------|
| `city_id` | integer | 0-based city index. City `0` is the start/end city of every tour. |
| `city_name` | string | Name of the city / location (e.g., Mumbai, Pune, Nagpur, etc.). |
| `x` | float | x coordinate (distance units, 0-100) |
| `y` | float | y coordinate (distance units, 0-100) |

### Loading

```python
import pandas as pd
from src.distance_utils import euclidean_distance_matrix, coords_from_dataframe

cities = pd.read_csv("datasets/cities_small.csv")
coords = coords_from_dataframe(cities)          # numpy array of shape (n, 2)
D = euclidean_distance_matrix(coords)           # (n, n) distance matrix
```

## Custom distance-matrix dataset: `sample_distance_matrix.csv`

A small hand-made example (6 places) showing how to supply **your own** distance matrix instead of
coordinates. The distance between two places is the **Manhattan (taxi-cab) distance** `|dx| + |dy|` on an
integer street grid, in km, so the matrix is **not Euclidean** but is symmetric, non-negative, has a zero
diagonal and satisfies the triangle inequality. It is an illustrative example, not real road data.

### Format

* Plain CSV, comma-separated, with a header row.
* The first column and the first row contain the **city labels** (the top-left cell is `city`).
* The labels in the first row must be identical (and in the same order) to the labels in the first column.
* The remaining cells are the distances `D[i][j]` from row city *i* to column city *j* (km).
* Required properties: square, numeric and finite, no negative values, zero diagonal.
  For the MST-preorder and Christofides algorithms the matrix must also be **symmetric**; their
  approximation guarantees additionally need the **triangle inequality**. Exact algorithms and Nearest
  Neighbor also accept asymmetric matrices.

```text
city,Depot,North,East,South,West,Market
Depot,0.0,9.0,11.0,8.0,7.0,14.0
North,9.0,0.0,10.0,13.0,12.0,5.0
...
```

### Loading

```python
from src.distance_utils import load_distance_matrix_csv, validate_distance_matrix

D, labels = load_distance_matrix_csv("datasets/sample_distance_matrix.csv")
D = validate_distance_matrix(D)                 # raises ValueError if the matrix is malformed
```

To use your own file, save it in the same layout, load it with `load_distance_matrix_csv`, and pass the
resulting array `D` to any algorithm in `src/`.
