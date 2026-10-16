# Different Exact and Approximation Algorithms for the Travelling Salesperson Problem (TSP)

An academic project (undergraduate Computer Engineering / AI & ML) that implements, tests and compares six TSP algorithms in
Python, with Jupyter Notebook as the main working environment.

| Kind | Algorithms |
|------|-----------|
| Exact | Brute Force, Held-Karp dynamic programming, Branch and Bound |
| Heuristic / approximation | Nearest Neighbor, MST-based preorder (2-approximation), Christofides (1.5-approximation) |

Data are synthetic, generated from fixed random seeds. All result tables and figures come from actually running the code.

## Requirements

* **Python 3.9 or newer.** Development and all checks were done on **Python 3.13**; older versions were not tested.
* Packages (see `requirements.txt`): NumPy, pandas, Matplotlib, NetworkX, SciPy, and `notebook` / `ipykernel` for Jupyter.

## Installation

```bash
# 1. unzip the archive and enter the folder
unzip TSP_Algorithm_Comparison_Project.zip
cd TSP_Algorithm_Comparison_Project

# 2. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install the dependencies
pip install -r requirements.txt
```

## Running the project

**Notebook (main way).**

```bash
jupyter notebook TSP_Algorithm_Comparison.ipynb      # or: jupyter lab
```

Then choose *Kernel -> Restart & Run All*. The full notebook takes a few minutes. For a faster run set `QUICK = True`
in the setup cell. The notebook is shipped with its outputs already filled in (see the note below).

**Script (no notebook needed).** Regenerates all datasets, result CSV files and figures:

```bash
python run_project.py              # full experiments, roughly 2 minutes
python run_project.py --quick      # reduced experiments
python run_project.py --tests      # run the unit tests first, then the experiments
```

**Unit tests.**

```bash
python -m unittest discover -s tests -v
python -m pytest tests             # alternative, if pytest is installed
```

## Folder structure

```text
TSP_Algorithm_Comparison_Project/
├── README.md                       this file
├── requirements.txt
├── TSP_Algorithm_Comparison.ipynb  main notebook (theory, experiments, plots, conclusion)
├── run_project.py                  script that regenerates datasets, results and figures
├── datasets/
│   ├── cities_small.csv            11 cities  (seed 42)
│   ├── cities_medium.csv           50 cities  (seed 2024)
│   ├── cities_large.csv           200 cities  (seed 7)
│   ├── sample_distance_matrix.csv  custom 6x6 distance matrix (format documented)
│   └── dataset_README.md           description of every dataset, columns, units, loading
├── src/
│   ├── data_generator.py           reproducible dataset generation
│   ├── distance_utils.py           distance matrices, validation, Prim MST, CSV loading
│   ├── exact_algorithms.py         brute force, Held-Karp, branch and bound, 1-tree lower bound
│   ├── approximation_algorithms.py nearest neighbor, MST preorder, Christofides
│   ├── validation.py               tour validation and independent cost recomputation
│   ├── visualization.py            plots of cities, tours and benchmark charts
│   └── benchmarking.py             timing / memory / gap / scalability experiments
├── results/
│   ├── algorithm_comparison.csv    all algorithms on the small, medium and large datasets
│   ├── scalability_results.csv     time, memory and quality versus number of cities
│   ├── optimality_gap_results.csv  gaps over many random instances
│   ├── tours.csv, branch_and_bound_node_counts.csv   extra outputs
│   └── figures/                    PNG charts; route_visualizations/ holds the tour drawings
├── docs/
│   ├── project_report.md           full report
│   ├── algorithm_explanations.md   pseudo-code and correctness arguments
│   ├── complexity_analysis.md      time and space complexity
│   ├── viva_questions_answers.md   30 viva questions with answers
│   └── sample_outputs.md           real output excerpts from the final run
└── tests/
    ├── test_algorithms.py          algorithm correctness, bounds, guarantees, edge cases
    └── test_tour_validation.py     tour/matrix validation, datasets, SciPy cross-checks
```

## Using your own data

* **City coordinates:** a CSV with columns `city_id,x,y` (see `datasets/dataset_README.md`).
* **Distance matrix:** a labelled square CSV (format in `datasets/dataset_README.md`), loaded with
  `src.distance_utils.load_distance_matrix_csv`. Exact algorithms and Nearest Neighbor accept asymmetric matrices;
  MST preorder and Christofides need a symmetric matrix, and their approximation guarantees additionally need the
  triangle inequality (a warning is issued otherwise).

```python
from src.distance_utils import load_distance_matrix_csv
from src.exact_algorithms import held_karp_tsp
from src.approximation_algorithms import christofides_tsp

D, labels = load_distance_matrix_csv("datasets/sample_distance_matrix.csv")
tour, cost = held_karp_tsp(D)           # exact
tour2, cost2 = christofides_tsp(D)      # 1.5-approximation (metric instances)
```

## Important notes and limits

* Exact algorithms have size limits (`max_n`): Brute Force 11, Held-Karp 20 (benchmark: 18), Branch and Bound 18 by default
  (benchmark: 20). Larger inputs raise `ValueError` unless you raise the limit on purpose.
* Timings depend on your machine; compare growth rates, not absolute seconds. Your numbers will differ slightly from the
  shipped CSV files.
* Memory is the peak traced by `tracemalloc` (Python/NumPy allocations during one call) and is only recorded for runs shorter than
  2 seconds.
* The approximation guarantees apply to symmetric metric instances only. Nearest Neighbor has no constant guarantee.

### How the shipped notebook outputs were produced (honest note)

The sandbox used to build this project could not install Jupyter (the package index was unreachable), so the notebook was executed
by a small Python runner that executes the code cells in order and stores their printed output, tables and figures in the
`.ipynb` file. The code is ordinary Python and was not changed for this. The notebook has **not** been run through a real Jupyter kernel
by the author; if anything behaves differently on your machine, re-run it with *Restart & Run All*.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Start Jupyter from the project folder (the notebook also adds the project root to `sys.path`). Run scripts from the project root. |
| `ModuleNotFoundError: networkx` (or another package) | `pip install -r requirements.txt` in the active environment; in Jupyter select the kernel of that environment. |
| Notebook kernel not found | `python -m ipykernel install --user --name tsp-project` and select that kernel. |
| `ValueError: ... limited to n <= ...` | You asked an exponential algorithm to solve a too-large instance; reduce `n` or pass a larger `max_n` knowingly. |
| `RuntimeWarning: ... violates the triangle inequality` | Your matrix is not metric: tours are valid but the 2x / 1.5x guarantee does not apply. |
| `ValueError: distance matrix must be symmetric` | MST preorder / Christofides need symmetry; use an exact algorithm or Nearest Neighbor for asymmetric costs. |
| Figures do not appear in a script | Scripts save PNG files into `results/figures/`; they do not open windows (Agg backend). |
| Run takes very long | Use `python run_project.py --quick` or set `QUICK = True` in the notebook. |
| Garbled characters in docs on Windows | Open the Markdown files with a UTF-8 aware editor. |

## References

See the end of `docs/project_report.md` for the list of sources.
