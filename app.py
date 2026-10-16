"""
TSP Interactive Web Application & Benchmark Dashboard
Built with Streamlit and Plotly
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Setup project root path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import benchmarking as bm
from src import visualization as viz
from src.data_generator import generate_cities
from src.distance_utils import (
    coords_from_dataframe,
    euclidean_distance_matrix,
    is_symmetric,
    load_cities_csv,
    satisfies_triangle_inequality,
    validate_distance_matrix,
)
from src.exact_algorithms import one_tree_lower_bound
from src.validation import is_valid_tour, recompute_tour_cost

# Page Configuration
st.set_page_config(
    page_title="Travelling Salesperson Problem (TSP) Visualizer & Benchmark",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark/Modern theme with glassmorphism accent)
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)

ALGO_COLORS = viz.ALGO_COLORS

# Cache dataset loading
@st.cache_data
def get_cached_dataset(dataset_name: str) -> pd.DataFrame:
    path = ROOT / "datasets" / f"cities_{dataset_name}.csv"
    if path.exists():
        return load_cities_csv(path)
    return generate_cities(11, seed=42)


def create_plotly_tour_map(coords: np.ndarray, tour: list[int] | None, title: str, color: str = "#38bdf8") -> go.Figure:
    """Create an interactive Plotly 2D tour visualization."""
    fig = go.Figure()
    n = len(coords)

    # Plot tour edges if available
    if tour is not None and len(tour) > 1:
        tour_coords = coords[tour]
        fig.add_trace(
            go.Scatter(
                x=tour_coords[:, 0],
                y=tour_coords[:, 1],
                mode="lines",
                line=dict(color=color, width=2.5),
                name="Tour Route",
                hoverinfo="skip",
            )
        )

    # Plot regular cities (except start city 0)
    non_start_idx = [i for i in range(n) if i != 0]
    if non_start_idx:
        fig.add_trace(
            go.Scatter(
                x=coords[non_start_idx, 0],
                y=coords[non_start_idx, 1],
                mode="markers+text" if n <= 30 else "markers",
                text=[f"City {i}" for i in non_start_idx],
                textposition="top center",
                textfont=dict(size=10, color="#cbd5e1"),
                marker=dict(size=8 if n > 50 else 11, color="#38bdf8", line=dict(color="#0f172a", width=1.5)),
                name="Cities",
                hovertemplate="<b>City %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )

    # Highlight Start City 0
    fig.add_trace(
        go.Scatter(
            x=[coords[0, 0]],
            y=[coords[0, 1]],
            mode="markers+text",
            text=["Start (City 0)"],
            textposition="bottom center",
            textfont=dict(size=11, color="#f59e0b", family="sans-serif"),
            marker=dict(size=14, color="#ef4444", symbol="star", line=dict(color="#ffffff", width=2)),
            name="Start City",
            hovertemplate="<b>Start City 0</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title="X Coordinate",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94a3b8",
        ),
        yaxis=dict(
            title="Y Coordinate",
            gridcolor="#334155",
            zerolinecolor="#475569",
            color="#94a3b8",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8"),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520,
    )
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Configuration")

dataset_type = st.sidebar.radio(
    "Data Source",
    ["Predefined Dataset", "Random City Generator", "Custom CSV Upload"],
    index=0,
)

if dataset_type == "Predefined Dataset":
    predefined_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Small (11 cities, seed 42)", "Medium (50 cities, seed 2024)", "Large (200 cities, seed 7)"],
        index=0,
    )
    if "Small" in predefined_choice:
        cities_df = get_cached_dataset("small")
        dataset_name = "small"
    elif "Medium" in predefined_choice:
        cities_df = get_cached_dataset("medium")
        dataset_name = "medium"
    else:
        cities_df = get_cached_dataset("large")
        dataset_name = "large"

elif dataset_type == "Random City Generator":
    n_cities = st.sidebar.slider("Number of Cities (n)", min_value=4, max_value=100, value=12, step=1)
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=99999, value=42, step=1)
    cities_df = generate_cities(n=n_cities, seed=int(seed))
    dataset_name = f"random_{n_cities}"

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (city_id, x, y)", type=["csv"])
    if uploaded_file is not None:
        try:
            cities_df = pd.read_csv(uploaded_file)
            dataset_name = "custom"
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            cities_df = get_cached_dataset("small")
            dataset_name = "small"
    else:
        st.sidebar.info("Upload a CSV with columns `city_id, x, y`. Defaulting to Small dataset.")
        cities_df = get_cached_dataset("small")
        dataset_name = "small"

coords = coords_from_dataframe(cities_df)
D = validate_distance_matrix(euclidean_distance_matrix(coords))
n = len(coords)

st.sidebar.markdown(f"**Current Dataset:** `{n}` cities")
st.sidebar.markdown(f"- Symmetric: `{'Yes' if is_symmetric(D) else 'No'}`")
st.sidebar.markdown(f"- Triangle Inequality (Metric): `{'Yes' if satisfies_triangle_inequality(D) else 'No'}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Algorithms")

# Safe algorithm filtering based on size n
available_algos = []
for algo_name, spec in bm.ALGORITHMS.items():
    max_limit = spec["max_n"]
    if max_limit is None or n <= max_limit:
        available_algos.append(algo_name)

if n > 11:
    st.sidebar.caption(f"⚠️ Exact exponential algorithms automatically hidden/disabled for n={n} to prevent UI freezing.")

selected_algos = st.sidebar.multiselect(
    "Select Algorithms to Run:",
    options=list(bm.ALGORITHMS.keys()),
    default=available_algos if len(available_algos) <= 6 else [
        "Nearest Neighbor",
        "MST Preorder",
        "Christofides",
    ],
)

start_button = st.sidebar.button("🚀 Solve / Compare", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Travelling Salesperson Problem (TSP) Solver</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive benchmarking and visualization of Exact (Brute Force, Held-Karp, Branch & Bound) and Approximation (Nearest Neighbor, MST Preorder, Christofides) algorithms.</div>',
    unsafe_allow_html=True,
)

tab_visualizer, tab_comparison, tab_precomputed, tab_theory = st.tabs([
    "🗺️ Interactive Route Map",
    "⚡ Algorithm Comparison",
    "📊 Benchmarks & Scalability",
    "📚 Theory & Viva Notes",
])

# Ensure session state has latest execution results
if "last_results" not in st.session_state or start_button or st.session_state.get("last_n") != n:
    with st.spinner(f"Computing solutions for {len(selected_algos)} algorithm(s) on {n} cities..."):
        results_df = bm.one_instance(D, algorithms=selected_algos, repeats=3, measure_memory=True)
        st.session_state["last_results"] = results_df
        st.session_state["last_n"] = n
        st.session_state["last_coords"] = coords

results_df = st.session_state["last_results"]
ok_results = results_df[results_df["status"] == "ok"]

# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE ROUTE MAP
# -----------------------------------------------------------------------------
with tab_visualizer:
    if ok_results.empty:
        st.warning("No algorithm returned a valid tour. Please adjust your algorithm selection.")
    else:
        col_select, col_info = st.columns([1, 2])
        with col_select:
            view_algo = st.selectbox("Choose Algorithm Tour to View:", options=ok_results["algorithm"].tolist(), index=0)

        algo_row = ok_results[ok_results["algorithm"] == view_algo].iloc[0]
        tour = algo_row["tour"]
        cost = algo_row["tour_cost"]
        time_ms = algo_row["time_s"] * 1000
        mem_kb = algo_row["peak_memory_kb"]

        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="Tour Distance (Cost)", value=f"{cost:.2f}")
        with m2:
            st.metric(label="Execution Time", value=f"{time_ms:.2f} ms" if time_ms >= 0.01 else "< 0.01 ms")
        with m3:
            st.metric(
                label="Peak Extra RAM",
                value=f"{mem_kb:.1f} KB" if pd.notnull(mem_kb) else "N/A",
            )
        with m4:
            valid = algo_row["valid_tour"]
            st.metric(label="Valid Tour?", value="✅ Valid" if valid else "❌ Invalid")

        # Tour Map: Interactive Plotly if available, else clean Matplotlib
        if HAS_PLOTLY:
            fig = create_plotly_tour_map(
                coords,
                tour,
                title=f"{view_algo} Tour (Total Distance: {cost:.2f})",
                color=ALGO_COLORS.get(view_algo, "#38bdf8"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = viz.plot_tour(
                coords,
                tour,
                title=f"{view_algo} Tour (Distance: {cost:.2f})",
                cost=cost,
                color=ALGO_COLORS.get(view_algo, "#0072B2"),
            )
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("📍 View Tour Sequence (Order of Cities Visited)"):
            if tour is not None:
                tour_str = " → ".join(map(str, tour))
                st.code(tour_str, language="text")
                st.caption(f"Starts at City 0, visits all {n} unique cities, and returns to City 0.")

# -----------------------------------------------------------------------------
# TAB 2: ALGORITHM COMPARISON
# -----------------------------------------------------------------------------
with tab_comparison:
    st.markdown("### ⚡ Head-to-Head Performance Comparison")

    if ok_results.empty:
        st.info("Run algorithms using the sidebar button to see comparison.")
    else:
        # Summary highlights
        best_row = ok_results.loc[ok_results["tour_cost"].idxmin()]
        fastest_row = ok_results.loc[ok_results["time_s"].idxmin()]

        h1, h2, h3 = st.columns(3)
        with h1:
            st.success(f"🏆 **Best Tour Quality:** **{best_row['algorithm']}**\n\nDistance: `{best_row['tour_cost']:.2f}`")
        with h2:
            st.info(f"⚡ **Fastest Algorithm:** **{fastest_row['algorithm']}**\n\nTime: `{fastest_row['time_s']*1000:.3f} ms`")
        with h3:
            lower_bound_val = ok_results["lower_bound"].iloc[0] if "lower_bound" in ok_results.columns and pd.notnull(ok_results["lower_bound"].iloc[0]) else None
            if lower_bound_val:
                st.warning(f"🎯 **1-Tree Lower Bound:** `{lower_bound_val:.2f}`\n\n(No tour can be shorter than this)")
            else:
                st.warning("🎯 **Lower Bound:** Calculated for symmetric metric instances.")

        # Comparison Charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            if HAS_PLOTLY:
                fig_cost = px.bar(
                    ok_results,
                    x="algorithm",
                    y="tour_cost",
                    color="algorithm",
                    color_discrete_map=ALGO_COLORS,
                    title="Tour Distance (Lower is Better)",
                    labels={"tour_cost": "Tour Length", "algorithm": "Algorithm"},
                    text_auto=".2f",
                )
                fig_cost.update_layout(
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    showlegend=False,
                    margin=dict(l=30, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            else:
                st.markdown("##### Tour Distance (Lower is Better)")
                chart_cost_df = ok_results.set_index("algorithm")[["tour_cost"]]
                st.bar_chart(chart_cost_df)

        with col_chart2:
            ok_plot = ok_results.copy()
            ok_plot["time_ms"] = ok_plot["time_s"] * 1000
            if HAS_PLOTLY:
                fig_time = px.bar(
                    ok_plot,
                    x="algorithm",
                    y="time_ms",
                    color="algorithm",
                    color_discrete_map=ALGO_COLORS,
                    title="Execution Time in ms (Lower is Better)",
                    log_y=True if ok_plot["time_ms"].max() / (ok_plot["time_ms"].min() + 1e-6) > 20 else False,
                    labels={"time_ms": "Time (ms)", "algorithm": "Algorithm"},
                    text_auto=".3f",
                )
                fig_time.update_layout(
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    showlegend=False,
                    margin=dict(l=30, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.markdown("##### Execution Time in ms (Lower is Better)")
                chart_time_df = ok_plot.set_index("algorithm")[["time_ms"]]
                st.bar_chart(chart_time_df)

        # Full Data Table
        st.markdown("#### 📋 Detailed Comparison Table")
        display_df = results_df[[
            "algorithm", "type", "status", "tour_cost", "time_s", "peak_memory_kb", "valid_tour"
        ]].copy()
        display_df["time_ms"] = (display_df["time_s"] * 1000).round(3)
        display_df["tour_cost"] = display_df["tour_cost"].round(2)
        display_df = display_df.rename(columns={
            "algorithm": "Algorithm",
            "type": "Category",
            "status": "Status",
            "tour_cost": "Tour Cost",
            "time_ms": "Time (ms)",
            "peak_memory_kb": "Peak RAM (KB)",
            "valid_tour": "Valid Tour?",
        }).drop(columns=["time_s"])

        st.dataframe(display_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: BENCHMARKS & SCALABILITY
# -----------------------------------------------------------------------------
with tab_precomputed:
    st.markdown("### 📊 Comprehensive Scalability & Optimality Gap Benchmarks")
    st.caption("These charts and result tables were generated by running the full test suite and `run_project.py` script.")

    col_fig1, col_fig2 = st.columns(2)
    fig_dir = ROOT / "results" / "figures"

    scal_img = fig_dir / "scalability_comparison.png"
    gap_img = fig_dir / "optimality_gap_by_size.png"
    dist_img = fig_dir / "route_distance_comparison.png"
    time_img = fig_dir / "execution_time_comparison.png"

    with col_fig1:
        if scal_img.exists():
            st.image(str(scal_img), caption="Scalability: Time & Memory vs Number of Cities (n up to 200)", use_container_width=True)
        if dist_img.exists():
            st.image(str(dist_img), caption="Route Length by Algorithm on Small Dataset (11 Cities)", use_container_width=True)

    with col_fig2:
        if gap_img.exists():
            st.image(str(gap_img), caption="Optimality Gap (%) vs Size n (Held-Karp as ground truth)", use_container_width=True)
        if time_img.exists():
            st.image(str(time_img), caption="Execution Time Comparison (Log scale)", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: THEORY & VIVA NOTES
# -----------------------------------------------------------------------------
with tab_theory:
    st.markdown("### 📚 Theoretical Foundations & Viva Questions")

    st.markdown(
        """
| Algorithm | Category | Time Complexity | Space Complexity | Theoretical Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **Brute Force** | Exact | $O(n!)$ | $O(n)$ | Optimal ($1.0 \\times \\text{OPT}$) |
| **Held-Karp DP** | Exact | $O(2^n n^2)$ | $O(2^n n)$ | Optimal ($1.0 \\times \\text{OPT}$) |
| **Branch & Bound** | Exact | $O(n!)$ worst-case | $O(n)$ | Optimal ($1.0 \\times \\text{OPT}$) |
| **Nearest Neighbor** | Heuristic | $O(n^2)$ | $O(n)$ | No constant bound ($O(\\log n)$ on metric) |
| **MST Preorder** | Approximation | $O(n^2)$ | $O(n)$ | $\\le 2.0 \\times \\text{OPT}$ (Metric TSP) |
| **Christofides** | Approximation | $O(n^3)$ | $O(n^2)$ | $\\le 1.5 \\times \\text{OPT}$ (Metric TSP) |
"""
    )

    st.markdown("---")
    st.markdown("#### 💡 Key Viva Q&A Highlights")

    with st.expander("Q1: Why is the TSP NP-hard?"):
        st.write(
            "The Hamiltonian Cycle problem (which is known to be NP-complete) reduces directly to the decision version of the TSP. Hence, unless P = NP, no polynomial-time exact algorithm can solve TSP in the worst case."
        )

    with st.expander("Q2: Why does Christofides guarantee a 1.5 approximation factor?"):
        st.write(
            "Christofides builds a Minimum Spanning Tree (MST $T$) of cost $\\le \\text{OPT}$, and adds a minimum-weight perfect matching $M$ on the odd-degree vertices of $T$. It is proven that $w(M) \\le 0.5 \\times \\text{OPT}$. Combining them gives an Eulerian graph of cost $\\le 1.5 \\times \\text{OPT}$, and shortcutting via the triangle inequality never increases the length."
        )

    with st.expander("Q3: What is the 1-Tree Lower Bound?"):
        st.write(
            "A 1-tree is formed by picking a city, finding the MST on the remaining $n-1$ cities, and connecting the picked city to its two nearest neighbors. Every valid TSP tour is a 1-tree with all vertex degrees equal to 2. Therefore, the minimum-weight 1-tree is always less than or equal to the optimal TSP tour length."
        )

    with st.expander("Q4: Why does Nearest Neighbor fail to provide a constant approximation guarantee?"):
        st.write(
            "Nearest Neighbor makes greedy local choices without foresight. It can easily get forced into traversing an extremely expensive edge at the very end to return home to the start city."
        )

# Footer
st.markdown("---")
st.caption("Travelling Salesperson Problem (TSP) Academic Project | Built with Python, Streamlit, Plotly & SciPy.")
