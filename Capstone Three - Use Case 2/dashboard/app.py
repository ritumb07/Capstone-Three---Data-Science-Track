"""
Streamlit dashboard for the Government Budget Intelligence & Forecasting System.

Run locally with:
    pip install streamlit plotly pandas
    streamlit run dashboard/app.py

Expects the processed data produced by the notebook pipeline:
    ../data/processed/master_analytical_table.csv
    ../data/processed/feature_table.csv
If those don't exist yet, run the notebook (or src/generate_data.py +
the feature-engineering cells) first.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Budget Intelligence Dashboard", layout="wide")

DATA_PATH = "../data/processed/master_analytical_table.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


st.title("🏛️ Government Budget Intelligence & Forecasting")
st.caption(
    "Historical execution, variance detection, and multi-year budget projection "
    "— built on the same analytical table used in the capstone notebook."
)

try:
    master = load_data()
except FileNotFoundError:
    st.error(
        "Processed data not found. Run `src/generate_data.py` and the notebook's "
        "Data Engineering section first to create `data/processed/master_analytical_table.csv`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
years = sorted(master.fiscal_year.unique())
depts = sorted(master.dept_name.unique())

year_range = st.sidebar.select_slider(
    "Fiscal Year Range", options=years, value=(years[0], years[-1])
)
selected_depts = st.sidebar.multiselect("Departments", depts, default=depts)

filtered = master[
    (master.fiscal_year >= year_range[0])
    & (master.fiscal_year <= year_range[1])
    & (master.dept_name.isin(selected_depts))
]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
latest_fy = filtered.fiscal_year.max()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Allocated (filtered)", f"${filtered.budget_allocated.sum():,.0f}K")
c2.metric("Total Actual Spend (filtered)", f"${filtered.actual_spend.sum():,.0f}K")
avg_var = filtered.variance_pct.mean()
c3.metric("Avg. Variance %", f"{avg_var:+.1f}%")
risk_lines = (filtered.variance_pct.abs() > 15).sum()
c4.metric("Lines with >15% Variance", f"{risk_lines}")

st.divider()

# ---------------------------------------------------------------------------
# Trend + composition
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    trend = filtered.groupby(["fiscal_year", "dept_name"]).actual_spend.sum().reset_index()
    fig = px.line(
        trend, x="fiscal_year", y="actual_spend", color="dept_name", markers=True,
        title="Spend Trend by Department",
        labels={"actual_spend": "$ Thousands", "fiscal_year": "Fiscal Year"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    snap = filtered[filtered.fiscal_year == latest_fy]
    fig2 = px.sunburst(
        snap, path=["dept_name", "category_name"], values="actual_spend",
        color="variance_pct", color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0,
        title=f"FY{latest_fy} Breakdown (color = variance %)",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Variance table with risk flags
# ---------------------------------------------------------------------------
st.subheader("Budget Line Variance & Risk Flags")
snap2 = filtered[filtered.fiscal_year == latest_fy].copy()


def flag(v):
    if v > 5:
        return "🔴 Overrun"
    if v < -15:
        return "🟠 Lapse Risk"
    return "🟢 Normal"


snap2["risk_flag"] = snap2.variance_pct.apply(flag)
st.dataframe(
    snap2[
        ["dept_name", "category_name", "budget_allocated", "actual_spend",
         "variance_amount", "variance_pct", "risk_flag"]
    ].sort_values("variance_pct"),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Data: synthetic dataset for demonstration (see `src/generate_data.py`). "
    "Replace with agency GL/ERP extracts using the same schema for production use."
)
