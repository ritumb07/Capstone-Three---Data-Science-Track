"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC dataset simulating a government agency's
multi-year, multi-department budget planning, execution, and workload data.

No real government data is used. The structure mirrors common public-sector
budget systems (appropriation -> allotment -> obligation -> expenditure) so
the project generalizes to real agency data with minimal changes.

Outputs (written to ../data/raw/):
  1. budget_master.csv        - annual budget line items (allocation vs actuals)
  2. monthly_execution.csv    - monthly obligations/expenditures per line item
  3. workload_drivers.csv     - annual organizational drivers (headcount, caseload, etc.)
  4. macro_indicators.csv     - annual macro/economic indicators (inflation, CPI, COLA)
  5. capital_projects.csv     - multi-year capital project spend profiles
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Reference dimensions
# ---------------------------------------------------------------------------
DEPARTMENTS = [
    ("DEPT-01", "Public Safety"),
    ("DEPT-02", "Health & Human Services"),
    ("DEPT-03", "Transportation & Infrastructure"),
    ("DEPT-04", "Education"),
    ("DEPT-05", "Administration & Finance"),
    ("DEPT-06", "Environment & Natural Resources"),
    ("DEPT-07", "Information Technology"),
    ("DEPT-08", "Parks & Recreation"),
]

CATEGORIES = [
    ("PERS", "Personnel & Compensation", 0.55),   # base share of dept budget
    ("OPS", "Operating Expenses", 0.20),
    ("CAP", "Capital Outlay", 0.12),
    ("GRANTS", "Grants & Subsidies", 0.08),
    ("CONTR", "Contracted Services", 0.05),
]

FISCAL_YEARS = list(range(2019, 2027))  # FY2019 - FY2026 (2026 = current/partial year)
CURRENT_FY = 2026
CURRENT_FY_MONTHS_ELAPSED = 8  # Oct(FY start)-May equivalent -> 8 months of FY26 actuals known

# Department-level baseline annual budget (in $ thousands) and growth trend
DEPT_BASE = {
    "DEPT-01": 185_000, "DEPT-02": 240_000, "DEPT-03": 160_000, "DEPT-04": 210_000,
    "DEPT-05": 75_000, "DEPT-06": 60_000, "DEPT-07": 95_000, "DEPT-08": 40_000,
}
# Underlying annual real growth rate per department (drives multi-year trend)
DEPT_GROWTH = {
    "DEPT-01": 0.030, "DEPT-02": 0.045, "DEPT-03": 0.020, "DEPT-04": 0.025,
    "DEPT-05": 0.015, "DEPT-06": 0.018, "DEPT-07": 0.060, "DEPT-08": 0.010,
}
# Departments/categories prone to systematic over/under spending (variance realism)
VARIANCE_BIAS = {
    ("DEPT-01", "OPS"): 0.06, ("DEPT-02", "GRANTS"): -0.08, ("DEPT-03", "CAP"): -0.15,
    ("DEPT-07", "CONTR"): 0.10, ("DEPT-04", "PERS"): -0.03, ("DEPT-06", "CAP"): -0.20,
}

MONTHS = pd.date_range("2000-10-01", periods=12, freq="MS").month  # Oct..Sep fiscal months
FY_MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Seasonal obligation curve (fraction of annual obligations by fiscal month) - typical gov't
# pattern: slow start, ramps up, year-end spending spike ("use it or lose it")
SEASONAL_CURVE = np.array([0.045, 0.055, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.105, 0.115, 0.145])
SEASONAL_CURVE = SEASONAL_CURVE / SEASONAL_CURVE.sum()


def macro_indicators():
    rows = []
    cpi = 100.0
    for fy in FISCAL_YEARS:
        inflation = float(np.clip(RNG.normal(0.028, 0.012), -0.01, 0.09))
        cola = float(np.clip(inflation + RNG.normal(0.0, 0.005), 0, 0.08))  # cost of living adj
        cpi *= (1 + inflation)
        unemployment = float(np.clip(RNG.normal(4.2, 0.6), 2.5, 9.0))
        interest_rate = float(np.clip(RNG.normal(3.5, 1.2), 0.1, 8.0))
        rows.append({
            "fiscal_year": fy, "inflation_rate": round(inflation, 4),
            "cola_pct": round(cola, 4), "cpi_index": round(cpi, 2),
            "unemployment_rate": round(unemployment, 2),
            "interest_rate_pct": round(interest_rate, 2),
        })
    return pd.DataFrame(rows)


def workload_drivers():
    rows = []
    # base workload metrics per department, each with its own growth + noise
    base_metrics = {
        "DEPT-01": ("calls_for_service", 210_000, 0.02),
        "DEPT-02": ("caseload_clients", 95_000, 0.035),
        "DEPT-03": ("lane_miles_maintained", 4_200, 0.005),
        "DEPT-04": ("student_enrollment", 88_000, 0.012),
        "DEPT-05": ("transactions_processed", 1_450_000, 0.02),
        "DEPT-06": ("permits_issued", 12_500, 0.015),
        "DEPT-07": ("supported_endpoints", 18_000, 0.08),
        "DEPT-08": ("park_visitors", 2_100_000, 0.03),
    }
    for dept_id, dept_name in DEPARTMENTS:
        metric_name, base_val, growth = base_metrics[dept_id]
        val = base_val
        headcount = DEPT_BASE[dept_id] / 95  # rough FTE proxy from budget
        for fy in FISCAL_YEARS:
            noise = RNG.normal(0, 0.02)
            val = val * (1 + growth + noise)
            hc_noise = RNG.normal(0, 0.015)
            headcount = headcount * (1 + growth * 0.6 + hc_noise)
            rows.append({
                "fiscal_year": fy, "dept_id": dept_id, "dept_name": dept_name,
                "workload_metric": metric_name, "workload_value": round(val, 0),
                "authorized_fte": round(headcount, 1),
            })
    return pd.DataFrame(rows)


def budget_master(macro_df):
    rows = []
    for dept_id, dept_name in DEPARTMENTS:
        base = DEPT_BASE[dept_id]
        growth = DEPT_GROWTH[dept_id]
        level = base
        for fy in FISCAL_YEARS:
            infl = macro_df.loc[macro_df.fiscal_year == fy, "inflation_rate"].values[0]
            level = level * (1 + growth + infl * 0.4 + RNG.normal(0, 0.01))
            for cat_id, cat_name, share in CATEGORIES:
                cat_noise = RNG.normal(1.0, 0.04)
                allocated = level * share * cat_noise
                # one-time supplemental / rescission events (budget realism)
                if RNG.random() < 0.06:
                    allocated *= RNG.choice([0.85, 1.15])

                bias = VARIANCE_BIAS.get((dept_id, cat_id), 0.0)
                exec_rate = np.clip(RNG.normal(0.94 + bias, 0.05), 0.55, 1.15)

                if fy == CURRENT_FY:
                    # current year: only partial actuals exist; apply seasonal YTD fraction
                    ytd_fraction = SEASONAL_CURVE[:CURRENT_FY_MONTHS_ELAPSED].sum()
                    actual = allocated * exec_rate * ytd_fraction
                    is_partial_year = True
                else:
                    actual = allocated * exec_rate
                    is_partial_year = False

                rows.append({
                    "fiscal_year": fy,
                    "dept_id": dept_id, "dept_name": dept_name,
                    "category_id": cat_id, "category_name": cat_name,
                    "budget_allocated": round(allocated, 1),
                    "actual_spend": round(actual, 1),
                    "is_partial_year": is_partial_year,
                })
    df = pd.DataFrame(rows)
    df["variance_amount"] = round(df.actual_spend - df.budget_allocated, 1)
    df["variance_pct"] = round(df.variance_amount / df.budget_allocated * 100, 2)
    return df


def monthly_execution(budget_df):
    rows = []
    for _, r in budget_df.iterrows():
        fy = r.fiscal_year
        months_to_gen = CURRENT_FY_MONTHS_ELAPSED if r.is_partial_year else 12
        curve = SEASONAL_CURVE[:months_to_gen]
        curve = curve / curve.sum()
        monthly_noise = RNG.normal(1.0, 0.08, size=months_to_gen)
        monthly_amounts = r.actual_spend * curve * monthly_noise
        # rescale so sum matches actual_spend
        monthly_amounts = monthly_amounts * (r.actual_spend / monthly_amounts.sum())
        for i in range(months_to_gen):
            fy_month = FY_MONTH_ORDER[i]
            cal_year = fy - 1 if fy_month >= 10 else fy
            rows.append({
                "fiscal_year": fy, "fiscal_month": i + 1,
                "calendar_month": fy_month, "calendar_year": cal_year,
                "period": f"{cal_year}-{fy_month:02d}",
                "dept_id": r.dept_id, "category_id": r.category_id,
                "obligated_amount": round(monthly_amounts[i], 1),
            })
    return pd.DataFrame(rows)


def capital_projects():
    rows = []
    project_names = [
        "Regional Data Center Modernization", "Bridge Rehabilitation Program",
        "Emergency Dispatch System Upgrade", "School Facilities Renewal",
        "Water Treatment Plant Expansion", "Fleet Electrification Initiative",
        "Public Records Digitization", "Highway Corridor Widening",
        "Parks Trail Network Expansion", "Cybersecurity Infrastructure Program",
    ]
    depts_for_cap = ["DEPT-01", "DEPT-02", "DEPT-03", "DEPT-04", "DEPT-06", "DEPT-07", "DEPT-08"]
    for i, name in enumerate(project_names):
        dept_id = depts_for_cap[i % len(depts_for_cap)]
        start_fy = int(RNG.choice([2019, 2020, 2021, 2022, 2023, 2024]))
        duration = int(RNG.integers(2, 5))
        total_budget = float(RNG.uniform(8_000, 60_000))
        # front-loaded / back-loaded spend curve
        raw_curve = RNG.dirichlet(np.linspace(1, 2.2, duration) * RNG.choice([1, -1]) * -1 + 3)
        pct_complete = 0.0
        for yr_offset in range(duration):
            fy = start_fy + yr_offset
            if fy > CURRENT_FY:
                continue
            planned = total_budget * raw_curve[yr_offset]
            slip = RNG.normal(0, 0.12)  # schedule/cost variance
            actual = planned * (1 + slip) if fy < CURRENT_FY else planned * (1 + slip) * 0.5
            pct_complete = min(100.0, pct_complete + 100 / duration + RNG.normal(0, 3))
            rows.append({
                "project_id": f"CAP-{i+1:03d}", "project_name": name, "dept_id": dept_id,
                "fiscal_year": fy, "planned_spend": round(planned, 1),
                "actual_spend": round(actual, 1), "total_project_budget": round(total_budget, 1),
                "pct_complete": round(pct_complete, 1),
                "status": "In Progress" if fy == CURRENT_FY or yr_offset == duration - 1 and fy == CURRENT_FY else
                          ("Complete" if yr_offset == duration - 1 and fy < CURRENT_FY else "In Progress"),
            })
    return pd.DataFrame(rows)


def main():
    macro_df = macro_indicators()
    budget_df = budget_master(macro_df)
    monthly_df = monthly_execution(budget_df)
    workload_df = workload_drivers()
    capital_df = capital_projects()

    out = "/home/claude/capstone/data/raw"
    macro_df.to_csv(f"{out}/macro_indicators.csv", index=False)
    budget_df.to_csv(f"{out}/budget_master.csv", index=False)
    monthly_df.to_csv(f"{out}/monthly_execution.csv", index=False)
    workload_df.to_csv(f"{out}/workload_drivers.csv", index=False)
    capital_df.to_csv(f"{out}/capital_projects.csv", index=False)

    print("Generated files:")
    for f, d in [("budget_master.csv", budget_df), ("monthly_execution.csv", monthly_df),
                 ("workload_drivers.csv", workload_df), ("macro_indicators.csv", macro_df),
                 ("capital_projects.csv", capital_df)]:
        print(f"  {f:28s} shape={d.shape}")


if __name__ == "__main__":
    main()
