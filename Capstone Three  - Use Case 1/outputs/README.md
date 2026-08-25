# Weather- and Holiday-Aware Retail Demand Forecasting

**Capstone Three — Data Science Career Track**

Retailers routinely over- or under-stock because standard demand forecasts rely only on historical sales and ignore external demand drivers like weather and holidays. This project wrangles and models daily, category-level retail sales — joined with weather and holiday-calendar data — to answer two questions:

1. Can external factors (weather, holidays) improve on a sales-history-only forecast?
2. Which product categories are most sensitive to weather and holidays?

## Data Sources

- **Sales + holidays:** [Corporación Favorita Grocery Sales Forecasting (Kaggle)](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) — `train.csv` (date, store, item/category, sales), `holidays_events.csv`, `stores.csv`.
- **Weather:** [Open-Meteo Historical API](https://open-meteo.com/) or [NOAA CDO](https://www.ncdc.noaa.gov/cdo-web/), joined on date.

> The notebooks currently run end-to-end on a **synthetic dataset** that mimics the real schema and dynamics (weekly/annual seasonality, holiday spikes, weather sensitivity, stockouts). Swapping `load_sales()`, `load_weather()`, and `load_holidays()` for real file/API loaders leaves every downstream cell unchanged.

## Project Structure

| Notebook | Purpose |
|---|---|
| `1__retail_demand_eda.ipynb` | Data wrangling (merge sales + weather + holidays, handle stockouts/missing data, feature engineering) and exploratory data analysis |
| `2__retail_demand_Pre-processing_Work.ipynb` | Model-ready dataset prep; rebuilds the EDA notebook's output via `papermill` before encoding/scaling |
| `3__modeling_prep.ipynb` | Same pre-processing pipeline as notebook 2, self-contained (re-derives the wrangled data inline instead of via `papermill`) |
| `4__modeling.ipynb` | Baseline + three candidate models, time-series-aware tuning, evaluation, and final model selection |

## Pipeline

### 1. Data Wrangling & EDA (`1__retail_demand_eda.ipynb`)
- Merges three sources on different natural grains: daily per-store/category sales, daily regional weather, and calendar holiday events.
- Treats `sales == 0` as ambiguous (no demand vs. no inventory) and carries a `stockout_flag` rather than trusting raw sales as ground truth.
- Engineers calendar features (day-of-week, weekend, month, week-of-year), lag/rolling sales features (`sales_lag_1`, `sales_lag_7`, `sales_roll_7`, `sales_roll_28`), and a "days to nearest holiday" feature to capture pre-holiday demand ramp-up.
- Uses a **time-respecting 90-day holdout** (no shuffling) for the later train/test split.
- EDA covers response skew/zero-inflation, feature distributions, per-category correlation with weather/holiday/promo (a heatmap directly answering "which categories are most weather-sensitive"), collinearity among lag/rolling features, outliers by weather condition and holiday status, seasonal decomposition, and faceted scatter/LOWESS plots to catch nonlinear threshold effects (e.g. demand jumping only below freezing).

### 2. Pre-processing (`2__...Pre-processing_Work.ipynb` / `3__modeling_prep.ipynb`)
- Fixes a minor unit-mismatch edge case producing negative `days_to_holiday` values.
- Confirms and one-hot/dummy-encodes categorical columns (`store`, `category`, `dow`, `month`, `holiday_type`, `weather_bucket`), casts existing boolean flags to 0/1.
- Confirms features span wildly different magnitudes (0–1 indicators, 0–100 temperature/day/month, hundreds for sales-based features) and standardizes the continuous predictors with `StandardScaler` — **fit on train only, applied to test** to avoid leakage. The target (`sales`) is left unscaled.
- Drops the warm-up rows where lag/rolling features are undefined (first 28 days per store-category series).
- Re-applies the same chronological 90-day train/test split and writes `train_clean.csv` / `test_clean.csv` for modeling.

### 3. Modeling (`4__modeling.ipynb`)
- Confirms the forecasting/time-series nature of the problem and the need for chronological (not random) validation.
- **Baseline:** naive seasonal forecast (predict today's sales using the same weekday last week, i.e. `sales_lag_7`).
- **Candidate models**, tuned with `TimeSeriesSplit` cross-validation (`RandomizedSearchCV`):
  - **Ridge Regression** — fast, interpretable linear baseline; L2 penalty addresses collinearity among lag/rolling features.
  - **Random Forest Regressor** — bagging ensemble, captures nonlinear/threshold weather effects without manual feature crosses.
  - **HistGradientBoostingRegressor** — boosting ensemble, typically strongest on tabular data of this shape.
- **Primary metric:** RMSE (penalizes large misses, which are costlier for a retailer); MAE and R² used as secondary checks, all computed on the held-out test set.

## Results

| Model | RMSE | MAE | R² | Fit time (s) |
|---|---|---|---|---|
| **Random Forest (tuned)** | **25.29** | 17.05 | **0.967** | 97.7 |
| Gradient Boosting (tuned) | 25.77 | **17.01** | 0.965 | 10.5 |
| Ridge Regression (tuned) | 35.41 | 23.83 | 0.935 | 0.5 |
| Naive seasonal (lag-7) | 67.89 | 35.78 | 0.760 | — |

**Selected model: Random Forest (tuned)** — `n_estimators=150, min_samples_leaf=1, max_depth=12`.

All three tuned models beat the naive seasonal baseline by a wide margin, confirming that weather, holiday, and calendar features add real forecasting value beyond sales history alone. Ridge Regression trails the tree ensembles since it can't capture the threshold-style weather effects surfaced in the EDA (e.g. demand jumping only below freezing). Random Forest and Gradient Boosting perform comparably; Random Forest edges out on RMSE/R² while Gradient Boosting trains roughly 9x faster.

## Requirements

- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `statsmodels` (seasonal decomposition)
- `scikit-learn` (`StandardScaler`, `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `TimeSeriesSplit`, `RandomizedSearchCV`)
- `papermill` (only for notebook 2's parameterized re-execution of the EDA notebook)

## How to Run

Run the notebooks in order:

```
1__retail_demand_eda.ipynb
3__modeling_prep.ipynb      # (or 2__..., which additionally re-executes notebook 1 via papermill)
4__modeling.ipynb
```

Each notebook currently regenerates the synthetic data inline, so they can also be run independently. `train_clean.csv` and `test_clean.csv` produced by the pre-processing step are required inputs for `4__modeling.ipynb`.

## Next Steps

- Swap the synthetic `load_sales()` / `load_weather()` / `load_holidays()` loaders for the real Favorita + weather data sources.
- Extend weather/holiday sensitivity analysis to more store/region combinations once multi-region weather data is available.
- Consider a Tweedie or zero-inflated model formulation for categories with high zero-sales rates.
