"""
ml_model.py
===========
Module 1 (Weather-Normalised Expected Output Model) and Module 9 (Explainable ML).

Approach
--------
We don't predict raw kW directly, because two inverters of different rated
sizes would then look "underperforming" simply for being smaller. Instead we:

1. Predict a capacity-normalized PERFORMANCE_RATIO (AC_POWER / rated capacity)
   from weather + time + inverter-history features. This is the same idea as
   the industry-standard solar "Performance Ratio" used by real plant O&M
   teams: it isolates equipment/operational health from panel size and
   weather.
2. Train on a "healthy" subset of the data rather than everything. If we fit
   on all data (including faulty periods), the model partly learns the faults
   as normal and expected output quietly drops to match reality -- which
   defeats the purpose of an anomaly detector. So we fit once, drop the worst
   residual outliers (likely real faults), and refit on the cleaner subset.
   This produces an expected-output curve closer to "what a healthy inverter
   should do", not "what inverters on average did including their downtime".
3. Convert the predicted ratio back to kW using each inverter's own estimated
   capacity to get EXPECTED_POWER for downstream gap/loss calculations.

Evaluation
----------
`evaluate_expected_output_model` provides an honest, held-out assessment of
model quality (MAE, MSE, RMSE, R^2) -- something the original module lacked.
Two things make this evaluation non-trivial for this dataset and are handled
explicitly:

* Chronological split, not a random row split. Consecutive 15-minute rows are
  strongly autocorrelated (PREV_INTERVAL_RATIO/ROLLING_24H_RATIO literally
  encode recent history), so a random split would leak near-duplicate
  information between train and test and overstate accuracy. We instead hold
  out the most recent slice of *days* per plant as the test set, which
  mimics how the model would actually be used (trained on history, scored on
  new incoming data).
* The held-out split happens BEFORE any fitting, including the first-pass /
  healthy-refit trimming inside `train_expected_output_model`, so test rows
  never influence training in any way.
* Metrics are reported both on the model's native target (PERFORMANCE_RATIO,
  unitless, comparable across inverters of any size) and on the
  back-converted EXPECTED_POWER (kW), since that's the scale the downstream
  gap/loss/financial modules (2, 5, 6) actually consume.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "HOUR_DECIMAL",
    "PREV_INTERVAL_RATIO",
    "ROLLING_24H_RATIO",
]
TARGET = "PERFORMANCE_RATIO"

RESIDUAL_TRIM_THRESHOLD = -0.15  # drop rows performing >15% below the first-pass model
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2  # fraction of days per plant held out for evaluation


def train_expected_output_model(df: pd.DataFrame, n_estimators: int = 200):
    """
    Fits the weather-normalized expected-output model using the two-pass
    "healthy data" trimming approach described above.

    Returns
    -------
    model : fitted RandomForestRegressor (second pass, used for scoring)
    diagnostics : dict with the first-pass model, trimmed row count, etc.
    """
    X = df[FEATURES]
    y = df[TARGET]

    first_pass = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    first_pass.fit(X, y)
    first_pred = first_pass.predict(X)

    residual_ratio = np.where(first_pred > 1e-6, (y - first_pred) / first_pred, 0)
    healthy_mask = residual_ratio > RESIDUAL_TRIM_THRESHOLD

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X[healthy_mask], y[healthy_mask])

    diagnostics = {
        "first_pass_model": first_pass,
        "n_total": len(df),
        "n_trimmed": int((~healthy_mask).sum()),
        "pct_trimmed": round(100 * (~healthy_mask).mean(), 2),
    }
    return model, diagnostics


def score_expected_output(df: pd.DataFrame, model) -> pd.DataFrame:
    """Adds EXPECTED_RATIO and EXPECTED_POWER (kW) columns to df."""
    df = df.copy()
    df["EXPECTED_RATIO"] = model.predict(df[FEATURES]).clip(0, 1.5)
    df["EXPECTED_POWER"] = df["EXPECTED_RATIO"] * df["INVERTER_CAPACITY_KW"]
    return df


def _chronological_split(df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE):
    """
    Splits by DATE (per plant) rather than by row.

    For each plant independently, the last `test_size` fraction of its
    calendar days becomes the test set and everything earlier becomes the
    training set. Splitting per-plant (rather than on the pooled date range)
    keeps both plants represented in train and test even if their logging
    periods don't fully overlap. Using whole days (not individual rows)
    avoids splitting a single day's rolling/lag features across the
    train/test boundary.
    """
    train_parts, test_parts = [], []
    for plant_id, plant_df in df.groupby("PLANT_ID"):
        dates = np.sort(plant_df["DATE"].unique())
        if len(dates) < 5:
            # Too little history to hold out a meaningful test slice; keep
            # everything in train for this plant rather than fail.
            train_parts.append(plant_df)
            continue
        split_idx = max(1, int(len(dates) * (1 - test_size)))
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])
        train_parts.append(plant_df[plant_df["DATE"].isin(train_dates)])
        test_parts.append(plant_df[plant_df["DATE"].isin(test_dates)])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return train_df, test_df


def _regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_expected_output_model(df: pd.DataFrame, n_estimators: int = 200,
                                    test_size: float = DEFAULT_TEST_SIZE):
    """
    Trains the expected-output model on a chronological train split and
    reports held-out regression metrics on the remaining, unseen days.

    Returns
    -------
    model : RandomForestRegressor fit on the training split only (use
            `train_expected_output_model(df)` separately if you want a final
            production model refit on 100% of the data after validating here).
    diagnostics : dict from `train_expected_output_model`, plus:
        - "n_train_days" / "n_test_days_by_plant": split sizes
        - "ratio_*": MAE/MSE/RMSE/R^2 on PERFORMANCE_RATIO (unitless, 0-1.5 scale)
        - "power_kw_*": MAE/MSE/RMSE/R^2 on EXPECTED_POWER vs AC_POWER (kW)
        - "daylight_only_ratio_*": same ratio metrics restricted to daylight
          rows (IRRADIATION > 0), since nighttime rows are trivially ~0 for
          every model and would otherwise inflate R^2 with easy points.
    """
    train_df, test_df = _chronological_split(df, test_size=test_size)
    if len(test_df) == 0:
        raise ValueError(
            "No test rows produced by the chronological split -- the dataset "
            "doesn't span enough distinct days per plant to hold any out."
        )

    model, diagnostics = train_expected_output_model(train_df, n_estimators=n_estimators)

    test_scored = score_expected_output(test_df, model)
    y_true_ratio = test_scored[TARGET]
    y_pred_ratio = test_scored["EXPECTED_RATIO"]
    y_true_kw = test_scored["AC_POWER"]
    y_pred_kw = test_scored["EXPECTED_POWER"]

    daylight = test_scored["IRRADIATION"] > 0

    diagnostics.update({
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        **{f"ratio_{k}": v for k, v in _regression_metrics(y_true_ratio, y_pred_ratio).items()},
        **{f"power_kw_{k}": v for k, v in _regression_metrics(y_true_kw, y_pred_kw).items()},
        **{f"daylight_only_ratio_{k}": v
           for k, v in _regression_metrics(y_true_ratio[daylight], y_pred_ratio[daylight]).items()},
    })
    return model, diagnostics


def get_feature_importance(model) -> pd.Series:
    """Global feature importance (module 9), always available (no extra deps)."""
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)


def get_shap_explanation(model, df_row: pd.DataFrame):
    """
    Local, per-alert explanation of which features pushed the prediction for
    ONE row up or down (module 9). Falls back gracefully if shap isn't
    installed -- feature importance from get_feature_importance still covers
    the requirement.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_row[FEATURES])
        return pd.Series(shap_values[0], index=FEATURES).sort_values(key=abs, ascending=False)
    except Exception:
        return None


if __name__ == "__main__":
    from data_pipeline import load_and_prepare

    data = load_and_prepare()

    # Held-out evaluation first (honest accuracy check on unseen days)
    _, eval_diag = evaluate_expected_output_model(data)
    print("=== Held-out evaluation (chronological split) ===")
    print(f"Train rows: {eval_diag['n_train_rows']} | Test rows: {eval_diag['n_test_rows']}")
    print(f"Trimmed as unhealthy during training: {eval_diag['n_trimmed']} "
          f"({eval_diag['pct_trimmed']}%)")
    print("\nPERFORMANCE_RATIO (unitless, target scale):")
    print(f"  MAE={eval_diag['ratio_mae']:.4f}  MSE={eval_diag['ratio_mse']:.4f}  "
          f"RMSE={eval_diag['ratio_rmse']:.4f}  R2={eval_diag['ratio_r2']:.4f}")
    print("PERFORMANCE_RATIO, daylight-only rows (harder, more meaningful subset):")
    print(f"  MAE={eval_diag['daylight_only_ratio_mae']:.4f}  "
          f"MSE={eval_diag['daylight_only_ratio_mse']:.4f}  "
          f"RMSE={eval_diag['daylight_only_ratio_rmse']:.4f}  "
          f"R2={eval_diag['daylight_only_ratio_r2']:.4f}")
    print("EXPECTED_POWER vs AC_POWER (kW):")
    print(f"  MAE={eval_diag['power_kw_mae']:.2f}  MSE={eval_diag['power_kw_mse']:.2f}  "
          f"RMSE={eval_diag['power_kw_rmse']:.2f}  R2={eval_diag['power_kw_r2']:.4f}")

    # Final production model, refit on 100% of the data once validated above
    model, diag = train_expected_output_model(data)
    print("\n=== Production model (fit on all data) diagnostics ===", diag)
    scored = score_expected_output(data, model)
    print(scored[["AC_POWER", "EXPECTED_POWER", "EXPECTED_RATIO"]].describe())
    print("\nFeature importance:\n", get_feature_importance(model))
