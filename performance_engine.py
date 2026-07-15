"""
performance_engine.py
======================
Modules 2-8 of the Solar Power Generation & Plant Performance Analyzer:

  2. Actual vs Expected Performance Gap + Normal/Warning/Critical classification
  3. Inverter-Level Benchmarking (peer comparison under identical weather)
  4. Persistent Underperformance Detection (de-noises cloud-driven blips)
  5. Energy-Loss Estimation (kWh)
  6. Financial Impact Estimation (configurable tariff)
  7. Likely-Cause Identification (rule-based, interpretable)
  8. Action Recommendation Engine (human-readable next steps)
"""

import numpy as np
import pandas as pd

from data_pipeline import INTERVAL_HOURS, NIGHT_IRRADIANCE_THRESHOLD, OVERHEAT_TEMP_C

# ---- Module 2: classification thresholds ----
WARNING_GAP_PCT = 0.15
CRITICAL_GAP_PCT = 0.30

# ---- Module 3: peer benchmarking ----
PEER_DEVIATION_THRESHOLD = 0.20   # this inverter >20% below the peer median => flagged

# ---- Module 4: persistence ----
PERSISTENCE_INTERVALS = 4          # 4 x 15min = 1 hour of sustained underperformance
MIN_EXPECTED_POWER_KW = 5.0        # below this, % gaps are noisy/meaningless -> treat as "no generation"

# ---- Module 7: plant-wide outage ----
PLANT_OUTAGE_INVERTER_FRACTION = 0.75  # if >=75% of online inverters are underperforming together


def calculate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Module 2: actual-vs-expected gap and Normal/Warning/Critical classification."""
    df = df.copy()
    df["GAP_KW"] = df["EXPECTED_POWER"] - df["AC_POWER"]

    daylight_and_meaningful = (df["IRRADIATION"] > NIGHT_IRRADIANCE_THRESHOLD) & \
                               (df["EXPECTED_POWER"] >= MIN_EXPECTED_POWER_KW)

    df["GAP_PCT"] = np.where(
        daylight_and_meaningful,
        (df["GAP_KW"] / df["EXPECTED_POWER"]).clip(-2, 2),
        0.0,
    )

    conditions = [
        ~daylight_and_meaningful,
        df["GAP_PCT"] > CRITICAL_GAP_PCT,
        df["GAP_PCT"] > WARNING_GAP_PCT,
    ]
    choices = ["Nighttime / No Generation Expected", "Critical", "Warning"]
    df["STATUS"] = np.select(conditions, choices, default="Normal")
    return df


def benchmark_against_peers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 3: at every timestamp, all inverters within a plant share identical
    weather (one weather station per plant). So any inverter producing well
    below the median PERFORMANCE_RATIO of its peers at that same instant is
    likely an equipment-specific problem, not weather.
    """
    df = df.copy()
    peer_median = df.groupby(["PLANT_ID", "DATE_TIME"])["PERFORMANCE_RATIO"].transform("median")
    peer_count = df.groupby(["PLANT_ID", "DATE_TIME"])["PERFORMANCE_RATIO"].transform("count")

    df["PEER_MEDIAN_RATIO"] = peer_median
    safe_peer = peer_median.replace(0, np.nan)
    df["PEER_DEVIATION_PCT"] = ((peer_median - df["PERFORMANCE_RATIO"]) / safe_peer).fillna(0)
    df["PEER_UNDERPERFORMER"] = (peer_count >= 4) & (df["PEER_DEVIATION_PCT"] > PEER_DEVIATION_THRESHOLD)
    return df


def flag_plant_wide_outage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 7 helper: if almost every inverter is underperforming at once
    (during clear daylight), the likely cause is a grid/plant-wide event
    (curtailment, grid outage, transformer trip) rather than N separate
    inverter faults.
    """
    df = df.copy()
    daylight = df["STATUS"] != "Nighttime / No Generation Expected"
    flagged = df["STATUS"].isin(["Warning", "Critical"])

    total_online = df[daylight].groupby(["PLANT_ID", "DATE_TIME"])["INVERTER_ID"].transform("count")
    flagged_online = df[daylight & flagged].groupby(["PLANT_ID", "DATE_TIME"])["INVERTER_ID"].transform("count")

    df["FLAGGED_FRACTION"] = (flagged_online / total_online).reindex(df.index).fillna(0)
    df["PLANT_WIDE_EVENT"] = daylight & flagged & (df["FLAGGED_FRACTION"] >= PLANT_OUTAGE_INVERTER_FRACTION)
    return df


def detect_persistence(df: pd.DataFrame, min_intervals: int = PERSISTENCE_INTERVALS) -> pd.DataFrame:
    """
    Module 4: only escalate to a real ALERT when an inverter's Warning/Critical
    status holds for `min_intervals` consecutive readings. A single bad
    15-minute reading (passing cloud, momentary curtailment) is noise, not
    a maintenance issue.
    """
    df = df.sort_values(["PLANT_ID", "INVERTER_ID", "DATE_TIME"]).copy()
    is_bad = df["STATUS"].isin(["Warning", "Critical"])

    grp_key = df["PLANT_ID"].astype(str) + "_" + df["INVERTER_ID"]
    # run-length encode consecutive "bad" streaks per inverter
    change = (is_bad != is_bad.groupby(grp_key).shift(1)).cumsum()
    run_length = is_bad.groupby([grp_key, change]).cumcount() + 1
    df["CONSECUTIVE_BAD_INTERVALS"] = np.where(is_bad, run_length, 0)
    df["PERSISTENT_ALERT"] = df["CONSECUTIVE_BAD_INTERVALS"] >= min_intervals
    return df.reset_index(drop=True)


def estimate_losses(df: pd.DataFrame, tariff_per_kwh: float) -> pd.DataFrame:
    """Modules 5 & 6: energy loss (kWh) and financial impact for a given tariff."""
    df = df.copy()
    df["LOSS_KWH"] = df["GAP_KW"].clip(lower=0) * INTERVAL_HOURS
    df["FINANCIAL_IMPACT"] = df["LOSS_KWH"] * tariff_per_kwh
    return df


def identify_cause(row) -> str:
    """
    Module 7: interpretable rule-based cause identification. Checked in
    order of specificity so the most actionable diagnosis wins.
    """
    if row["STATUS"] == "Nighttime / No Generation Expected":
        return "N/A (no generation expected)"
    if row["STATUS"] == "Normal":
        return "N/A (performing as expected)"

    if row.get("PLANT_WIDE_EVENT", False):
        return "Plant-Wide Event (grid outage / curtailment)"

    # Zero/near-zero output despite decent sunlight and a healthy expected
    # output => the inverter or its sensor line has very likely tripped.
    if row["AC_POWER"] < 1.0 and row["IRRADIATION"] > 0.3:
        return "Possible Inverter Trip / Sensor Fault"

    if row["MODULE_TEMPERATURE"] > OVERHEAT_TEMP_C:
        return "Overheating / Thermal Derating"

    if row.get("PEER_UNDERPERFORMER", False):
        return "Inverter-Specific Fault (electrical / soiling / shading)"

    if row["IRRADIATION"] < 0.15:
        return "Weather-Related Reduction (low irradiation / cloud cover)"

    return "Inverter/Electrical Issue (inspect connections)"


ACTION_MAP = {
    "Plant-Wide Event (grid outage / curtailment)":
        "Check grid connection, transformer status, and plant-level SCADA/curtailment logs.",
    "Possible Inverter Trip / Sensor Fault":
        "Inspect inverter status remotely; verify DC input and sensor wiring on-site if it does not self-recover.",
    "Overheating / Thermal Derating":
        "Check ventilation/cooling around the inverter and module array; inspect for shading that traps heat.",
    "Inverter-Specific Fault (electrical / soiling / shading)":
        "Inspect this inverter's panel string for soiling, shading, or loose electrical connections.",
    "Weather-Related Reduction (low irradiation / cloud cover)":
        "No action needed; output is consistent with reduced sunlight.",
    "Inverter/Electrical Issue (inspect connections)":
        "Schedule a technician inspection of inverter status and electrical connections.",
    "N/A (no generation expected)": "No action needed.",
    "N/A (performing as expected)": "No action needed.",
}


def build_action_text(row) -> str:
    """Module 8: generates the human-readable recommendation, e.g.
    'Inverter INV-07 is operating 28% below expected output ... Inspect ...'
    """
    if row["CAUSE"].startswith("N/A"):
        return "No action needed."

    gap_pct = abs(row["GAP_PCT"]) * 100
    action = ACTION_MAP.get(row["CAUSE"], "Investigate further.")
    return (
        f"Inverter {row['INVERTER_ID']} ({row['PLANT_LABEL']}) is operating "
        f"{gap_pct:.0f}% below expected output ({row['CAUSE']}). "
        f"Estimated energy loss is {row['LOSS_KWH']:.2f} kWh "
        f"(~{row['FINANCIAL_IMPACT']:.2f} currency units at the configured tariff). "
        f"{action}"
    )


def run_full_analysis(df: pd.DataFrame, tariff_per_kwh: float,
                       min_persistence_intervals: int = PERSISTENCE_INTERVALS) -> pd.DataFrame:
    """Runs modules 2-8 end to end. `df` must already contain EXPECTED_POWER
    (i.e. have been scored by ml_model.score_expected_output)."""
    df = calculate_gap(df)
    df = benchmark_against_peers(df)
    df = flag_plant_wide_outage(df)
    df = detect_persistence(df, min_persistence_intervals)
    df = estimate_losses(df, tariff_per_kwh)
    df["CAUSE"] = df.apply(identify_cause, axis=1)
    df["RECOMMENDED_ACTION"] = df.apply(build_action_text, axis=1)
    return df


if __name__ == "__main__":
    from data_pipeline import load_and_prepare
    from ml_model import train_expected_output_model, score_expected_output

    data = load_and_prepare()
    model, diag = train_expected_output_model(data)
    scored = score_expected_output(data, model)
    analyzed = run_full_analysis(scored, tariff_per_kwh=7.0)

    print(analyzed["STATUS"].value_counts())
    print("\nPersistent alerts:", analyzed["PERSISTENT_ALERT"].sum())
    alerts = analyzed[analyzed["PERSISTENT_ALERT"]]
    print("\nCause breakdown among persistent alerts:\n", alerts["CAUSE"].value_counts())
    print("\nSample recommendation:\n", alerts["RECOMMENDED_ACTION"].iloc[0] if len(alerts) else "none")
    print("\nTotal estimated loss (kWh):", analyzed["LOSS_KWH"].sum().round(1))
    print("Total financial impact:", analyzed["FINANCIAL_IMPACT"].sum().round(1))
