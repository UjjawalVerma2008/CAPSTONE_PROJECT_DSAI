"""
data_pipeline.py
================
Data ingestion, cleaning and feature engineering for the Solar Power
Generation & Plant Performance Analyzer.

Handles two real-world data quirks found in this dataset (verified by
direct inspection, not assumed):

1. Plant 1's generation file stores DATE_TIME as dd-mm-yyyy while Plant 2
   and both weather files use yyyy-mm-dd. Parsed explicitly per-file.

2. Plant 1's DC_POWER column is reported on a different scale than
   AC_POWER (ratio ~9.8-10x, whereas Plant 2's DC/AC ratio is ~1.02,
   i.e. a normal inverter efficiency). Cross-checking AC_POWER against
   DAILY_YIELD confirms AC_POWER is the trustworthy, correctly-scaled
   generation figure for both plants. We therefore treat AC_POWER (kW)
   as "actual generation" everywhere, and rescale Plant 1's DC_POWER by
   /10 purely for reference/diagnostics.
"""

import os

import numpy as np
import pandas as pd

# This file lives in .../src/, and the data folder is one level up at
# .../data/. Anchoring to __file__ (instead of a bare "data" string) means
# this resolves correctly regardless of the current working directory or
# which machine the project is copied to.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SRC_DIR), "data")
INTERVAL_HOURS = 0.25  # 15-minute logging interval, confirmed from timestamps

PLANT_CONFIG = {
    1: {
        "gen_file": "Plant_1_Generation_Data.csv",
        "weather_file": "Plant_1_Weather_Sensor_Data.csv",
        "gen_dayfirst": True,     # dd-mm-yyyy
        "dc_power_correction": 1 / 10,  # fixes the DC/AC scale mismatch
    },
    2: {
        "gen_file": "Plant_2_Generation_Data.csv",
        "weather_file": "Plant_2_Weather_Sensor_Data.csv",
        "gen_dayfirst": False,    # yyyy-mm-dd
        "dc_power_correction": 1.0,
    },
}

HIGH_IRRADIANCE_THRESHOLD = 0.75   # used to estimate each inverter's rated capacity
NIGHT_IRRADIANCE_THRESHOLD = 0.02  # below this we treat the interval as "no generation expected"
OVERHEAT_TEMP_C = 55.0             # module temps above this cause real thermal derating


def _load_single_plant(plant_id: int, data_dir: str = DATA_DIR) -> pd.DataFrame:
    cfg = PLANT_CONFIG[plant_id]

    gen = pd.read_csv(f"{data_dir}/{cfg['gen_file']}")
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=cfg["gen_dayfirst"])
    gen = gen.rename(columns={"SOURCE_KEY": "INVERTER_ID"})

    wea = pd.read_csv(f"{data_dir}/{cfg['weather_file']}")
    wea["DATE_TIME"] = pd.to_datetime(wea["DATE_TIME"], dayfirst=False)
    wea = wea.drop(columns=["SOURCE_KEY"])  # single weather station per plant; would duplicate columns on merge

    df = pd.merge(gen, wea, on=["DATE_TIME", "PLANT_ID"], how="inner")

    # Correct the known DC_POWER scale issue (kept only for reference/diagnostics;
    # AC_POWER is used as "actual generation" throughout the pipeline).
    df["DC_POWER_CORRECTED"] = df["DC_POWER"] * cfg["dc_power_correction"]

    df["PLANT_LABEL"] = f"Plant {plant_id}"
    return df


def load_raw_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and merge both plants' generation + weather data into one frame."""
    frames = [_load_single_plant(pid, data_dir) for pid in PLANT_CONFIG]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["PLANT_ID", "INVERTER_ID", "DATE_TIME"]).reset_index(drop=True)
    return df


def _estimate_inverter_capacity(df: pd.DataFrame) -> pd.Series:
    """
    Estimate each inverter's rated capacity from its own high-irradiance output
    (95th percentile of AC_POWER when IRRADIATION > 0.75). This lets us compare
    inverters of different sizes on a common, capacity-normalized scale instead
    of assuming every inverter should produce identically.
    """
    high_sun = df[df["IRRADIATION"] > HIGH_IRRADIANCE_THRESHOLD]
    cap = high_sun.groupby(["PLANT_ID", "INVERTER_ID"])["AC_POWER"].quantile(0.95)

    # Fallback for any inverter with too few high-irradiance samples: use its
    # overall max, then the plant-wide median capacity as a last resort.
    overall_max = df.groupby(["PLANT_ID", "INVERTER_ID"])["AC_POWER"].max()
    cap = cap.reindex(overall_max.index)
    cap = cap.fillna(overall_max)
    plant_median_cap = cap.groupby(level=0).transform("median")
    cap = cap.where(cap > 0, plant_median_cap)
    cap.name = "INVERTER_CAPACITY_KW"
    return cap


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time features, capacity normalization, and inverter-history features
    (module 1 requirement: "...time, and inverter history").
    """
    df = df.copy()

    # ---- Time features ----
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["MINUTE"] = df["DATE_TIME"].dt.minute
    df["HOUR_DECIMAL"] = df["HOUR"] + df["MINUTE"] / 60.0
    df["DATE"] = df["DATE_TIME"].dt.date

    # ---- Capacity normalization ----
    capacity = _estimate_inverter_capacity(df)
    df = df.merge(capacity.rename("INVERTER_CAPACITY_KW"),
                   on=["PLANT_ID", "INVERTER_ID"], how="left")
    df["PERFORMANCE_RATIO"] = (df["AC_POWER"] / df["INVERTER_CAPACITY_KW"]).clip(lower=0, upper=1.5)

    # ---- Inverter-history features (avoid leakage: only look backward) ----
    df = df.sort_values(["PLANT_ID", "INVERTER_ID", "DATE_TIME"])
    grp = df.groupby(["PLANT_ID", "INVERTER_ID"])["PERFORMANCE_RATIO"]

    # Ratio in the immediately preceding interval (captures sudden trips/faults)
    df["PREV_INTERVAL_RATIO"] = grp.shift(1)
    # Trailing 24h (96 x 15-min) rolling average, shifted so "today" isn't included
    df["ROLLING_24H_RATIO"] = grp.transform(lambda s: s.shift(1).rolling(96, min_periods=8).mean())

    # Fill early-series gaps with that inverter's own long-run average, then a global fallback
    inv_mean = grp.transform("mean")
    df["PREV_INTERVAL_RATIO"] = df["PREV_INTERVAL_RATIO"].fillna(inv_mean)
    df["ROLLING_24H_RATIO"] = df["ROLLING_24H_RATIO"].fillna(inv_mean)
    df["PREV_INTERVAL_RATIO"] = df["PREV_INTERVAL_RATIO"].fillna(df["PERFORMANCE_RATIO"].mean())
    df["ROLLING_24H_RATIO"] = df["ROLLING_24H_RATIO"].fillna(df["PERFORMANCE_RATIO"].mean())

    df["IS_DAYLIGHT"] = df["IRRADIATION"] > NIGHT_IRRADIANCE_THRESHOLD

    return df.reset_index(drop=True)


def load_and_prepare(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Single entry point: raw load -> merge -> feature engineering."""
    raw = load_raw_data(data_dir)
    return engineer_features(raw)


if __name__ == "__main__":
    data = load_and_prepare()
    print(data.shape)
    print(data[["PLANT_LABEL", "INVERTER_ID", "DATE_TIME", "AC_POWER",
                "INVERTER_CAPACITY_KW", "PERFORMANCE_RATIO",
                "PREV_INTERVAL_RATIO", "ROLLING_24H_RATIO"]].head(10))
