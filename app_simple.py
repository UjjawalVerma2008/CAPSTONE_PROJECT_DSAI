"""
app_simple.py
=============
Simple Business-User Front End for the Solar Performance Analyzer.

Four interactive tabs, one per requested area:
  1. Weather Relationship Analysis  - how irradiation/temperature relate to output
  2. Output Prediction               - "what would this inverter produce under X weather?"
  3. Generation Forecasting          - expected vs actual generation curve for a chosen day
  4. Underperformance Detection      - alerts, causes, recommended actions

Reuses the existing pipeline/model/engine untouched — this file is a thinner,
tab-based UI on top of the same modules app.py uses.

Run with:  streamlit run app_simple.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_pipeline import load_and_prepare, DATA_DIR
from ml_model import train_expected_output_model, score_expected_output, FEATURES
from performance_engine import run_full_analysis, PERSISTENCE_INTERVALS

st.set_page_config(page_title="Solar Analyzer — Simple View", layout="wide", page_icon="☀️")


# ----------------------------------------------------------------------------
# Cached pipeline (same stages as the main dashboard, so results match)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading plant data...")
def get_data():
    return load_and_prepare(DATA_DIR)


@st.cache_resource(show_spinner="Training expected-output model...")
def get_model(df):
    return train_expected_output_model(df)


@st.cache_data(show_spinner="Scoring + running alert analysis...")
def get_analyzed(df, _model, tariff, persistence):
    scored = score_expected_output(df, _model)
    return run_full_analysis(scored, tariff_per_kwh=tariff, min_persistence_intervals=persistence)


raw = get_data()
model, diag = get_model(raw)

st.title("☀️ Solar Performance — Simple View")
st.caption("A business-friendly view: how weather drives output, what output to expect, "
           "the generation curve for a chosen day, and which inverters need attention.")

# Global controls used across tabs
c1, c2, c3 = st.columns(3)
plant = c1.selectbox("Plant", sorted(raw["PLANT_LABEL"].unique()))
tariff = c2.number_input("Tariff (currency/kWh)", 0.5, 50.0, 7.0, 0.5)
persistence = c3.slider("Alert persistence (15-min intervals)", 1, 12, PERSISTENCE_INTERVALS)

analyzed = get_analyzed(raw, model, tariff, persistence)
plant_df = analyzed[analyzed["PLANT_LABEL"] == plant]

tab1, tab2, tab3, tab4 = st.tabs([
    "🌤️ Weather Relationship", "🔮 Output Prediction",
    "📈 Generation Forecast", "🚨 Underperformance",
])

# ----------------------------------------------------------------------------
# TAB 1 — Weather Relationship Analysis
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("How weather drives generation")
    daylight = plant_df[plant_df["IRRADIATION"] > 0.02]

    colA, colB = st.columns(2)
    with colA:
        fig = px.scatter(
            daylight.sample(min(5000, len(daylight)), random_state=1),
            x="IRRADIATION", y="AC_POWER", color="MODULE_TEMPERATURE",
            opacity=0.5, title="Irradiation vs Actual Output (colored by module temp)",
            labels={"AC_POWER": "Power (kW)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig2 = px.scatter(
            daylight.sample(min(5000, len(daylight)), random_state=1),
            x="MODULE_TEMPERATURE", y="PERFORMANCE_RATIO",
            opacity=0.5, title="Module Temperature vs Performance Ratio",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Correlation with actual output**")
    corr_cols = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_POWER"]
    corr = daylight[corr_cols].corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      title="Correlation matrix (daylight readings)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Irradiation is expected to dominate. High module-temperature correlation with a "
               "*lower* ratio indicates thermal derating — real, not a data issue.")

# ----------------------------------------------------------------------------
# TAB 2 — Output Prediction ("what should this inverter produce?")
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Predict expected output under given weather conditions")
    st.caption("Pick an inverter and a weather scenario — the trained model predicts what that "
               "inverter *should* produce, using its own capacity and recent-history behaviour as a baseline.")

    inverters = sorted(plant_df["INVERTER_ID"].unique())
    inv = st.selectbox("Inverter", inverters)
    inv_hist = plant_df[plant_df["INVERTER_ID"] == inv]
    inv_capacity = inv_hist["INVERTER_CAPACITY_KW"].iloc[0]
    inv_own_avg_ratio = inv_hist["PERFORMANCE_RATIO"].mean()

    p1, p2, p3, p4 = st.columns(4)
    irr = p1.slider("Irradiation", 0.0, 1.2, 0.6, 0.01)
    mod_t = p2.slider("Module Temp (°C)", 15.0, 75.0, 45.0, 0.5)
    amb_t = p3.slider("Ambient Temp (°C)", 15.0, 45.0, 30.0, 0.5)
    hour = p4.slider("Hour of day", 0.0, 23.75, 12.0, 0.25)

    st.caption(f"Recent-history features default to this inverter's own historical average "
               f"performance ratio ({inv_own_avg_ratio:.2f}) — override if simulating a fault scenario.")
    override_hist = st.checkbox("Simulate a recent fault (low recent history)", value=False)
    hist_ratio = 0.1 if override_hist else inv_own_avg_ratio

    if st.button("Predict expected output", type="primary"):
        row = pd.DataFrame([{
            "IRRADIATION": irr, "MODULE_TEMPERATURE": mod_t, "AMBIENT_TEMPERATURE": amb_t,
            "HOUR_DECIMAL": hour, "PREV_INTERVAL_RATIO": hist_ratio, "ROLLING_24H_RATIO": hist_ratio,
        }])[FEATURES]
        pred_ratio = float(np.clip(model.predict(row)[0], 0, 1.5))
        pred_power = pred_ratio * inv_capacity

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Performance Ratio", f"{pred_ratio:.2f}")
        m2.metric("Predicted Output", f"{pred_power:,.0f} kW")
        m3.metric("Inverter Rated Capacity", f"{inv_capacity:,.0f} kW")

# ----------------------------------------------------------------------------
# TAB 3 — Generation Forecasting
#   Mode A: replay a past day's real weather (nowcast, for sanity-checking)
#   Mode B: manually enter a weather forecast (irradiation/temps per hour) and
#           get a predicted generation curve for a future day you don't have
#           real readings for yet.
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Generation forecast")
    mode = st.radio(
        "Mode",
        ["Manual forecast (enter tomorrow's expected weather yourself)",
         "Replay a past day (uses real recorded weather)"],
        horizontal=False,
    )

    # ---- Mode B: manual forecast ----
    if mode.startswith("Manual"):
        st.caption("Enter your own irradiation/temperature forecast per hour (e.g. from a weather "
                   "provider or your own estimate). The model converts each hour into predicted output "
                   "for the chosen inverter. Recent-history features are chained hour-to-hour from the "
                   "inverter's own historical average, since no real sequential history exists yet for a future day.")

        inv3 = st.selectbox("Inverter", sorted(plant_df["INVERTER_ID"].unique()), key="mf_inv")
        inv3_hist = plant_df[plant_df["INVERTER_ID"] == inv3]
        inv3_capacity = inv3_hist["INVERTER_CAPACITY_KW"].iloc[0]
        inv3_avg_ratio = inv3_hist["PERFORMANCE_RATIO"].mean()

        # Sensible daylight-shaped defaults the user can overwrite; irradiation
        # roughly bell-curved between sunrise/sunset, temps following a typical day.
        default_hours = list(range(6, 19))
        default_irr = [0.05, 0.20, 0.40, 0.58, 0.72, 0.82, 0.85, 0.80, 0.68, 0.52, 0.32, 0.14, 0.03]
        default_mod = [22, 28, 34, 40, 46, 50, 52, 50, 46, 40, 34, 28, 23]
        default_amb = [20, 22, 24, 26, 28, 29, 30, 30, 29, 27, 25, 23, 21]

        template = pd.DataFrame({
            "Hour": default_hours,
            "Irradiation (0-1)": default_irr,
            "Module Temp (°C)": default_mod,
            "Ambient Temp (°C)": default_amb,
        })

        st.markdown("**Forecast input — edit any cell, add/remove rows as needed:**")
        edited = st.data_editor(
            template, num_rows="dynamic", use_container_width=True, key="manual_forecast_editor",
        )

        if st.button("Generate forecast", type="primary"):
            edited = edited.dropna().sort_values("Hour").reset_index(drop=True)
            if edited.empty:
                st.warning("Add at least one row of forecast weather first.")
            else:
                prev_ratio = inv3_avg_ratio
                pred_rows = []
                for _, r in edited.iterrows():
                    feat_row = pd.DataFrame([{
                        "IRRADIATION": float(r["Irradiation (0-1)"]),
                        "MODULE_TEMPERATURE": float(r["Module Temp (°C)"]),
                        "AMBIENT_TEMPERATURE": float(r["Ambient Temp (°C)"]),
                        "HOUR_DECIMAL": float(r["Hour"]),
                        "PREV_INTERVAL_RATIO": prev_ratio,
                        "ROLLING_24H_RATIO": inv3_avg_ratio,
                    }])[FEATURES]
                    pred_ratio = float(np.clip(model.predict(feat_row)[0], 0, 1.5))
                    prev_ratio = pred_ratio  # chain to next hour
                    pred_rows.append({
                        "Hour": r["Hour"], "Predicted Ratio": pred_ratio,
                        "Predicted Power (kW)": pred_ratio * inv3_capacity,
                    })

                forecast_df = pd.DataFrame(pred_rows)
                fig4 = px.line(forecast_df, x="Hour", y="Predicted Power (kW)", markers=True,
                                title=f"Manually forecast generation — Inverter {inv3}")
                st.plotly_chart(fig4, use_container_width=True)

                # Trapezoidal integration over unevenly-spaced hours -> kWh.
                # (np.trapz was removed in NumPy 2.0+ in favor of np.trapezoid,
                # so this is computed manually to work on either version.)
                hours = forecast_df["Hour"].to_numpy()
                power = forecast_df["Predicted Power (kW)"].to_numpy()
                if len(hours) > 1:
                    dt = np.diff(hours)
                    total_kwh = float(np.sum(dt * (power[:-1] + power[1:]) / 2.0))
                else:
                    total_kwh = 0.0

                m1, m2 = st.columns(2)
                m1.metric("Total forecast energy", f"{total_kwh:,.0f} kWh")
                m2.metric("Estimated value at tariff", f"{total_kwh * tariff:,.0f} currency units")
                st.dataframe(forecast_df, use_container_width=True)

    # ---- Mode A: replay a real past day ----
    else:
        st.caption("This replays the model against a chosen day's actual recorded weather to produce "
                   "an expected-generation curve alongside what was really produced — useful for "
                   "sanity-checking the model, not a forecast of an unknown future day.")

        dates = sorted(plant_df["DATE"].unique())
        day = st.select_slider("Date", options=dates, value=dates[0])
        scope = st.radio("Scope", ["Whole plant (sum of all inverters)", "Single inverter"], horizontal=True)

        day_df = plant_df[plant_df["DATE"] == day]
        if scope == "Single inverter":
            inv2 = st.selectbox("Inverter", sorted(day_df["INVERTER_ID"].unique()), key="fc_inv")
            day_df = day_df[day_df["INVERTER_ID"] == inv2]
        curve = day_df.groupby("DATE_TIME")[["AC_POWER", "EXPECTED_POWER"]].sum().reset_index()

        fig4b = px.line(curve, x="DATE_TIME", y=["AC_POWER", "EXPECTED_POWER"],
                         labels={"value": "Power (kW)", "DATE_TIME": "Time", "variable": "Series"},
                         title=f"Actual vs Expected Output — {day}")
        st.plotly_chart(fig4b, use_container_width=True)

        gap_kwh = (curve["EXPECTED_POWER"] - curve["AC_POWER"]).clip(lower=0).sum() * 0.25
        st.metric("Estimated lost energy that day", f"{gap_kwh:,.0f} kWh", f"≈ {gap_kwh * tariff:,.0f} currency units")

# ----------------------------------------------------------------------------
# TAB 4 — Underperformance Detection
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Alerts & recommended actions")
    persistent = plant_df[plant_df["PERSISTENT_ALERT"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Persistent Alerts", f"{len(persistent):,}")
    k2.metric("Critical Readings", f"{(plant_df['STATUS'] == 'Critical').sum():,}")
    k3.metric("Total Lost Energy", f"{plant_df['LOSS_KWH'].sum():,.0f} kWh")
    k4.metric("Total Financial Impact", f"{plant_df['FINANCIAL_IMPACT'].sum():,.0f}")

    colE, colF = st.columns(2)
    with colE:
        cause_counts = persistent["CAUSE"].value_counts().reset_index()
        cause_counts.columns = ["Cause", "Count"]
        fig5 = px.bar(cause_counts, x="Count", y="Cause", orientation="h", title="Alerts by likely cause")
        st.plotly_chart(fig5, use_container_width=True)
    with colF:
        worst = (persistent.groupby("INVERTER_ID")["LOSS_KWH"].sum()
                 .sort_values(ascending=False).head(10).reset_index())
        fig6 = px.bar(worst, x="INVERTER_ID", y="LOSS_KWH", title="Top 10 inverters by lost energy (kWh)")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Alert detail**")
    if len(persistent):
        show = persistent[["DATE_TIME", "INVERTER_ID", "STATUS", "GAP_PCT", "LOSS_KWH",
                            "FINANCIAL_IMPACT", "CAUSE", "RECOMMENDED_ACTION"]].copy()
        show["GAP_PCT"] = (show["GAP_PCT"] * 100).round(1)
        st.dataframe(show.sort_values("DATE_TIME", ascending=False), use_container_width=True, height=350)
        st.success(persistent.iloc[0]["RECOMMENDED_ACTION"])
    else:
        st.info("No persistent alerts for this plant with the current settings.")
