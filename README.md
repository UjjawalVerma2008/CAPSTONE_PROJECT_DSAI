# Solar Power Generation & Plant Performance Analyzer

**Track:** Data Science
**Team members:** Himshikhar · Khushboo Khilnaney (742) · Satwik Srijan (722) · Ujjawal Verma (723)

---

## Problem Statement

Solar plant operators log generation and weather data every 15 minutes, but
raw numbers alone don't tell them what they actually need to know: *is each
inverter producing what it should be, given the weather right now, and if
not, why?*

Without a weather-normalized baseline, operators either miss real faults
(a cloudy afternoon looks the same as a tripped inverter in raw kW) or chase
false alarms (a small inverter looks "underperforming" next to a bigger one
that's simply rated higher). This project builds an end-to-end analyzer that:

- predicts the **expected** output of each inverter given current weather
  and its own recent behavior,
- flags **real, sustained** underperformance (not passing-cloud noise),
- benchmarks inverters against their peers under identical weather,
- estimates the **energy and financial loss** of each alert, and
- explains the **likely cause** and a **recommended action** in plain
  English — so a plant operator can act on it without needing to read a
  model's internals.

## Dataset / Reference Source

Two real solar plants' generation and weather-sensor logs (publicly
available "Solar Power Generation Data" style dataset), each with:

- 15-minute logging interval
- 22 inverters per plant
- ~34 days of readings
- Two files per plant: a **generation** log (DC/AC power, daily yield, per
  inverter) and a **weather sensor** log (irradiation, module temperature,
  ambient temperature — one sensor per plant, shared across all inverters)

```
data/
├── Plant_1_Generation_Data.csv
├── Plant_1_Weather_Sensor_Data.csv
├── Plant_2_Generation_Data.csv
└── Plant_2_Weather_Sensor_Data.csv
```

Two real-world data quirks were found by directly inspecting the CSVs (not
assumed) and handled explicitly in `data_pipeline.py`:

1. **Mixed date formats** — Plant 1's generation file uses `dd-mm-yyyy`;
   Plant 2's generation file and both weather files use `yyyy-mm-dd`. Each
   file is parsed with the correct `dayfirst` setting.
2. **DC_POWER unit mismatch (Plant 1 only)** — Plant 2's DC/AC ratio is
   ~1.02 (a normal inverter efficiency); Plant 1's is ~9.8–10x, which isn't
   physically real. Cross-checking `AC_POWER` against `DAILY_YIELD` confirms
   `AC_POWER` is correctly scaled for both plants, so **AC_POWER (kW) is
   used as "actual generation" everywhere**; Plant 1's `DC_POWER` is
   rescaled ÷10 only for reference/diagnostics.

## Tools Used

- **Python 3** — core language
- **pandas / NumPy** — data ingestion, cleaning, feature engineering
- **scikit-learn** (`RandomForestRegressor`) — expected-output prediction
- **SHAP** *(optional)* — per-alert local explainability
- **Plotly Express** — interactive charts
- **Streamlit** — dashboard front end (`app.py` and `app_simple.py`)

## Project Workflow

```
capstone/
├── data/                    # Plant_{1,2}_Generation/Weather CSVs
├── src/
│   ├── data_pipeline.py     # ingestion, cleaning, feature engineering
│   ├── ml_model.py          # Module 1 + 9: expected-output model, explainability
│   └── performance_engine.py# Modules 2-8: gap, benchmarking, alerts, loss, cause, action
├── app.py                   # Module 10: full Streamlit dashboard
├── app_simple.py            # Simplified, tab-based business-user front end
├── run_app_simple.bat       # Windows one-click launcher for app_simple.py
└── requirements.txt
```

End-to-end flow:

1. **Ingest & clean** (`data_pipeline.py`) — load both plants, fix date
   formats and the DC_POWER scale issue, merge generation with weather.
2. **Feature engineer** — time features (hour, decimal hour), each
   inverter's estimated rated capacity (95th percentile output at high
   irradiance), capacity-normalized `PERFORMANCE_RATIO`, and look-back-only
   history features (`PREV_INTERVAL_RATIO`, `ROLLING_24H_RATIO`) so there's
   no data leakage.
3. **Model expected output** (`ml_model.py`) — a Random Forest predicts the
   performance ratio a *healthy* inverter should show given weather + time +
   its own recent history, then converts that back to kW.
4. **Analyze performance** (`performance_engine.py`) — gap calculation and
   Normal/Warning/Critical classification, peer benchmarking, plant-wide
   event detection, persistence filtering, energy-loss and financial-impact
   estimation, rule-based cause identification, and recommended-action text.
5. **Present** (`app.py` / `app_simple.py`) — interactive Streamlit
   dashboards for exploring all of the above.

## AI/ML/Agent/Software Component

**Module 1 — Weather-Normalized Expected Output Model** (`ml_model.py`)
A `RandomForestRegressor` predicts each inverter's capacity-normalized
`PERFORMANCE_RATIO` from irradiation, module temperature, ambient
temperature, hour of day, and two inverter-history features. The prediction
is converted back to kW using that inverter's estimated capacity to produce
`EXPECTED_POWER`. This is the core AI component because "expected output"
isn't a fixed number — it depends non-linearly on sunlight, thermal
derating, and each inverter's own recent behavior; a fixed rule can't
capture that jointly.

**Two-pass "healthy data" training.** Fitting once on all data would let the
model partly learn faults as normal (expected output quietly drops to match
reality during downtime). Instead, a first-pass model is fit, the worst
residual outliers (~likely genuine faults) are trimmed, and a second model
is refit on the cleaner subset — so "expected" reflects healthy operation,
not an average that already includes the faults being detected.

**Honest evaluation.** `evaluate_expected_output_model` reports MAE/MSE/
RMSE/R² using a **chronological** split (most recent days per plant held
out), not a random row split, since consecutive 15-minute rows are strongly
autocorrelated and a random split would leak information and overstate
accuracy.

**Module 9 — Explainable ML.** Global feature importance is always
available. Per-alert local explanations use SHAP (`TreeExplainer`) when
installed, showing which factor (irradiation, temperature, or the
inverter's own recent history) pushed a specific alert's expected output up
or down.

**Modules 2–8 (`performance_engine.py`) are deliberately rule-based**, since
these are business/physics rules that need to stay interpretable and
auditable by a plant operator:
- Gap % thresholds → Normal / Warning / Critical
- Peer benchmarking (inverters sharing one weather sensor should perform
  similarly at the same instant)
- Plant-wide event detection (most inverters underperforming together →
  likely a grid/curtailment event, not many separate faults)
- Persistence filter (≥1 hour of sustained underperformance by default,
  filters out passing-cloud noise)
- Priority-ordered cause identification (plant-wide event → inverter
  trip/sensor fault → overheating → peer-relative fault → weather → generic)
- Energy loss (kWh) and financial impact (configurable tariff)
- Auto-generated, human-readable recommended-action text per alert

## How to Run the Project

```bash
pip install -r requirements.txt

# Full dashboard (all modules, plant/date/inverter filters, explainability charts)
streamlit run app.py

# Simplified, tab-based business-user view (weather relationship, output
# prediction, generation forecasting, underperformance alerts)
streamlit run app_simple.py
```

Both apps reuse the same `src/data_pipeline.py`, `src/ml_model.py`, and
`src/performance_engine.py` untouched, so results are identical between the
two front ends.

**Windows one-click launcher.** For the simplified view, `run_app_simple.bat`
is included so non-technical users don't need to touch a terminal:

```
capstone/
└── run_app_simple.bat   # place in the same folder as app_simple.py
```

Just double-click it. It will:

1. `cd` into its own folder (wherever it's placed),
2. check that Python is installed (and points you to the installer if not),
3. check that Streamlit is installed, installing everything in
   `requirements.txt` (or `streamlit pandas numpy plotly` as a fallback) if
   it's missing, and
4. run `streamlit run app_simple.py`, opening the dashboard in a browser tab
   automatically. Closing the console window stops the app.

## Demo Screenshots

*(Add screenshots of the dashboard here — e.g. the generation trend chart,
inverter benchmarking view, and the alerts table with recommended actions.)*

## Results and Insights

- **Irradiation** dominates output as expected; **module temperature**
  correlates with a *lower* performance ratio at high values, consistent
  with real thermal derating rather than a data artifact.
- Chronological held-out evaluation gives an honest read on generalization
  to unseen days, reported both on the unitless performance ratio and on
  the back-converted kW scale (the units the downstream loss/financial
  modules actually consume).
- Persistent alerts cluster around a small number of root causes (peer-
  relative inverter faults, overheating, and occasional plant-wide events),
  which the rule-based cause/action layer turns directly into a
  prioritized, technician-readable worklist with estimated kWh and
  financial impact per alert.

## Limitations

- Rated inverter capacity is **estimated** from historical high-irradiance
  output (95th percentile), not taken from a manufacturer nameplate — a
  reasonable proxy, but not ground truth.
- The dataset covers only **two plants over ~34 days**, so seasonal effects
  (e.g. very different sun angles or temperatures across a full year) are
  not captured.
- SHAP-based local explanations are optional and silently skipped if the
  `shap` package isn't installed; only global feature importance is then
  available.
- Cause identification is rule-based and interpretable, but not
  exhaustive — some fault types (e.g. specific electrical failure modes)
  are grouped into a generic "inspect connections" bucket.

## Future Improvements

- Incorporate a full year (or more) of data to capture seasonal patterns
  and improve the expected-output model's robustness.
- Add manufacturer-specified inverter capacity where available, instead of
  relying solely on the estimated 95th-percentile proxy.
- Extend cause identification with additional weather signals (e.g. wind
  speed, soiling/rain data) to further disambiguate fault types.
- Add automated alerting (e.g. email/SMS) when a new persistent alert is
  detected, rather than requiring an operator to check the dashboard.
- Package the model training/evaluation pipeline as a scheduled job so the
  expected-output model can be periodically retrained on fresh data.
