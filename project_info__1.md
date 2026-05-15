# Kenya Flood Risk Prediction System — Codebase Overview

## Summary
This repository contains a Python capstone application that predicts flood risk for Kenya’s counties by combining (1) live rainfall forecast/antecedent rainfall from Open-Meteo, (2) a heuristic Flood Risk Index (FRI) computed from terrain/drainage metadata plus rainfall-derived features, and (3) a trained RandomForest ML model that outputs a “ML flood probability” for each county. It then renders an interactive Folium map and two matplotlib charts (risk ranking + 7-day forecast), and finally prints a Rich-formatted “alert” table and a text report.

Who uses it: a developer/runs-it-user executes `flood predictor/main_1.py` directly; output is written to `outputs/` (or configured `OUTPUT_DIR`) and displayed in the terminal as Rich spinners and tables.

## Architecture
**Primary pattern:** a simple procedural pipeline orchestrated in `main_1.py`, with “modules” implementing each stage. The terminal UX is implemented using **Rich**’s `Progress` spinner widgets.

**Subsystems**
1. **Orchestration / CLI UX**: `flood predictor/main_1.py`
2. **Data acquisition + caching**: `flood predictor/data_loader.py`
3. **Heuristic risk scoring (FRI)**: `flood predictor/risk_calculator.py`
4. **ML probability prediction**: `flood predictor/flood_predictor.py`
5. **Visualization**: `flood predictor/visualizer.py` (Folium + matplotlib)
6. **Alerting/reporting**: `flood predictor/alert_system.py` (Rich console panels/tables + text report)

**Technology stack**
- Python
- `rich` for CLI progress spinners/tables
- `requests` for Open-Meteo API calls
- `pandas`/`numpy` for data transformations
- `scikit-learn` (RandomForest + StandardScaler + joblib serialization)
- `folium` for interactive map (saved as HTML)
- `matplotlib` for charts (saved as image files)
- Additional local data/config dependencies:
  - `config` module (constants such as URLs, thresholds, file paths)
  - `data.kenya_flood_drainage_dataset` (training samples, drainage scores, and worst-subcounty lookups)

**How execution starts**
- Entry point is `main_1.py`:
  - `setup()` ensures `outputs/` and `data/` directories exist.
  - `main()` prints a header.
  - A Rich `Progress` context manager starts 4 concurrent spinner “tasks” sequentially (they spin until each stage completes).
  - After the progress block finishes, `run_alerts(risk_df)` is called to render the summary/alert table and save the text report.

## Directory Structure
The repository’s important runtime code lives under the “flood predictor” folder. Conceptually:

```
project-root/
├── Python_Capstone_Project/
│   ├── flood predictor/
│   │   ├── main_1.py                — Rich progress spinners + pipeline orchestration
│   │   ├── data_loader.py         — Open-Meteo fetch + per-day parsing + JSON caching
│   │   ├── risk_calculator.py      — Heuristic FRI scoring + risk level mapping
│   │   ├── flood_predictor.py     — RandomForest training + probability prediction
│   │   ├── visualizer.py          — Folium map + matplotlib charts
│   │   └── alert_system.py       — Rich summary/table + saves text report
│   └── (supporting modules outside “flood predictor” folder)
└── (config.py, data/..., outputs/, model/..., etc. expected at runtime)
```

Note: your exploration shows `Python_Capstone_Project/flood predictor/README.md` is referenced but not present; the meaningful entrypoint documentation is `Python_Capstone_Project/README.md` (very minimal).

## Key Abstractions

### `main_1.main()` (pipeline orchestrator)
- **File**: `Python_Capstone_Project/flood predictor/main_1.py` (main function)
- **Responsibility**: Orchestrates the full end-to-end pipeline and controls terminal UX (Rich spinners).
- **Interface (conceptual)**:
  - `setup()` creates `outputs` and `data`
  - `main()` prints header, runs a Rich `Progress` context, then calls `run_alerts(risk_df)`
- **Lifecycle**: executed once per run; all intermediate results are local variables.
- **Used by**: executed directly via `if __name__ == "__main__": main()`.

**What matters for spinners/progress output**
- Rich `Progress` uses:
  - `SpinnerColumn()` (animated spinner)
  - `TextColumn("[progress.description]{task.description}")` (shows description text)
  - `transient=True` (the progress UI is removed when leaving the context)
- Each stage does:
  1. `t1 = progress.add_task("Fetching...", total=None)` then calls `load_all_counties(...)`
  2. `progress.update(t1, description="[green]Rainfall data loaded[/green]")`
  3. Repeat for risk calculation, ML predictions, visualization outputs.

### `load_all_counties(use_cache=True)`
- **File**: `Python_Capstone_Project/flood predictor/data_loader.py`
- **Responsibility**: Fetches forecast + antecedent rainfall per county and returns a `pandas.DataFrame`. Implements a simple daily JSON cache.
- **Interface**:
  - `load_all_counties(use_cache=True)` → `pd.DataFrame` with columns including:
    - `county`, `lat`, `lon`
    - `forecast_7day_mm`, `peak_daily_mm`, `soil_moisture`, `antecedent_14day_mm`, `daily_forecast`
- **Lifecycle**: called once per program run by `main_1.py`.
- **Used by**: `main_1.py` → `risk_calculator.calculate_all_risks`.

**Non-obvious behavior**
- It sleeps `0.25s` between counties (to be gentle to Open-Meteo).
- Any exception during a county fetch results in `fallback_record(county)` (so the pipeline continues rather than failing the whole run).
- Caching is “by date”:
  - If `DATA_CACHE_FILE` exists and its `"date"` matches `datetime.today().date()`, cached data is returned.

### `calculate_all_risks(rainfall_df)`
- **File**: `Python_Capstone_Project/flood predictor/risk_calculator.py`
- **Responsibility**: Computes the heuristic Flood Risk Index (FRI) and maps it to discrete risk levels, colors, and emojis.
- **Interface**:
  - `calculate_all_risks(rainfall_df)` → `pd.DataFrame` sorted by `fri_score` descending
- **Key computations**:
  - Converts continuous rainfall/terrain features into 0–1 scores:
    - `soil_moisture`: `min(sm / saturated_threshold, 1.0)`
    - `antecedent_rainfall`: `min(mm / antecedent_alert_threshold, 1.0)`
    - `peak_intensity`: `min(mm / peak_extreme_threshold, 1.0)`
    - `elevation`: `1.0 - min(elevation/3000, 1.0)` (higher elevation reduces risk score)
    - `slope_class`: `1.0 - ((slope_class - 1)/4.0)`
    - drainage is obtained via dataset lookup (`get_county_drainage_score`) and then converted to a 1..0 scale
  - `fri = round(sum(scores[k] * FRI_WEIGHTS[k] for k in FRI_WEIGHTS) * 100, 2)`

**Non-obvious behavior**
- The “baseline” is built from `KENYA_COUNTIES` config and then augmented with drainage dataset values via `drainage_risk()`.
- If the dataset drainage score lookup returns `None`, it falls back to `config_drainage`.
- Final `risk_level` uses `RISK_THRESHOLDS` iteration with `low <= fri < high`; if no interval matches, it returns `"CRITICAL"`.

### `train_model(force=False)` + `predict_flood_probability(risk_df)`
- **File**: `Python_Capstone_Project/flood predictor/flood_predictor.py`
- **Responsibility**:
  - `train_model()` trains and caches a RandomForest + StandardScaler in a single serialized bundle (joblib).
  - `predict_flood_probability(risk_df)` uses that model to generate per-county ML probability and a label-derived predicted level.
- **Interfaces**:
  - `train_model(force=False)` → either loaded model bundle or newly trained bundle
  - `predict_flood_probability(risk_df)` → `pd.DataFrame` with `county`, `ml_flood_prob`, `ml_predicted_level`
- **Key behavior**:
  - If `MODEL_FILE` exists and `force=False`, training is skipped and the saved model is loaded.
  - ML features used for prediction are **derived from config baseline**, not from the computed FRI scores:
    - `features = [c.get("drainage"), c.get("elevation"), c.get("near_water"), c.get("slope_class"), c.get("hist_flood")]`
  - The output `flood_prob` is computed oddly but intentionally:
    - If the model produces more than 3 probability entries: `proba[2] + proba[3]`
    - Else: `proba[-1]`
  - This implies the model class indices are assumed to correspond to risk tiers where indices 2+3 represent “flood” or “high+critical”.

**Important gotcha for developers**
- There is an inconsistency between training feature extraction and prediction feature extraction:
  - Training: `extract_features()` uses `DRAINAGE_QUALITY_MAP.get(sample["Drainage_Quality"], 2)` as the first feature.
  - Prediction: directly uses `c.get("drainage", 3)` from the county config.
  - If `c["drainage"]` is on a different scale than `Drainage_Quality` mapping, predictions may be skewed—this may be “by design” but it’s a potential integration risk.

### `create_risk_map`, `create_risk_chart`, `create_forecast_chart`
- **File**: `Python_Capstone_Project/flood predictor/visualizer.py`
- **Responsibility**: Create visual artifacts and return file paths to embed/print later.
- **Interfaces**:
  - `create_risk_map(risk_df)` → saves `MAP_HTML_FILE` and returns that path
  - `create_risk_chart(risk_df)` → saves `RISK_CHART_FILE`
  - `create_forecast_chart(risk_df)` → saves `FORECAST_CHART_FILE`
- **Key behavior**:
  - Map popup includes both FRI components and ML probability if present.
  - Charting uses `risk_df.head(20)` and uses `risk_color` for bar coloring.
  - Forecast chart filters to top risk counties (`risk_level in ["CRITICAL","HIGH"]`) and expects `daily_forecast` to be a 7-length list.

### `run_alerts(risk_df)` (Rich output + report saving)
- **File**: `Python_Capstone_Project/flood predictor/alert_system.py`
- **Responsibility**: Produces a Rich summary panel, a Rich alert table, and writes a text report file.
- **Interfaces**:
  - `run_alerts(risk_df)` → returns filtered `alerted` DataFrame (HIGH/CRITICAL)
- **Key behavior**:
  - Summary panel uses counts from `risk_df["risk_level"].value_counts()`.
  - Table shows “Worst sub-county” and “Flood mechanism” by calling `get_worst_subcounties(row["county"], top_n=1)`.
  - `save_report()` writes `REPORT_TXT_FILE` to disk under `OUTPUT_DIR`.

## Data Flow
1. **User runs** `Python_Capstone_Project/flood predictor/main_1.py`
2. `main_1.setup()` creates directories (`outputs`, `data`)
3. Rich `Progress` starts 4 spinner tasks (total=None, transient UI)
4. `t1`:
   - `data_loader.load_all_counties(use_cache=True)` fetches Open-Meteo data per county
   - Parses:
     - forecast precipitation sum (7-day)
     - peak daily precipitation
     - hourly soil moisture mean over the first 24 samples
     - antecedent 14-day rainfall sum
   - Caches per date to `DATA_CACHE_FILE`
5. `t2`:
   - `risk_calculator.calculate_all_risks(rainfall_df)` computes `fri_score`, `risk_level`, `risk_color`, `risk_emoji`
   - Sorts by `fri_score` descending
6. `t3`:
   - `flood_predictor.predict_flood_probability(risk_df)` trains/loads ML model bundle
   - Returns `ml_flood_prob` + `ml_predicted_level`
   - `main_1.py` merges predictions into `risk_df` on `county`
7. `t4`:
   - `visualizer.create_risk_map(risk_df)` → HTML map
   - `visualizer.create_risk_chart(risk_df)` → bar chart image
   - `visualizer.create_forecast_chart(risk_df)` → line chart image
8. Progress UI exits (transient), then:
9. `alert_system.run_alerts(risk_df)` prints a Rich panel/table and saves the text report
10. `main_1.py` prints the final “Interactive map / Risk chart / Forecast chart” file paths

## Non-Obvious Behaviors & Design Decisions

- **Spinners don’t represent % progress**: each stage uses `total=None`, so spinners spin until the stage completes. This is intentional because the pipeline stages are not naturally “percentage” progress.
- **Transient progress UI**: `transient=True` means the spinner block disappears after all four stages finish; the “final result” is printed after the progress context closes.
- **Robustness via fallback records**: if a county API fetch fails, that county still appears with zeros/default soil moisture/empty rainfall forecast. This prevents total failure due to intermittent API issues.
- **Daily cache**: caching is keyed only by calendar date; it does not include parameters such as `FORECAST_DAYS` or `ANTECEDENT_DAYS`. If those constants change between runs, the cache may become inconsistent (but still returns same schema).
- **ML model feature mapping asymmetry**: training and prediction drainage features come from different sources/scales (training uses a mapped `Drainage_Quality`, prediction uses config `drainage`). If you see degraded ML performance, this is a first place to check.
- **Probability aggregation**: `flood_prob = proba[2] + proba[3]` assumes class ordering in the RandomForest matches the intended “flood/higher risk” classes.
- **Visualization depends on daily_forecast length**: forecast chart only plots counties where `daily_forecast` is a list of length 7. The fallback record sets this to `[0.0] * FORECAST_DAYS`, so it will work as long as `FORECAST_DAYS == 7`.

## Module Reference
| File | Purpose |
|---|---|
| `flood predictor/main_1.py` | Program entrypoint; Rich spinner/progress UI; orchestrates the full pipeline |
| `flood predictor/data_loader.py` | Open-Meteo data fetch, parsing, per-day JSON cache; returns county rainfall/features |
| `flood predictor/risk_calculator.py` | Heuristic Flood Risk Index scoring; risk levels/colors/emojis; sorting |
| `flood predictor/flood_predictor.py` | Train/load RandomForest + scaler; compute ML flood probability |
| `flood predictor/visualizer.py` | Save Folium interactive map + matplotlib charts |
| `flood predictor/alert_system.py` | Rich summary + alert table; save text report; worst sub-county annotation |

## Suggested Reading Order
1. `Python_Capstone_Project/flood predictor/main_1.py` — see how progress spinners are wired and what the pipeline stages are.
2. `Python_Capstone_Project/flood predictor/data_loader.py` — understand caching + parsing that feeds everything else.
3. `Python_Capstone_Project/flood predictor/risk_calculator.py` — learn how FRI components and weights translate into risk levels.
4. `Python_Capstone_Project/flood predictor/flood_predictor.py` — understand ML training/loading and the exact feature vector used for prediction.
5. `Python_Capstone_Project/flood predictor/visualizer.py` — confirm what fields are expected for charts and map popups.
6. `Python_Capstone_Project/flood predictor/alert_system.py` — understand final CLI output and report generation.
