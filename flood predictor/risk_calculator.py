import pandas as pd

from config import (
    FRI_WEIGHTS,
    KENYA_COUNTIES,
    RAINFALL_THRESHOLDS,
    RISK_COLORS,
    RISK_EMOJI,
    RISK_THRESHOLDS,
    SOIL_MOISTURE_THRESHOLDS,
)
from data.kenya_flood_drainage_dataset import get_county_drainage_score


def build_baseline():
    return {c["name"]: c for c in KENYA_COUNTIES}


def score_soil_moisture(sm):
    return min(sm / SOIL_MOISTURE_THRESHOLDS["saturated"], 1.0)


def score_antecedent(mm):
    return min(mm / RAINFALL_THRESHOLDS["antecedent_alert"], 1.0)


def score_peak(mm):
    return min(mm / RAINFALL_THRESHOLDS["peak_extreme"], 1.0)


def score_elevation(m):
    return 1.0 - min(m / 3000.0, 1.0)


def score_slope(slope_class):
    return 1.0 - ((slope_class - 1) / 4.0)


def drainage_risk(county_name, config_drainage):
    dataset_score = get_county_drainage_score(county_name)
    numeric = dataset_score if dataset_score is not None else config_drainage
    return numeric


def get_risk_level(fri):
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= fri < high:
            return level
    return "CRITICAL"


def compute_row(row, baseline):
    c = baseline.get(row["county"], {})

    elevation  = c.get("elevation", 1000)
    hist_flood = c.get("hist_flood", 0.30)
    near_water = c.get("near_water", 0.30)
    slope      = c.get("slope_class", 3)
    drain_num  = drainage_risk(row["county"], c.get("drainage", 3))
    drain_score = 1.0 - ((drain_num - 1) / 4.0)

    scores = {
        "soil_moisture":       score_soil_moisture(row["soil_moisture"]),
        "antecedent_rainfall": score_antecedent(row["antecedent_14day_mm"]),
        "peak_intensity":      score_peak(row["peak_daily_mm"]),
        "elevation":           score_elevation(elevation),
        "hist_flood":          float(hist_flood),
        "near_water":          float(near_water),
        "slope_class":         score_slope(slope),
    }

    fri = round(sum(scores[k] * FRI_WEIGHTS[k] for k in FRI_WEIGHTS) * 100, 2)
    return fri, scores


def calculate_all_risks(rainfall_df):
    baseline = build_baseline()
    results = []

    for _, row in rainfall_df.iterrows():
        fri, component_scores = compute_row(row, baseline)
        level = get_risk_level(fri)

        entry = {
            "county":              row["county"],
            "lat":                 row["lat"],
            "lon":                 row["lon"],
            "fri_score":           fri,
            "risk_level":          level,
            "risk_color":          RISK_COLORS[level],
            "risk_emoji":          RISK_EMOJI[level],
            "forecast_7day_mm":    row["forecast_7day_mm"],
            "peak_daily_mm":       row["peak_daily_mm"],
            "soil_moisture":       row["soil_moisture"],
            "antecedent_14day_mm": row["antecedent_14day_mm"],
            "daily_forecast":      row.get("daily_forecast", []),
        }
        entry.update({f"score_{k}": v for k, v in component_scores.items()})
        results.append(entry)

    df = pd.DataFrame(results)
    return df.sort_values("fri_score", ascending=False).reset_index(drop=True)
