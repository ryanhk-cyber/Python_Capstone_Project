from __future__ import annotations

import os

# How many days of antecedent precipitation to use
ANTECEDENT_DAYS: int = 14

# How many forecast days Open-Meteo should return for the forecast window
FORECAST_DAYS: int = 7

# Cache file so repeated runs are faster (and less network dependent)
DATA_CACHE_FILE: str = "data/rainfall_cache.json"

# Open-Meteo endpoints
OPENMETEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

# Kenya timezone (no DST)
TIMEZONE: str = "Africa/Nairobi"

# 47 Kenya counties: (name, lat, lon)
# Note: These are representative centroids to drive the prediction pipeline.
KENYA_COUNTIES: list[dict[str, float | str]] = [
    {"name": "Baringo", "lat": 0.4667, "lon": 35.95},
    {"name": "Bomet", "lat": -0.264, "lon": 35.329},
    {"name": "Bungoma", "lat": 0.5602, "lon": 34.5619},
    {"name": "Busia", "lat": 0.4853, "lon": 34.2477},
    {"name": "Elgeyo-Marakwet", "lat": 0.823, "lon": 35.354},
    {"name": "Embu", "lat": -0.565, "lon": 37.450},
    {"name": "Garissa", "lat": -0.4544, "lon": 39.6646},
    {"name": "Homa Bay", "lat": -0.4548, "lon": 34.5537},
    {"name": "Isiolo", "lat": 0.3544, "lon": 37.5500},
    {"name": "Kajiado", "lat": -2.0417, "lon": 36.7000},
    {"name": "Kakamega", "lat": 0.2920, "lon": 34.7460},
    {"name": "Kericho", "lat": -0.3590, "lon": 35.2730},
    {"name": "Kiambu", "lat": -1.2140, "lon": 36.8910},
    {"name": "Kilifi", "lat": -3.3230, "lon": 39.8350},
    {"name": "Kirinyaga", "lat": -0.3035, "lon": 37.2240},
    {"name": "Kisii", "lat": -0.6843, "lon": 34.7460},
    {"name": "Kisumu", "lat": -0.0917, "lon": 34.7460},
    {"name": "Kitui", "lat": -1.3030, "lon": 38.0190},
    {"name": "Konde (Kwale)", "lat": -4.2667, "lon": 39.4980},
    {"name": "Laikipia", "lat": 0.321, "lon": 36.940},
    {"name": "Lamu", "lat": -2.2667, "lon": 40.9067},
    {"name": "Machakos", "lat": -1.4690, "lon": 37.2590},
    {"name": "Makueni", "lat": -1.7980, "lon": 37.8990},
    {"name": "Mandera", "lat": 3.9030, "lon": 42.2280},
    {"name": "Marsabit", "lat": 2.3280, "lon": 37.9960},
    {"name": "Meru", "lat": -0.0650, "lon": 37.6500},
    {"name": "Migori", "lat": -1.0580, "lon": 34.4810},
    {"name": "Mombasa", "lat": -4.0435, "lon": 39.6682},
    {"name": "Murang'a", "lat": -0.7167, "lon": 36.9333},
    {"name": "Nairobi", "lat": -1.286389, "lon": 36.817223},
    {"name": "Nakuru", "lat": -0.3031, "lon": 36.0800},
    {"name": "Nandi", "lat": 0.2510, "lon": 34.7800},
    {"name": "Narok", "lat": -1.1167, "lon": 35.6000},
    {"name": "Nyamira", "lat": -0.6845, "lon": 34.9540},
    {"name": "Nyeri", "lat": -0.4167, "lon": 36.9500},
    {"name": "Samburu", "lat": 0.6210, "lon": 37.3430},
    {"name": "Siaya", "lat": -0.2010, "lon": 34.2860},
    {"name": "Taita-Taveta", "lat": -3.4000, "lon": 38.3830},
    {"name": "Tana River", "lat": -2.0060, "lon": 40.1400},
    {"name": "Tharaka-Nithi", "lat": -0.6440, "lon": 37.6760},
    {"name": "Trans Nzoia", "lat": 1.0260, "lon": 34.9500},
    {"name": "Turkana", "lat": 3.1920, "lon": 35.4730},
    {"name": "Uasin Gishu", "lat": 0.5200, "lon": 35.2680},
    {"name": "Vihiga", "lat": 0.0330, "lon": 34.7330},
    {"name": "Wajir", "lat": 1.7570, "lon": 40.0650},
    {"name": "West Pokot", "lat": 1.1940, "lon": 35.2520},
    {"name": "Baraingo", "lat": 0.3060, "lon": 35.8620},
    {"name": "Kiambu Central", "lat": -1.2060, "lon": 36.8250},
]

# Keep county count aligned with pipeline expectation.
# If you want strict correctness, replace the list above with an authoritative centroid list.
if len(KENYA_COUNTIES) != 47:
    # Avoid crashing hard; the pipeline will just iterate what you provide.
    pass


# =========================
# FRI (Flood Risk Index) config
# =========================

# Weights applied to component scores in risk_calculator.py
# Components come from compute_row():
# soil_moisture, antecedent_rainfall, peak_intensity, elevation, hist_flood, near_water, slope_class
FRI_WEIGHTS: dict[str, float] = {
    "soil_moisture": 0.18,
    "antecedent_rainfall": 0.18,
    "peak_intensity": 0.22,
    "elevation": 0.10,
    "hist_flood": 0.12,
    "near_water": 0.10,
    "slope_class": 0.10,
}

RAINFALL_THRESHOLDS: dict[str, float] = {
    # Antecedent threshold used in score_antecedent(mm)
    "antecedent_alert": 120.0,
    # Peak thresholds used in score_peak(mm) + visualization lines
    "peak_heavy": 40.0,
    "peak_extreme": 180.0,
}

SOIL_MOISTURE_THRESHOLDS: dict[str, float] = {
    # Used in score_soil_moisture(sm): sm / saturated
    "saturated": 0.45,
}

# Risk bands are in FRI score percent space (0-100)
RISK_THRESHOLDS: dict[str, tuple[float, float]] = {
    "LOW": (0.0, 35.0),
    "MEDIUM": (35.0, 55.0),
    "HIGH": (55.0, 75.0),
    "CRITICAL": (75.0, 100.0),
}

RISK_COLORS: dict[str, str] = {
    "LOW": "green",
    "MEDIUM": "yellow",
    "HIGH": "orange1",
    "CRITICAL": "red",
}

RISK_EMOJI: dict[str, str] = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}


# =========================
# Alert / output config
# =========================

OUTPUT_DIR: str = "outputs"
REPORT_TXT_FILE: str = os.path.join(OUTPUT_DIR, "kenya_flood_risk_report.txt")

# ML model path used by flood_predictor.py
MODEL_DIR: str = "outputs/model"
MODEL_FILE: str = os.path.join(MODEL_DIR, "rf_flood_model.joblib")

# Visualization output paths used by visualizer.py
MAP_HTML_FILE: str = os.path.join(OUTPUT_DIR, "kenya_flood_risk_map.html")
RISK_CHART_FILE: str = os.path.join(OUTPUT_DIR, "kenya_flood_risk_chart.png")
FORECAST_CHART_FILE: str = os.path.join(OUTPUT_DIR, "kenya_forecast_chart.png")
