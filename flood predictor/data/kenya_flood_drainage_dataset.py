from __future__ import annotations

from typing import Optional

# -----------------------------------------------------------------------------
# Minimal stub so the application can run end-to-end.
# -----------------------------------------------------------------------------

# Drainage quality string -> numeric used by flood_predictor features.
DRAINAGE_QUALITY_MAP: dict[str, float] = {
    "Poor": 1.0,
    "Moderate": 2.0,
    "Good": 4.0,
    "Excellent": 5.0,
}

# Flood risk label string -> integer class for ML training.
FLOOD_RISK_LABEL_MAP: dict[str, int] = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Severe": 3,
}

# Example training rows used by flood_predictor.get_training_samples().
# Shape must match what flood_predictor.extract_features() and training loop expect:
# - "County"
# - "Drainage_Quality"
# - "Elevation" optional (not used directly)
# - "near_water", "slope_class", "hist_flood" optional (not used directly)
# - "Flood_Risk_Level"
# Our flood_predictor uses county_lookup + sample["Drainage_Quality"] + label.
#
# We keep this small to avoid heavy dataset maintenance.
kenya_flood_drainage: list[dict[str, str]] = [
    {"County": "Nairobi", "Drainage_Quality": "Good", "Flood_Risk_Level": "Low"},
    {"County": "Nairobi", "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe"},
    {"County": "Nakuru", "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate"},
    {"County": "Nakuru", "Drainage_Quality": "Poor", "Flood_Risk_Level": "High"},
    {"County": "Mombasa", "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High"},
    {"County": "Kiambu", "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate"},
]


# Example drainage score (1..4) used by risk_calculator fallback when dataset is missing.
_DRAINAGE_SCORE: dict[str, float] = {
    "Nairobi": 2.5,
    "Nakuru": 3.0,
    "Mombasa": 2.0,
    "Kisumu": 2.0,
    "Kiambu": 3.0,
}


def get_county_drainage_score(county_name: str) -> Optional[float]:
    if not county_name:
        return None
    return _DRAINAGE_SCORE.get(county_name, 2.5)


def get_worst_subcounties(county_name: str, top_n: int = 3) -> list[dict[str, str]]:
    if not county_name or top_n <= 0:
        return []

    sub = [
        {"Sub_County": f"{county_name} Central", "Notes": "High risk sub-county (stub dataset)."},
        {"Sub_County": f"{county_name} East", "Notes": "High risk sub-county (stub dataset)."},
        {"Sub_County": f"{county_name} West", "Notes": "High risk sub-county (stub dataset)."},
    ]
    return sub[:top_n]


def get_training_samples() -> list[dict[str, str]]:
    # flood_predictor.build_training_data expects `samples` where each `s` has:
    # - s["County"]
    # - s["Drainage_Quality"]
    # - s["Flood_Risk_Level"]
    return kenya_flood_drainage
