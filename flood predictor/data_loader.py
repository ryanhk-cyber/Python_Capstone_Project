import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import (
    ANTECEDENT_DAYS,
    DATA_CACHE_FILE,
    FORECAST_DAYS,
    KENYA_COUNTIES,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_FORECAST_URL,
    TIMEZONE,
)


def fetch_forecast(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "hourly": "soil_moisture_0_to_1cm",
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
    }
    # Keep this small; we also cap total fetch runtime in load_all_counties().
    r = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=(0.5, 0.5))
    r.raise_for_status()
    return r.json()


def fetch_antecedent(lat, lon):
    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=ANTECEDENT_DAYS - 1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "precipitation_sum",
        "timezone": TIMEZONE,
    }
    # Keep this small; we also cap total fetch runtime in load_all_counties().
    r = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=(0.5, 0.5))
    r.raise_for_status()
    return r.json()


def parse_forecast(data):
    daily = data.get("daily", {})
    precip = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]
    hourly = data.get("hourly", {})
    soil = [s if s is not None else 0.0 for s in hourly.get("soil_moisture_0_to_1cm", [])]
    return {
        "forecast_7day_mm": round(sum(precip), 2),
        "peak_daily_mm": round(max(precip), 2) if precip else 0.0,
        "soil_moisture": round(float(np.mean(soil[:24])), 4) if soil else 0.20,
        "daily_forecast": precip,
    }


def parse_antecedent(data):
    daily = data.get("daily", {})
    precip = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]
    return round(sum(precip), 2)


def fetch_county(county):
    lat, lon = county["lat"], county["lon"]
    parsed = parse_forecast(fetch_forecast(lat, lon))
    parsed["antecedent_14day_mm"] = parse_antecedent(fetch_antecedent(lat, lon))
    parsed["county"] = county["name"]
    parsed["lat"] = lat
    parsed["lon"] = lon
    return parsed


def fallback_record(county):
    return {
        "county": county["name"],
        "lat": county["lat"],
        "lon": county["lon"],
        "forecast_7day_mm": 0.0,
        "peak_daily_mm": 0.0,
        "soil_moisture": 0.20,
        "antecedent_14day_mm": 0.0,
        "daily_forecast": [0.0] * FORECAST_DAYS,
    }


def load_all_counties(use_cache=True):
    os.makedirs(os.path.dirname(DATA_CACHE_FILE), exist_ok=True)

    cache_date = str(datetime.today().date())
    if use_cache and os.path.exists(DATA_CACHE_FILE):
        with open(DATA_CACHE_FILE) as f:
            cache = json.load(f)
        if cache.get("date") == cache_date:
            return pd.DataFrame(cache["data"])

    # Hard cap so the predictor completes even if Open-Meteo is slow/unreachable.
    max_total_seconds = 10.0
    started_at = time.time()

    records = []
    for i, county in enumerate(KENYA_COUNTIES):
        # If we’re running out of time, stop fetching and use fallbacks for the rest.
        if time.time() - started_at > max_total_seconds:
            records.extend([fallback_record(c) for c in KENYA_COUNTIES[i:]])
            break

        try:
            records.append(fetch_county(county))
        except Exception:
            records.append(fallback_record(county))

        # Keep as-is, but no meaningful delay required.
        if i < len(KENYA_COUNTIES) - 1:
            time.sleep(0.0)

    # Save whatever we have (some fallbacks are okay).
    with open(DATA_CACHE_FILE, "w") as f:
        json.dump({"date": cache_date, "data": records}, f)

    return pd.DataFrame(records)
