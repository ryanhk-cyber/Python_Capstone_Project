from __future__ import annotations

from datetime import date, timedelta

import pandas as pd  # type: ignore
from fastapi import APIRouter, HTTPException, Request

from flood_risk_kenya.schemas import CountyRisk, DayForecast, ForecastResponse

# Ensure core imports work (mirrors api/main.py)
import os
import sys

CORE_DIR = os.path.join("Python_Capstone_Project", "flood predictor")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

router = APIRouter(prefix="/api", tags=["forecast"])


def _get_df(app) -> pd.DataFrame:
    df = getattr(app.state, "risk_df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Risk data not ready yet")
    return df


@router.get("/forecast/{county_name}", response_model=ForecastResponse)
async def forecast(county_name: str, request: Request) -> ForecastResponse:
    df = _get_df(request.app)
    match = df[df["county"].str.lower() == county_name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"County '{county_name}' not found")

    row = match.iloc[0]
    daily = row.get("daily_forecast", [])
    if not isinstance(daily, list):
        daily = []

    today = date.today()
    days = [
        DayForecast(
            date=str(today + timedelta(days=i)),
            day=(today + timedelta(days=i)).strftime("%A"),
            rainfall_mm=float(daily[i]) if i < len(daily) else 0.0,
        )
        for i in range(7)
    ]

    return ForecastResponse(
        county=str(row["county"]),
        days=days,
        total_7day_mm=float(row["forecast_7day_mm"]),
        peak_daily_mm=float(row["peak_daily_mm"]),
    )
