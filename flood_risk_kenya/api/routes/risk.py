from __future__ import annotations

from typing import List

import pandas as pd  # type: ignore
from fastapi import APIRouter, HTTPException

from flood_risk_kenya.schemas import CountyRisk, SubCounty

# Import core helpers (so we can fetch sub-county details)
import sys
import os

# Ensure core imports work (mirrors api/main.py)
CORE_DIR = os.path.join("Python_Capstone_Project", "flood predictor")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from config import KENYA_COUNTIES  # type: ignore
from data.kenya_flood_drainage_dataset import get_worst_subcounties  # type: ignore

router = APIRouter(prefix="/api", tags=["risk"])


def _build_county_risk(row: pd.Series, include_subcounties: bool = False) -> CountyRisk:
    daily = row.get("daily_forecast", [])
    if not isinstance(daily, list):
        try:
            daily = list(daily)
        except Exception:
            daily = []

    ml_prob = row.get("ml_flood_prob")
    ml_prob_float = float(ml_prob) if pd.notna(ml_prob) else None

    worst_subs: List[SubCounty] = []
    if include_subcounties:
        raw = get_worst_subcounties(str(row["county"]), top_n=3)
        # Core stub dataset may not include drainage quality / risk level fields.
        worst_subs = [
            SubCounty(
                sub_county=str(s.get("Sub_County", "")),
                drainage_quality=str(s.get("Drainage_Quality", "Unknown")),
                flood_risk_level=str(s.get("Flood_Risk_Level", "MEDIUM")),
                notes=str(s.get("Notes", "—")),
            )
            for s in raw
        ]

    return CountyRisk(
        county=str(row["county"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        fri_score=float(row["fri_score"]),
        risk_level=str(row["risk_level"]),
        risk_color=str(row["risk_color"]),
        risk_emoji=str(row["risk_emoji"]),
        forecast_7day_mm=float(row["forecast_7day_mm"]),
        peak_daily_mm=float(row["peak_daily_mm"]),
        soil_moisture=float(row["soil_moisture"]),
        antecedent_14day_mm=float(row["antecedent_14day_mm"]),
        ml_flood_prob=ml_prob_float,
        ml_predicted_level=row.get("ml_predicted_level"),
        daily_forecast=daily,
        worst_subcounties=worst_subs,
    )


def _get_df(app) -> pd.DataFrame:
    df = getattr(app.state, "risk_df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Risk data not ready yet")
    return df


from fastapi import Request


@router.get("/risk", response_model=List[CountyRisk])
async def all_risk(request: Request):
    df = _get_df(request.app)
    return [_build_county_risk(row, include_subcounties=False) for _, row in df.iterrows()]


@router.get("/risk/{county_name}", response_model=CountyRisk)
async def county_risk(county_name: str, request: Request):
    df = _get_df(request.app)
    match = df[df["county"].str.lower() == county_name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"County '{county_name}' not found")
    return _build_county_risk(match.iloc[0], include_subcounties=True)
