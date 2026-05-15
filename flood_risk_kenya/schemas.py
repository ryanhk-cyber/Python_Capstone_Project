from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class SubCounty(BaseModel):
    sub_county: str
    drainage_quality: str
    flood_risk_level: str
    notes: str


class CountyRisk(BaseModel):
    county: str
    lat: float
    lon: float

    fri_score: float
    risk_level: str
    risk_color: str
    risk_emoji: str

    forecast_7day_mm: float
    peak_daily_mm: float

    soil_moisture: float
    antecedent_14day_mm: float

    ml_flood_prob: Optional[float] = None
    ml_predicted_level: Optional[str] = None

    daily_forecast: List[float] = []
    worst_subcounties: List[SubCounty] = []


class DayForecast(BaseModel):
    date: str
    day: str
    rainfall_mm: float


class RiskResponseList(BaseModel):
    # Convenience wrapper if frontend ever wants it; not required by current endpoints.
    items: List[CountyRisk]


class AlertsResponse(BaseModel):
    total_alerts: int
    critical_count: int
    high_count: int
    counties: List[CountyRisk]


class ForecastResponse(BaseModel):
    county: str
    days: List[DayForecast]
    total_7day_mm: float
    peak_daily_mm: float


class HealthResponse(BaseModel):
    status: str
    counties_loaded: int
    last_updated: str
    critical_counties: int
    high_counties: int


class CountyNameResponse(BaseModel):
    counties: List[str]


class MapResponse(BaseModel):
    # We return HTML content as a string so frontend can display it if it prefers.
    # (We also serve the file statically via /outputs.)
    html: str
