from typing import List, Optional

from pydantic import BaseModel


class SubCounty(BaseModel):
    sub_county: str
    drainage_quality: str
    flood_risk_level: str
    notes: str


class DayForecast(BaseModel):
    date: str
    day: str
    rainfall_mm: float


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


class SummaryResponse(BaseModel):
    last_updated: str
    total_counties: int
    critical: int
    high: int
    medium: int
    low: int
    top_risk_county: str
    top_risk_score: float
    top_risk_level: str


class AlertResponse(BaseModel):
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


class RefreshResponse(BaseModel):
    status: str
    last_updated: str
    counties: int
