import asyncio
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    AlertResponse,
    CountyRisk,
    DayForecast,
    ForecastResponse,
    HealthResponse,
    RefreshResponse,
    SubCounty,
    SummaryResponse,
)
from config import (
    FORECAST_CHART_FILE,
    KENYA_COUNTIES,
    MAP_HTML_FILE,
    OUTPUT_DIR,
    RISK_CHART_FILE,
)
from data.kenya_flood_drainage_dataset import get_worst_subcounties
from modules.data_loader import load_all_counties
from modules.flood_predictor import predict_flood_probability
from modules.risk_calculator import calculate_all_risks
from modules.visualizer import create_forecast_chart, create_risk_chart, create_risk_map


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rainfall_df = load_all_counties(use_cache=True)
    risk_df = calculate_all_risks(rainfall_df)
    predictions = predict_flood_probability(risk_df)
    risk_df = risk_df.merge(predictions, on="county", how="left")
    create_risk_map(risk_df)
    create_risk_chart(risk_df)
    create_forecast_chart(risk_df)
    return risk_df


def build_county_risk(row, include_subcounties=False):
    daily = row.get("daily_forecast", [])
    if not isinstance(daily, list):
        try:
            daily = list(daily)
        except Exception:
            daily = []

    ml_prob = row.get("ml_flood_prob")
    ml_prob = float(ml_prob) if pd.notna(ml_prob) else None

    sub = []
    if include_subcounties:
        raw = get_worst_subcounties(row["county"], top_n=3)
        sub = [
            SubCounty(
                sub_county=s["Sub_County"],
                drainage_quality=s["Drainage_Quality"],
                flood_risk_level=s["Flood_Risk_Level"],
                notes=s["Notes"],
            )
            for s in raw
        ]

    return CountyRisk(
        county=row["county"],
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        fri_score=float(row["fri_score"]),
        risk_level=row["risk_level"],
        risk_color=row["risk_color"],
        risk_emoji=row["risk_emoji"],
        forecast_7day_mm=float(row["forecast_7day_mm"]),
        peak_daily_mm=float(row["peak_daily_mm"]),
        soil_moisture=float(row["soil_moisture"]),
        antecedent_14day_mm=float(row["antecedent_14day_mm"]),
        ml_flood_prob=ml_prob,
        ml_predicted_level=row.get("ml_predicted_level"),
        daily_forecast=daily,
        worst_subcounties=sub,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    risk_df = await asyncio.to_thread(run_pipeline)
    app.state.risk_df = risk_df
    app.state.last_updated = datetime.now().isoformat()
    yield


app = FastAPI(
    title="Kenya Flood Risk API",
    description="Real-time flood prediction for all 47 Kenya counties",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    df = app.state.risk_df
    return HealthResponse(
        status="ok",
        counties_loaded=len(df),
        last_updated=app.state.last_updated,
        critical_counties=int((df["risk_level"] == "CRITICAL").sum()),
        high_counties=int((df["risk_level"] == "HIGH").sum()),
    )


@app.get("/api/summary", response_model=SummaryResponse)
async def summary():
    df = app.state.risk_df
    top = df.iloc[0]
    level_counts = df["risk_level"].value_counts()
    return SummaryResponse(
        last_updated=app.state.last_updated,
        total_counties=len(df),
        critical=int(level_counts.get("CRITICAL", 0)),
        high=int(level_counts.get("HIGH", 0)),
        medium=int(level_counts.get("MEDIUM", 0)),
        low=int(level_counts.get("LOW", 0)),
        top_risk_county=top["county"],
        top_risk_score=float(top["fri_score"]),
        top_risk_level=top["risk_level"],
    )


@app.get("/api/counties")
async def counties():
    return [
        {"name": c["name"], "lat": c["lat"], "lon": c["lon"]}
        for c in KENYA_COUNTIES
    ]


@app.get("/api/risk", response_model=List[CountyRisk])
async def all_risk():
    df = app.state.risk_df
    return [build_county_risk(row) for _, row in df.iterrows()]


@app.get("/api/risk/{county_name}", response_model=CountyRisk)
async def county_risk(county_name: str):
    df = app.state.risk_df
    match = df[df["county"].str.lower() == county_name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"County '{county_name}' not found")
    return build_county_risk(match.iloc[0], include_subcounties=True)


@app.get("/api/alerts", response_model=AlertResponse)
async def alerts():
    df = app.state.risk_df
    alerted = df[df["risk_level"].isin(["CRITICAL", "HIGH"])]
    return AlertResponse(
        total_alerts=len(alerted),
        critical_count=int((alerted["risk_level"] == "CRITICAL").sum()),
        high_count=int((alerted["risk_level"] == "HIGH").sum()),
        counties=[
            build_county_risk(row, include_subcounties=True)
            for _, row in alerted.iterrows()
        ],
    )


@app.get("/api/forecast/{county_name}", response_model=ForecastResponse)
async def forecast(county_name: str):
    df = app.state.risk_df
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
        county=row["county"],
        days=days,
        total_7day_mm=float(row["forecast_7day_mm"]),
        peak_daily_mm=float(row["peak_daily_mm"]),
    )


@app.get("/api/map")
async def map_file():
    if not os.path.exists(MAP_HTML_FILE):
        raise HTTPException(status_code=404, detail="Map not yet generated")
    return FileResponse(MAP_HTML_FILE, media_type="text/html")


@app.get("/api/chart/risk")
async def chart_risk():
    if not os.path.exists(RISK_CHART_FILE):
        raise HTTPException(status_code=404, detail="Chart not yet generated")
    return FileResponse(RISK_CHART_FILE, media_type="image/png")


@app.get("/api/chart/forecast")
async def chart_forecast():
    if not os.path.exists(FORECAST_CHART_FILE):
        raise HTTPException(status_code=404, detail="Chart not yet generated")
    return FileResponse(FORECAST_CHART_FILE, media_type="image/png")


@app.post("/api/refresh", response_model=RefreshResponse)
async def refresh():
    risk_df = await asyncio.to_thread(run_pipeline)
    app.state.risk_df = risk_df
    app.state.last_updated = datetime.now().isoformat()
    return RefreshResponse(
        status="refreshed",
        last_updated=app.state.last_updated,
        counties=len(risk_df),
    )
