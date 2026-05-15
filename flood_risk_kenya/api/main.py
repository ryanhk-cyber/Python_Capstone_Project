import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone as dt_timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Ensure we can import the existing core modules (which live in a folder with a space) ---
CORE_DIR = os.path.join("Python_Capstone_Project", "flood predictor")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from config import (  # type: ignore
    FORECAST_CHART_FILE,
    MAP_HTML_FILE,
    OUTPUT_DIR,
    RISK_CHART_FILE,
)
from data_loader import load_all_counties  # type: ignore
from risk_calculator import calculate_all_risks  # type: ignore
from flood_predictor import predict_flood_probability  # type: ignore
from visualizer import (  # type: ignore
    create_forecast_chart,
    create_risk_chart,
    create_risk_map,
)

from .routes.risk import router as risk_router
from .routes.forecast import router as forecast_router
from .routes.alerts import router as alerts_router

from flood_risk_kenya.schemas import CountyNameResponse, HealthResponse

# For /api/map and /api/counties
from config import KENYA_COUNTIES, MAP_HTML_FILE  # type: ignore

REFRESH_EVERY_SECONDS_DEFAULT = 60 * 60  # 1 hour


def _ensure_outputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _run_pipeline():
    _ensure_outputs()
    rainfall_df = load_all_counties(use_cache=True)  # daily cache in data/rainfall_cache.json
    risk_df = calculate_all_risks(rainfall_df)
    predictions = predict_flood_probability(risk_df)
    risk_df = risk_df.merge(predictions, on="county", how="left")

    create_risk_map(risk_df)
    create_risk_chart(risk_df)
    create_forecast_chart(risk_df)

    return risk_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    # state is used by routes
    app.state.risk_df = None
    app.state.last_updated = None
    app.state.refresh_task: Optional[asyncio.Task] = None

    # initial build
    risk_df = await asyncio.to_thread(_run_pipeline)
    app.state.risk_df = risk_df
    app.state.last_updated = datetime.now(dt_timezone.utc).isoformat()

    # start background refresh loop
    refresh_every_seconds = REFRESH_EVERY_SECONDS_DEFAULT
    app.state.refresh_task = asyncio.create_task(_refresh_loop(app, refresh_every_seconds))

    yield

    # graceful shutdown: cancel background job
    if app.state.refresh_task:
        app.state.refresh_task.cancel()
        try:
            await app.state.refresh_task
        except asyncio.CancelledError:
            pass


async def _refresh_loop(app: FastAPI, every_seconds: int):
    # Refresh indefinitely, but underlying Open-Meteo work is prevented by the daily cache.
    while True:
        try:
            risk_df = await asyncio.to_thread(_run_pipeline)
            app.state.risk_df = risk_df
            app.state.last_updated = datetime.now(dt_timezone.utc).isoformat()
        except Exception:
            # keep service alive; routes will serve last-known data
            pass

        await asyncio.sleep(every_seconds)


def create_app() -> FastAPI:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    app = FastAPI(
        title="Kenya Flood Risk API",
        description="Real-time flood prediction for all 47 Kenya counties",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        df = getattr(app.state, "risk_df", None)
        if df is None:
            raise HTTPException(status_code=503, detail="Service warming up")
        return HealthResponse(
            status="ok",
            counties_loaded=len(df),
            last_updated=app.state.last_updated,
            critical_counties=int((df["risk_level"] == "CRITICAL").sum()),
            high_counties=int((df["risk_level"] == "HIGH").sum()),
        )

    @app.get("/api/counties", response_model=CountyNameResponse)
    async def counties() -> CountyNameResponse:
        return CountyNameResponse(counties=[c["name"] for c in KENYA_COUNTIES])

    @app.get("/api/map")
    async def map_file() -> FileResponse:
        if not os.path.exists(MAP_HTML_FILE):
            raise HTTPException(status_code=404, detail="Map not yet generated")
        return FileResponse(MAP_HTML_FILE, media_type="text/html")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # static serving for map + charts
    _ensure_outputs()
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

    # routers
    app.include_router(risk_router)
    app.include_router(forecast_router)
    app.include_router(alerts_router)

    return app


app = create_app()

# Expose these paths for routes/tests (not required but helpful)
__all__ = [
    "app",
    "MAP_HTML_FILE",
    "RISK_CHART_FILE",
    "FORECAST_CHART_FILE",
]
