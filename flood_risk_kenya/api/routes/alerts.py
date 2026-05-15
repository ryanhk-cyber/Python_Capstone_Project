from __future__ import annotations

from typing import List

import pandas as pd  # type: ignore
from fastapi import APIRouter, HTTPException, Request

from flood_risk_kenya.schemas import AlertsResponse, CountyRisk
from .risk import _build_county_risk


router = APIRouter(prefix="/api", tags=["alerts"])


def _get_df(app) -> pd.DataFrame:
    df = getattr(app.state, "risk_df", None)
    if df is None:
        raise HTTPException(status_code=503, detail="Risk data not ready yet")
    return df


@router.get("/alerts", response_model=AlertsResponse)
async def alerts(request: Request) -> AlertsResponse:
    df = _get_df(request.app)
    alerted = df[df["risk_level"].isin(["CRITICAL", "HIGH"])]

    counties: List[CountyRisk] = [
        _build_county_risk(row, include_subcounties=True) for _, row in alerted.iterrows()
    ]

    return AlertsResponse(
        total_alerts=len(alerted),
        critical_count=int((alerted["risk_level"] == "CRITICAL").sum()),
        high_count=int((alerted["risk_level"] == "HIGH").sum()),
        counties=counties,
    )
