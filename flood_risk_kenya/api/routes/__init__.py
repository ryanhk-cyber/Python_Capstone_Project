from .risk import router as risk_router
from .forecast import router as forecast_router
from .alerts import router as alerts_router

__all__ = ["risk_router", "forecast_router", "alerts_router"]
