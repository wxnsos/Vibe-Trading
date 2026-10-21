"""Read-only inventory API for the persistent market-data cache."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from src.api.security import require_auth
from src.market_data_store import MarketDataStore


def register_market_data_routes(app: FastAPI) -> None:
    @app.get("/api/market-data/datasets", dependencies=[Depends(require_auth)])
    async def list_market_data_datasets():
        """List datasets collected by backtests and market-data tools."""
        datasets = MarketDataStore().list_datasets()
        return {
            "datasets": datasets,
            "dataset_count": len(datasets),
            "stored_bytes": sum(int(item.get("stored_bytes", 0)) for item in datasets),
        }
