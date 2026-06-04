"""
Store Intelligence — FastAPI Application Factory

Main entry point for the Intelligence API.
Registers all routers, middleware, CORS, and lifespan events.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_database, close_database, load_pos_transactions
from app.middleware import RequestLoggingMiddleware
from app.ingestion import router as ingestion_router
from app.metrics import router as metrics_router
from app.funnel import router as funnel_router
from app.heatmap import router as heatmap_router
from app.anomalies import router as anomalies_router
from app.health import router as health_router

# ─── Logging Configuration ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Application Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Store Intelligence API starting up...")
    await init_database()
    logger.info("✅ Database initialised")
    
    await load_pos_transactions()
    
    yield
    
    logger.info("🔻 Store Intelligence API shutting down...")
    await close_database()
    logger.info("✅ Database connection closed")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Store Intelligence API",
        description=(
            "Real-time retail analytics API for Apex Retail. "
            "Processes CCTV footage through YOLOv8 + ByteTrack + OSNet "
            "to track customer behaviour, measure conversion rates, "
            "and detect operational anomalies."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permissive for hackathon; restrict in production
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ──
    app.add_middleware(RequestLoggingMiddleware)

    # ── Register Routers ──
    app.include_router(ingestion_router)
    app.include_router(metrics_router)
    app.include_router(funnel_router)
    app.include_router(heatmap_router)
    app.include_router(anomalies_router)
    app.include_router(health_router)

    # ── Root Endpoint ──
    @app.get("/", tags=["root"])
    async def root():
        return {
            "service": "Store Intelligence API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "description": (
                "Turn raw CCTV footage into live store analytics. "
                "POST events to /events/ingest, query metrics at "
                "/stores/{store_id}/metrics"
            ),
        }

    return app


# ─── Application Instance ────────────────────────────────────────────────────

app = create_app()
