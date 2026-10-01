"""
FastAPI Main Application.
AI-Driven Spatio-Temporal Weather Anomaly Tracking & Data Preprocessing Core.
Ministry of Earth Sciences (MoES) - Problem Statement 26078.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.weather_routes import router as weather_router, get_datasets
from app.api.news_routes import router as news_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("weather_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Weather Tracking System...")
    # Pre-generate / load benchmark datasets into memory
    get_datasets()
    logger.info("Weather datasets ready. Ready to serve requests.")
    yield
    logger.info("Shutting down Weather Tracking System.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade backend for Ingestion, Ensemble Preprocessing, "
        "Extreme Forecast Index (EFI) Anomaly Detection, Spatio-Temporal Bounding Box Generation, "
        "and Live Weather Disaster News Integration."
    ),
    lifespan=lifespan,
)

# Enable CORS for local and cloud frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(weather_router)
app.include_router(news_router)


@app.get("/", tags=["System Status"])
def root():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "forecast": "/weather/forecast",
            "ensemble": "/weather/ensemble",
            "anomaly": "/weather/anomaly",
            "region": "/weather/region/{id}",
            "timeline": "/weather/timeline",
            "news": "/weather/news",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["System Status"])
def health_check():
    return {"status": "healthy"}
