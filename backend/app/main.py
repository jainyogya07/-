"""Standalone MoES extreme-weather anomaly tracking API."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.infrastructure.logging import logger
from backend.app.services.inference_queue import inference_queue
from backend.app.services.live_updates import live_updates
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.inference import router as inference_router
from backend.app.api.routes.events import router as events_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await inference_queue.start()
    await live_updates.start()
    logger.info("Started %s", settings.PROJECT_NAME)
    yield
    await inference_queue.stop()
    await live_updates.stop()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(inference_router, prefix=settings.API_PREFIX)
app.include_router(events_router, prefix=settings.API_PREFIX)


@app.get("/")
def root() -> dict:
    return {"service": settings.PROJECT_NAME, "docs": "/docs", "health": f"{settings.API_PREFIX}/health"}
