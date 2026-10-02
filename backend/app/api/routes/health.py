from fastapi import APIRouter, Response, status
from backend.app.config import settings
from backend.app.services.weather_models import weather_models

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.VERSION, "device": weather_models.device}


@router.get("/ready")
def ready(response: Response) -> dict:
    value = settings.WEATHER_API_KEY != "change-me-in-production" or bool(settings.WEATHER_API_KEYS)
    if not value:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": value, "execution_mode": settings.INFERENCE_EXECUTION_MODE, "redis_configured": bool(settings.REDIS_URL)}
