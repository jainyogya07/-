"""Single-job execution shared by local asyncio and Celery GPU workers."""
from typing import Any
from backend.app.config import settings
from backend.app.infrastructure.logging import log_audit_event
from backend.app.services.event_service import EventStore
from backend.app.services.weather_models import weather_models


def execute_inference_job(job_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run and persist one job. Safe to call in a separate Celery process."""
    store = EventStore(settings.EVENT_DATABASE_URL)
    store.initialize()
    store.update_job(job_id, "running")
    weather_models.load()
    try:
        tracked = weather_models.track(payload)
        if kind == "downscale":
            tracked["downscale"] = weather_models.downscale(payload, tracked)
        event = store.save_event(tracked)
        store.update_job(job_id, "completed", event["event_id"])
        log_audit_event("weather_inference", "GNN+Diffusion", "COMPLETED", task_id=job_id, confidence=event["probability"], result={"event_id": event["event_id"], "hazard": event["hazard"]})
        return event
    except Exception as exc:
        store.update_job(job_id, "failed", error=str(exc))
        raise
