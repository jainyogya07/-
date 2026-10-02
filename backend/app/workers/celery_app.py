"""Celery configuration for horizontally scaled inference workers.

The API uses the in-process queue during local development. In production,
route submissions to this Celery app (Redis broker/backend) so GPU workers can
run independently from web workers.
"""
import os
import json
from celery import Celery

broker = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
backend = os.getenv("CELERY_RESULT_BACKEND", broker)
celery_app = Celery("weather_inference", broker=broker, backend=backend)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"], task_acks_late=True, worker_prefetch_multiplier=1)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def run_weather_inference(self, job_id: str, kind: str, payload: dict) -> dict:
    """GPU-worker entry point. Run with `celery -A ...celery_app worker`."""
    from backend.app.services.inference_runner import execute_inference_job
    _publish({"type": "job", "job_id": job_id, "status": "running", "kind": kind})
    try:
        event = execute_inference_job(job_id, kind, payload)
        _publish({"type": "event", "job_id": job_id, "status": "completed", "event": event})
        if event["severity"] in ("high", "severe"):
            west, south, east, north = event["bbox"]
            _publish({"type": "alert", "event_id": event["event_id"], "hazard": event["hazard"], "severity": event["severity"], "probability": event["probability"], "impact_radius_km": 5, "centroid": [(west + east) / 2, (south + north) / 2]})
        return event
    except Exception:
        _publish({"type": "job", "job_id": job_id, "status": "failed", "error": "inference failed"})
        raise


def _publish(message: dict) -> None:
    """Publish directly from the synchronous Celery worker to API listeners."""
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return
    try:
        import redis
        redis.Redis.from_url(redis_url, decode_responses=True).publish("weather-events", json.dumps(message))
    except Exception:
        # A worker must not lose a completed database job merely because a
        # dashboard notification is temporarily unavailable.
        pass
