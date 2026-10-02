"""Asynchronous, GPU-concurrency-limited weather inference queue."""
import asyncio
from typing import Any
from backend.app.config import settings
from backend.app.infrastructure.logging import logger, log_audit_event
from backend.app.services.event_service import EventStore
from backend.app.services.live_updates import live_updates
from backend.app.services.weather_models import weather_models
from backend.app.services.inference_runner import execute_inference_job


class InferenceQueue:
    def __init__(self) -> None:
        self.store = EventStore(settings.EVENT_DATABASE_URL)
        self.queue: asyncio.Queue[tuple[str, str, dict[str, Any]]] = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.gpu_slots = asyncio.Semaphore(max(1, settings.GPU_CONCURRENCY))

    async def start(self) -> None:
        self.store.initialize()
        weather_models.load()
        if settings.INFERENCE_EXECUTION_MODE == "local" and not self.workers:
            self.workers = [asyncio.create_task(self._worker(index)) for index in range(max(1, settings.GPU_CONCURRENCY))]
        logger.info("Weather inference queue ready: device=%s workers=%s", weather_models.device, len(self.workers))

    async def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def enqueue(self, kind: str, payload: dict[str, Any]) -> str:
        job_id = self.store.create_job(kind)
        if settings.INFERENCE_EXECUTION_MODE == "celery":
            from backend.app.workers.celery_app import run_weather_inference
            run_weather_inference.delay(job_id, kind, payload)
        elif settings.INFERENCE_EXECUTION_MODE == "local":
            await self.queue.put((job_id, kind, payload))
        else:
            self.store.update_job(job_id, "failed", error="Unsupported INFERENCE_EXECUTION_MODE")
            raise RuntimeError("Unsupported INFERENCE_EXECUTION_MODE")
        await live_updates.publish({"type": "job", "job_id": job_id, "status": "queued", "kind": kind})
        return job_id

    async def _worker(self, worker_id: int) -> None:
        while True:
            job_id, kind, payload = await self.queue.get()
            self.store.update_job(job_id, "running")
            await live_updates.publish({"type": "job", "job_id": job_id, "status": "running", "kind": kind})
            try:
                async with self.gpu_slots:
                    event = await asyncio.to_thread(execute_inference_job, job_id, kind, payload)
                await live_updates.publish({"type": "event", "job_id": job_id, "status": "completed", "event": event})
                if event["severity"] in ("high", "severe"):
                    west, south, east, north = event["bbox"]
                    await live_updates.publish({"type": "alert", "event_id": event["event_id"], "hazard": event["hazard"], "severity": event["severity"], "probability": event["probability"], "impact_radius_km": 5, "centroid": [(west + east) / 2, (south + north) / 2]})
            except Exception as exc:  # errors are persisted and never leaked as a stack trace
                logger.exception("Weather inference failed for job %s", job_id)
                self.store.update_job(job_id, "failed", error=str(exc))
                await live_updates.publish({"type": "job", "job_id": job_id, "status": "failed", "error": "inference failed"})
            finally:
                self.queue.task_done()


inference_queue = InferenceQueue()
