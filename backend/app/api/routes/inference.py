"""Authenticated asynchronous endpoints for GNN and diffusion inference."""
import secrets
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from backend.app.config import settings
from backend.app.schemas.weather_events import InferenceInput, JobAccepted, JobResponse
from backend.app.services.inference_queue import inference_queue
from backend.app.infrastructure.security import api_key_guard

router = APIRouter(prefix="/inference", tags=["Weather Event Inference"])


def require_api_key(request: Request, x_api_key: str = Header(..., description="Service API key")) -> str:
    return api_key_guard.authorize(request, x_api_key)


@router.post("/anomaly", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def anomaly(payload: InferenceInput) -> JobAccepted:
    """Run GNN anomaly detection and persist the resulting tracked event."""
    return JobAccepted(job_id=await inference_queue.enqueue("anomaly", payload.model_dump(mode="json")))


@router.post("/track", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def track(payload: InferenceInput) -> JobAccepted:
    """Track a GNN-detected anomaly through its forecast horizon."""
    return JobAccepted(job_id=await inference_queue.enqueue("track", payload.model_dump(mode="json")))


@router.post("/downscale", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def downscale(payload: InferenceInput) -> JobAccepted:
    """Run GNN tracking followed by diffusion downscaling."""
    return JobAccepted(job_id=await inference_queue.enqueue("downscale", payload.model_dump(mode="json")))


@router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str) -> JobResponse:
    job = inference_queue.store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")
    return JobResponse(**job)
