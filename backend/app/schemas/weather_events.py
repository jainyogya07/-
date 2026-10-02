"""Request and response contracts for spatio-temporal weather events."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Hazard = Literal["extreme_precipitation", "cyclone", "heatwave", "cold_wave", "high_wind", "hail"]
Severity = Literal["low", "moderate", "high", "severe"]


class InferenceInput(BaseModel):
    hazard_hint: Optional[Hazard] = None
    # [west, south, east, north] in WGS84; persisted as a PostGIS-ready bbox.
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    forecast_hours: List[int] = Field(default_factory=lambda: [24, 48, 72])
    variables: Dict[str, Any] = Field(default_factory=dict)
    source: str = "NEPS-G"
    issued_at: Optional[datetime] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    event_id: Optional[str] = None


class EventResponse(BaseModel):
    event_id: str
    hazard: str
    probability: float
    severity: str
    bbox: List[float]
    forecast_hours: List[int]
    trajectory: List[Dict[str, Any]]
    status: str
    created_at: str
    downscale: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    event_id: Optional[str] = None
    error: Optional[str] = None
