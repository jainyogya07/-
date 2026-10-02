# AI-Driven Spatio-Temporal Extreme Weather Event Backend

**Ministry of Earth Sciences (MoES) — Problem Statement 26078**

Production-oriented backend for submitting extreme-weather inference jobs, tracking forecast events, generating 5 km impact summaries, persisting results, and streaming alert updates to a dashboard.

## System architecture

```text
       [Forecast anomaly payload / model-ready weather features]
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              EXTREME WEATHER INFERENCE API — FastAPI                    │
│ API-key authentication • rate limiting • async jobs • event/alert APIs  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                INFERENCE WORKER — Local Queue / Celery                 │
│ GNN tracking adapter • diffusion downscaling adapter • GPU limits       │
│ probability/severity classification • durable event storage             │
└────────────────┬───────────────────────────┬─────────────────────────────┘
                 ▼                           ▼
        SQLite (development)        PostgreSQL/PostGIS + Redis pub/sub
                                                │
                                                ▼
                              Frontend map, alerts, and event timeline
```

## Implemented capabilities

- Async anomaly, tracking, and downscaling inference jobs.
- GNN/diffusion adapter interfaces with lazy PyTorch device loading.
- Deterministic development fallback until trained checkpoints are connected.
- GPU concurrency limits and scalable Celery + Redis operation.
- SQLite local storage and PostgreSQL/PostGIS production support.
- Event IDs, probability, severity, bounding boxes, trajectory, forecast hours, and downscaling metadata.
- API-key authentication, SHA-256 key support, and per-key rate limiting.
- WebSocket and Server-Sent Events updates for jobs, events, and alerts.
- Docker build instructions and a PostGIS migration.

## Quick start

Requires Python 3.10+. Run these commands from the folder that contains `backend/`:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
python -m backend.run
```

The service starts at `http://127.0.0.1:8000`.

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`
- Readiness: `http://127.0.0.1:8000/api/v1/ready`

Local development API key:

```text
change-me-in-production
```

Send it in the `X-API-Key` header.

## REST API reference

### 1. Submit inference jobs

| Endpoint | Pipeline |
| --- | --- |
| `POST /api/v1/inference/anomaly` | Detect and classify an anomaly. |
| `POST /api/v1/inference/track` | Generate an anomaly forecast trajectory. |
| `POST /api/v1/inference/downscale` | Tracking plus 12 km → 5 km downscaling summary. |

```bash
curl -X POST http://localhost:8000/api/v1/inference/downscale \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-in-production" \
  -d '{
    "hazard_hint": "extreme_precipitation",
    "bbox": [80.8, 26.7, 81.0, 26.9],
    "forecast_hours": [24, 48, 72],
    "variables": { "efi": 3.0, "precipitation_mm": 95 },
    "source": "NEPS-G"
  }'
```

Response (`202 Accepted`):

```json
{ "job_id": "JOB-ABC123", "status": "queued", "event_id": null }
```

### 2. Check inference status

`GET /api/v1/inference/jobs/{job_id}`

```json
{
  "job_id": "JOB-ABC123",
  "kind": "downscale",
  "status": "completed",
  "event_id": "EVT-001",
  "error": null
}
```

Job statuses: `queued`, `running`, `completed`, `failed`.

### 3. Retrieve a tracked event

`GET /api/v1/events/{event_id}`

```json
{
  "event_id": "EVT-001",
  "hazard": "extreme_precipitation",
  "probability": 0.87,
  "severity": "high",
  "bbox": [80.8, 26.7, 81.0, 26.9],
  "forecast_hours": [24, 48, 72],
  "trajectory": [{ "forecast_hour": 24, "centroid": [80.9, 26.8], "probability": 0.87 }],
  "downscale": { "source_resolution_km": 12, "target_resolution_km": 5, "peak_intensity": 112.1, "units": "mm" }
}
```

### 4. Focused event APIs and alerts

| Endpoint | Output |
| --- | --- |
| `GET /api/v1/events/{event_id}/trajectory` | Forecast path and lead hours. |
| `GET /api/v1/events/{event_id}/probability` | Hazard, probability, severity. |
| `GET /api/v1/events/{event_id}/alert` | Categorised alert and 5 km impact-zone centroid. |

High and severe events automatically emit alert messages.

### 5. Live updates

WebSocket:

```text
ws://localhost:8000/api/v1/events/live?api_key=change-me-in-production
```

SSE:

```text
GET /api/v1/events/live/sse
X-API-Key: change-me-in-production
```

Messages have type `job`, `event`, or `alert`.

## Production deployment

```env
WEATHER_API_KEY=sha256:<sha256-of-a-long-random-key>
EVENT_DATABASE_URL=postgresql://weather:password@postgres:5432/weather
INFERENCE_EXECUTION_MODE=celery
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
GNN_MODEL_PATH=/models/gnn.pt
DIFFUSION_MODEL_PATH=/models/diffusion.pt
GPU_CONCURRENCY=1
```

Apply [migrations/001_weather_events_postgis.sql](migrations/001_weather_events_postgis.sql) before serving production traffic.

```bash
# From the folder that contains backend/
docker build -f backend/Dockerfile -t moes-weather-backend .
docker run -p 8000:8000 --env-file backend/.env moes-weather-backend

# Separate GPU worker
celery -A backend.app.workers.celery_app worker --loglevel=INFO --concurrency=1
```

## Directory structure

```text
backend/
├── app/
│   ├── api/routes/           # Health, inference, events
│   ├── infrastructure/       # Logging, storage, authentication
│   ├── schemas/              # API contracts
│   ├── services/             # Models, queue, persistence, live updates
│   ├── workers/              # Celery entry point
│   ├── config.py
│   └── main.py
├── migrations/               # PostGIS schema
├── Dockerfile
├── requirements.txt
├── run.py
└── README.md
```

## Model integration boundary

The current adapter provides a runnable development fallback. Replace `app/services/weather_models.py` with trained GNN and diffusion checkpoints, GRIB/NetCDF preprocessing, and scientifically validated EFI calculations when the model files and data contracts are available. The API, queue, storage, alerts, and frontend contract remain unchanged.
