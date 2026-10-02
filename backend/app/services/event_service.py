"""Durable event and job storage.

SQLite is used for local development; EVENT_DATABASE_URL may point to the
PostgreSQL/PostGIS deployment. Geometry is stored as WGS84 GeoJSON-compatible
bbox values, making the schema easy to migrate to a PostGIS geometry column.
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class EventStore:
    def __init__(self, database_url: str) -> None:
        self.postgres = database_url.startswith(("postgresql://", "postgres://"))
        self.path = database_url.removeprefix("sqlite:///") if not self.postgres else database_url
        self.lock = threading.Lock()

    def initialize(self) -> None:
        if self.postgres:
            with self._connect() as db, db.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                cur.execute("""CREATE TABLE IF NOT EXISTS weather_events (
                  event_id TEXT PRIMARY KEY, hazard TEXT NOT NULL, probability DOUBLE PRECISION NOT NULL,
                  severity TEXT NOT NULL, bbox JSONB NOT NULL, forecast_hours JSONB NOT NULL,
                  trajectory JSONB NOT NULL, downscale JSONB, footprint geometry(Polygon, 4326),
                  status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )""")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_weather_events_footprint ON weather_events USING GIST(footprint)")
                cur.execute("""CREATE TABLE IF NOT EXISTS inference_jobs (
                  job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, event_id TEXT,
                  error TEXT, created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL
                )""")
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True) if Path(self.path).parent != Path(".") else None
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS weather_events (
              event_id TEXT PRIMARY KEY, hazard TEXT NOT NULL, probability REAL NOT NULL,
              severity TEXT NOT NULL, bbox TEXT NOT NULL, forecast_hours TEXT NOT NULL,
              trajectory TEXT NOT NULL, downscale TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inference_jobs (
              job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
              event_id TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_event_created_at ON weather_events(created_at);
            """)

    def _connect(self):
        if self.postgres:
            import psycopg
            return psycopg.connect(self.path)
        db = sqlite3.connect(self.path, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def create_job(self, kind: str) -> str:
        job_id, now = f"JOB-{uuid.uuid4().hex[:12].upper()}", self._now()
        with self.lock, self._connect() as db:
            if self.postgres:
                with db.cursor() as cur:
                    cur.execute("INSERT INTO inference_jobs VALUES (%s, %s, 'queued', NULL, NULL, %s, %s)", (job_id, kind, now, now))
            else:
                db.execute("INSERT INTO inference_jobs VALUES (?, ?, 'queued', NULL, NULL, ?, ?)", (job_id, kind, now, now))
        return job_id

    def update_job(self, job_id: str, status: str, event_id: Optional[str] = None, error: Optional[str] = None) -> None:
        with self.lock, self._connect() as db:
            query = "UPDATE inference_jobs SET status={}, event_id=COALESCE({}, event_id), error={}, updated_at={} WHERE job_id={}".format(*(["%s"] * 5 if self.postgres else ["?"] * 5))
            if self.postgres:
                with db.cursor() as cur: cur.execute(query, (status, event_id, error, self._now(), job_id))
            else:
                db.execute(query, (status, event_id, error, self._now(), job_id))

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._connect() as db:
            if self.postgres:
                with db.cursor() as cur:
                    cur.execute("SELECT * FROM inference_jobs WHERE job_id=%s", (job_id,)); row = cur.fetchone(); columns = [d.name for d in cur.description] if row else []
                return dict(zip(columns, row)) if row else None
            row = db.execute("SELECT * FROM inference_jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def save_event(self, payload: dict[str, Any]) -> dict:
        event_id, now = f"EVT-{uuid.uuid4().hex[:10].upper()}", self._now()
        event = {**payload, "event_id": event_id, "created_at": now, "status": "completed"}
        with self.lock, self._connect() as db:
            values = (
                event_id, event["hazard"], event["probability"], event["severity"], json.dumps(event["bbox"]),
                json.dumps(event["forecast_hours"]), json.dumps(event["trajectory"]), json.dumps(event.get("downscale")),
                event["status"], now)
            if self.postgres:
                bbox = event["bbox"]
                with db.cursor() as cur: cur.execute("""INSERT INTO weather_events
                    (event_id, hazard, probability, severity, bbox, forecast_hours, trajectory, downscale, status, created_at, footprint)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, ST_MakeEnvelope(%s, %s, %s, %s, 4326))""", values + tuple(bbox))
            else:
                db.execute("INSERT INTO weather_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        return event

    def get_event(self, event_id: str) -> Optional[dict]:
        with self._connect() as db:
            if self.postgres:
                with db.cursor() as cur:
                    cur.execute("SELECT * FROM weather_events WHERE event_id=%s", (event_id,)); row = cur.fetchone(); columns = [d.name for d in cur.description] if row else []
                event = dict(zip(columns, row)) if row else None
            else:
                row = db.execute("SELECT * FROM weather_events WHERE event_id=?", (event_id,)).fetchone(); event = dict(row) if row else None
        if not event:
            return None
        for key in ("bbox", "forecast_hours", "trajectory", "downscale"):
            event[key] = json.loads(event[key]) if isinstance(event[key], str) else event[key]
        return event

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
