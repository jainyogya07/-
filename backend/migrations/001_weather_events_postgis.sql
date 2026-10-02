CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS weather_events (
  event_id TEXT PRIMARY KEY,
  hazard TEXT NOT NULL,
  probability DOUBLE PRECISION NOT NULL CHECK (probability >= 0 AND probability <= 1),
  severity TEXT NOT NULL CHECK (severity IN ('low', 'moderate', 'high', 'severe')),
  bbox JSONB NOT NULL,
  footprint geometry(Polygon, 4326) NOT NULL,
  forecast_hours JSONB NOT NULL,
  trajectory JSONB NOT NULL,
  downscale JSONB,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_weather_events_footprint ON weather_events USING GIST(footprint);
CREATE INDEX IF NOT EXISTS idx_weather_events_created_at ON weather_events(created_at DESC);

CREATE TABLE IF NOT EXISTS inference_jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  event_id TEXT REFERENCES weather_events(event_id),
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
