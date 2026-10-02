"""Runtime settings for the MoES weather-anomaly service."""
import os


class Settings:
    PROJECT_NAME = "MoES Extreme Weather Anomaly Tracker"
    VERSION = "1.0.0"
    API_PREFIX = "/api/v1"
    EVENT_DATABASE_URL = os.getenv("EVENT_DATABASE_URL", "sqlite:///./weather_events.db")
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "change-me-in-production")
    WEATHER_API_KEYS = os.getenv("WEATHER_API_KEYS", "")
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    GNN_MODEL_PATH = os.getenv("GNN_MODEL_PATH", "")
    DIFFUSION_MODEL_PATH = os.getenv("DIFFUSION_MODEL_PATH", "")
    GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "1"))
    INFERENCE_EXECUTION_MODE = os.getenv("INFERENCE_EXECUTION_MODE", "local")
    REDIS_URL = os.getenv("REDIS_URL", "")
    ARTIFACT_STORAGE_DIR = os.getenv("ARTIFACT_STORAGE_DIR", "./weather_artifacts")


settings = Settings()
