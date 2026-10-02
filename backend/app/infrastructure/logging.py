"""Structured application and inference audit logging."""
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("moes_weather_tracker")


def log_audit_event(action: str, model_id: str, status: str, **data: Any) -> None:
    logger.info("audit action=%s model=%s status=%s data=%s", action, model_id, status, data)
