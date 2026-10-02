"""Small local object-store adapter for oversized inference outputs.

Use an S3-compatible adapter behind this interface in cloud deployments.
"""
import json
import uuid
from pathlib import Path
from typing import Any
from backend.app.config import settings


class ArtifactStore:
    def save_json(self, event_id: str, name: str, value: Any) -> str:
        root = Path(settings.ARTIFACT_STORAGE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{event_id}-{name}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path.resolve())


artifact_store = ArtifactStore()
