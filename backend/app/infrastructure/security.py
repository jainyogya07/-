"""API-key verification and request throttling for service-to-service callers."""
import hashlib
import secrets
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request, status
from backend.app.config import settings


class ApiKeyGuard:
    def __init__(self) -> None:
        self.calls: dict[str, deque[float]] = defaultdict(deque)

    def authorize(self, request: Request, api_key: str) -> str:
        keys = [key.strip() for key in settings.WEATHER_API_KEYS.split(",") if key.strip()] or [settings.WEATHER_API_KEY]
        accepted = any(self._matches(api_key, candidate) for candidate in keys)
        if not accepted:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:12]
        now, requests = time.monotonic(), self.calls[fingerprint]
        while requests and now - requests[0] >= 60:
            requests.popleft()
        if len(requests) >= settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded", headers={"Retry-After": str(max(1, int(60 - (now - requests[0]))))})
        requests.append(now)
        return fingerprint

    @staticmethod
    def _matches(provided: str, configured: str) -> bool:
        # Production secrets should be set as sha256:<digest>, never plaintext.
        if configured.startswith("sha256:"):
            return secrets.compare_digest(hashlib.sha256(provided.encode()).hexdigest(), configured.removeprefix("sha256:"))
        return secrets.compare_digest(provided, configured)


api_key_guard = ApiKeyGuard()
