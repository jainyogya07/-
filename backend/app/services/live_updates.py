"""Process-local event publisher used by WebSocket clients.

For multi-process deployments this interface is deliberately small so it can be
replaced by a Redis pub/sub implementation without changing API routes.
"""
import asyncio
import json
import uuid
from typing import Any, Set
from backend.app.config import settings
from backend.app.infrastructure.logging import logger


class LiveUpdates:
    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._redis = None
        self._listener: asyncio.Task | None = None
        self._origin = uuid.uuid4().hex

    async def start(self) -> None:
        """Enable Redis fan-out when configured; retain local mode on failure."""
        if not settings.REDIS_URL:
            return
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await self._redis.ping()
            self._listener = asyncio.create_task(self._consume_redis())
        except Exception:
            logger.exception("Redis live updates unavailable; using process-local fan-out")
            self._redis = None

    async def stop(self) -> None:
        if self._listener:
            self._listener.cancel()
            await asyncio.gather(self._listener, return_exceptions=True)
        if self._redis:
            await self._redis.aclose()

    async def publish(self, message: dict[str, Any]) -> None:
        self._fan_out(message)
        if self._redis:
            await self._redis.publish("weather-events", json.dumps({**message, "_origin": self._origin}))

    def _fan_out(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)

    async def _consume_redis(self) -> None:
        assert self._redis is not None
        pubsub = self._redis.pubsub()
        await pubsub.subscribe("weather-events")
        try:
            async for item in pubsub.listen():
                if item.get("type") != "message":
                    continue
                message = json.loads(item["data"])
                if message.pop("_origin", None) != self._origin:
                    self._fan_out(message)
        finally:
            await pubsub.unsubscribe("weather-events")
            await pubsub.aclose()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)


live_updates = LiveUpdates()
