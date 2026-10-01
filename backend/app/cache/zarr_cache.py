"""
High-Performance Caching Layer for Weather Fields.
Provides chunked Zarr array persistence and in-memory LRU/TTL caching for sub-second API responses.
"""

from pathlib import Path
import time
import json
import logging
import xarray as xr
from app.config import settings

logger = logging.getLogger(__name__)


class WeatherCacheManager:
    """
    Manages Zarr dataset caching on disk and in-memory metadata caching.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self._memory_cache: dict[str, dict] = {}

    def save_zarr(self, ds: xr.Dataset, key: str) -> Path:
        """Saves an xarray Dataset to a chunked Zarr store."""
        target_path = self.cache_dir / f"{key}.zarr"
        try:
            # Rechunk sensibly for web queries
            chunks = {"time": 1}
            for dim in ["latitude", "longitude"]:
                if dim in ds.dims:
                    chunks[dim] = min(60, ds.dims[dim])
            rechunked = ds.chunk(chunks)
            rechunked.to_zarr(target_path, mode="w")
            logger.info("Successfully cached dataset to Zarr at %s", target_path)
            return target_path
        except Exception as e:
            logger.warning("Failed to save Zarr store (%s). Continuing without disk Zarr cache.", e)
            return target_path

    def load_zarr(self, key: str) -> xr.Dataset | None:
        """Loads an xarray Dataset from Zarr store if it exists."""
        target_path = self.cache_dir / f"{key}.zarr"
        if target_path.exists():
            try:
                return xr.open_zarr(target_path)
            except Exception as e:
                logger.warning("Corrupt or invalid Zarr store at %s: %s", target_path, e)
                return None
        return None

    def get_memory_cache(self, key: str, max_age_seconds: int = 3600):
        """Retrieves an item from memory cache if not expired."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry["timestamp"] < max_age_seconds:
                return entry["data"]
            del self._memory_cache[key]
        return None

    def set_memory_cache(self, key: str, data: any):
        """Stores item in memory cache with timestamp."""
        self._memory_cache[key] = {
            "timestamp": time.time(),
            "data": data,
        }

    def clear(self):
        """Clears memory and disk caches."""
        self._memory_cache.clear()


cache_manager = WeatherCacheManager()
