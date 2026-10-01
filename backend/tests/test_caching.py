"""Unit tests for weather caching layer (Zarr and memory cache)."""

import pytest
import numpy as np
import xarray as xr
from app.cache.zarr_cache import WeatherCacheManager


@pytest.fixture
def cache_mgr(tmp_path):
    return WeatherCacheManager(cache_dir=tmp_path)


def test_memory_cache(cache_mgr):
    cache_mgr.set_memory_cache("test_key", {"metric": 42})
    cached = cache_mgr.get_memory_cache("test_key", max_age_seconds=60)
    assert cached == {"metric": 42}

    # Test cache expiration
    expired = cache_mgr.get_memory_cache("test_key", max_age_seconds=-1)
    assert expired is None


def test_zarr_cache_save_and_load(cache_mgr, tmp_path):
    ds = xr.Dataset(
        {"wind": (["time", "latitude", "longitude"], np.ones((2, 10, 10), dtype=np.float32))},
        coords={"time": [0, 1], "latitude": np.arange(10), "longitude": np.arange(10)},
    )

    path = cache_mgr.save_zarr(ds, "test_field")
    assert path.exists()

    loaded = cache_mgr.load_zarr("test_field")
    assert loaded is not None
    assert "wind" in loaded
    assert loaded["wind"].shape == (2, 10, 10)


def test_cache_clear(cache_mgr):
    cache_mgr.set_memory_cache("key1", "val1")
    assert cache_mgr.get_memory_cache("key1") == "val1"
    cache_mgr.clear()
    assert cache_mgr.get_memory_cache("key1") is None
