"""Tests for EFI calculation and anomaly bounding box tracking."""

import pytest
import numpy as np
from app.data_ingestion.synthetic_data import (
    generate_synthetic_neps_g,
    generate_synthetic_era5_climatology,
)
from app.anomaly.efi import compute_efi_integral, compute_dataset_efi
from app.anomaly.bbox_tracker import (
    classify_severity,
    extract_spatial_features_at_step,
    track_anomaly_events,
)


@pytest.fixture(scope="module")
def benchmark_data():
    ds_forecast = generate_synthetic_neps_g(lead_hours=[24, 48, 72], n_members=6)
    ds_clim = generate_synthetic_era5_climatology()
    return ds_forecast, ds_clim


def test_compute_efi_integral():
    n_members = 10
    n_lats = 20
    n_lons = 20
    ens_vals = np.ones((n_members, n_lats, n_lons), dtype=np.float32) * 20.0
    clim_mean = np.ones((n_lats, n_lons), dtype=np.float32) * 5.0
    clim_std = np.ones((n_lats, n_lons), dtype=np.float32) * 2.0

    efi = compute_efi_integral(ens_vals, clim_mean, clim_std)
    assert efi.shape == (n_lats, n_lons)
    assert np.all(efi >= -1.0) and np.all(efi <= 1.0)
    # Since forecast is far higher than climatology, EFI should be close to 1.0
    assert float(efi.mean()) > 0.5


def test_compute_dataset_efi(benchmark_data):
    ds_forecast, ds_clim = benchmark_data
    efi_da = compute_dataset_efi(ds_forecast, ds_clim, "wind_speed_10m")

    assert efi_da.dims == ("time", "latitude", "longitude")
    assert efi_da.shape[0] == 3
    assert float(efi_da.max()) > 0.0


def test_extract_spatial_features():
    lats = np.linspace(10, 20, 21)
    lons = np.linspace(80, 90, 21)
    grid = np.zeros((21, 21), dtype=np.float32)
    # Set a 5x5 extreme anomaly patch
    grid[8:13, 8:13] = 0.85

    features = extract_spatial_features_at_step(grid, lats, lons, threshold=0.7)
    assert len(features) >= 1
    feat = features[0]
    assert feat["peak_efi"] == 0.85
    assert len(feat["bbox"]) == 4
    assert feat["bbox"][0] < feat["bbox"][2]  # min_lon < max_lon
    assert feat["bbox"][1] < feat["bbox"][3]  # min_lat < max_lat


def test_track_anomaly_events(benchmark_data):
    ds_forecast, ds_clim = benchmark_data
    efi_da = compute_dataset_efi(ds_forecast, ds_clim, "wind_speed_10m")
    events = track_anomaly_events(efi_da, hazard_type="cyclone", threshold=0.5)

    assert isinstance(events, list)
    if events:
        event = events[0]
        assert "event_id" in event
        assert "hazard" in event
        assert "severity" in event
        assert "bbox" in event
        assert "trajectory" in event
        assert len(event["trajectory"]) >= 1


def test_classify_severity():
    assert classify_severity(0.9, 0.85) == "extreme"
    assert classify_severity(0.75, 0.65) == "severe"
    assert classify_severity(0.55, 0.45) == "moderate"
    assert classify_severity(0.3, 0.2) == "low"
