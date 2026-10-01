"""Tests for spatial cropping and GeoJSON conversion utilities."""

import pytest
import numpy as np
from app.data_ingestion.synthetic_data import generate_synthetic_neps_g
from app.spatial.cropper import crop_spatial_bbox, crop_temporal_window
from app.spatial.geojson import (
    bbox_to_geojson_polygon,
    trajectory_to_geojson_linestring,
    event_to_geojson_features,
    events_to_geojson_feature_collection,
)


@pytest.fixture(scope="module")
def sample_ds():
    return generate_synthetic_neps_g(lead_hours=[24, 48, 72], n_members=2)


def test_crop_spatial_bbox(sample_ds):
    bbox = [80.0, 15.0, 90.0, 25.0]
    cropped = crop_spatial_bbox(sample_ds, bbox)

    assert float(cropped.latitude.min()) >= 15.0
    assert float(cropped.latitude.max()) <= 25.0
    assert float(cropped.longitude.min()) >= 80.0
    assert float(cropped.longitude.max()) <= 90.0


def test_crop_temporal_window(sample_ds):
    cropped = crop_temporal_window(sample_ds, start_hour=48, end_hour=72)
    assert len(cropped["lead_hour"]) == 2
    assert list(cropped["lead_hour"].values) == [48, 72]


def test_bbox_to_geojson():
    bbox = [82.0, 18.0, 86.0, 22.0]
    feature = bbox_to_geojson_polygon(bbox, properties={"name": "test_box"})

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["name"] == "test_box"
    coords = feature["geometry"]["coordinates"][0]
    assert len(coords) == 5
    assert coords[0] == coords[-1]  # Closed polygon


def test_trajectory_to_geojson():
    traj = [
        {"lead_hour": 24, "centroid": [88.0, 10.0]},
        {"lead_hour": 48, "centroid": [87.0, 13.0]},
    ]
    feature = trajectory_to_geojson_linestring(traj, properties={"event_id": "EVT-01"})
    assert feature["geometry"]["type"] == "LineString"
    assert len(feature["geometry"]["coordinates"]) == 2


def test_events_to_geojson_feature_collection():
    event = {
        "event_id": "EVT-001",
        "hazard": "cyclone",
        "severity": "extreme",
        "probability": 0.9,
        "max_efi": 0.88,
        "bbox": [82.0, 15.0, 88.0, 22.0],
        "forecast_hours": [24, 48],
        "trajectory": [
            {"lead_hour": 24, "centroid": [85.0, 17.0], "peak_efi": 0.85, "bbox": [83.0, 16.0, 87.0, 18.0]},
            {"lead_hour": 48, "centroid": [84.0, 20.0], "peak_efi": 0.88, "bbox": [82.0, 19.0, 86.0, 21.0]},
        ],
    }

    fc = events_to_geojson_feature_collection([event])
    assert fc["type"] == "FeatureCollection"
    # 1 bbox polygon + 1 trajectory linestring + 2 waypoint points = 4 features
    assert len(fc["features"]) == 4
