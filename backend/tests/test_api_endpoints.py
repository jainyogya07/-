"""Integration tests for all FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_weather_forecast():
    response = client.get("/weather/forecast?variable=wind_speed_10m")
    assert response.status_code == 200
    data = response.json()
    assert data["variable"] == "wind_speed_10m"
    assert "bounding_box" in data
    assert "summary" in data
    assert len(data["lead_hours"]) > 0


def test_weather_ensemble_point_probe():
    response = client.get("/weather/ensemble?variable=wind_speed_10m&lat=15.0&lon=85.0&lead_hour=72")
    assert response.status_code == 200
    data = response.json()
    assert "ensemble_mean" in data
    assert "ensemble_spread" in data
    assert "p95" in data
    assert "member_values" in data


def test_weather_ensemble_spatial_aggregate():
    response = client.get("/weather/ensemble?variable=wind_speed_10m&lead_hour=72")
    assert response.status_code == 200
    data = response.json()
    assert "spatial_aggregate" in data
    assert "max_ensemble_mean" in data["spatial_aggregate"]


def test_weather_anomaly_json():
    response = client.get("/weather/anomaly?variable=wind_speed_10m&threshold=0.60&format=json")
    assert response.status_code == 200
    data = response.json()
    assert "active_events_count" in data
    assert "events" in data


def test_weather_anomaly_geojson():
    response = client.get("/weather/anomaly?variable=wind_speed_10m&threshold=0.60&format=geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data


def test_weather_region():
    response = client.get("/weather/region/bay_of_bengal?variable=wind_speed_10m&lead_hour=72")
    assert response.status_code == 200
    data = response.json()
    assert data["region_id"] == "bay_of_bengal"
    assert "regional_mean" in data
    assert "regional_max" in data


def test_weather_region_not_found():
    response = client.get("/weather/region/unknown_region_xyz")
    assert response.status_code == 404


def test_weather_timeline():
    response = client.get("/weather/timeline?variable=wind_speed_10m")
    assert response.status_code == 200
    data = response.json()
    assert "timeline" in data
    assert len(data["timeline"]) > 0
    first_step = data["timeline"][0]
    assert "lead_hour" in first_step
    assert "max_efi" in first_step
    assert "peak_value" in first_step


def test_weather_news_feed():
    response = client.get("/weather/news?hazard=cyclone&region=Odisha&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert data["hazard"] == "cyclone"
    assert data["articles_count"] >= 1
    assert len(data["articles"]) >= 1


def test_weather_news_status():
    response = client.get("/weather/news/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_primary_provider" in data


def test_weather_news_key_update():
    response = client.post("/weather/news/key", json={"api_key": "sample_mock_key"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
