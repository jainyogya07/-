"""Tests for Modules 14 & 15: Pest & Disease Risk and Mandi Market Intelligence."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agri.pest_risk import pest_engine
from app.agri.market_intelligence import market_engine

client = TestClient(app)


def test_pest_disease_risk_calculation():
    result = pest_engine.evaluate_risk(
        crop="rice",
        stage="vegetative",
        temp_c=29.0,
        rh_pct=90.0,
        rain_mm=60.0,
        wind_kmh=40.0,
        region="Odisha",
    )

    assert result["crop"] == "Rice (Paddy)"
    assert result["overall_pest_disease_risk"] >= 0.70
    assert result["threat_level"] in ["high", "critical"]
    assert len(result["pathogens_evaluated"]) >= 2

    # Check that bacterial leaf blight is flagged
    blight = next(p for p in result["pathogens_evaluated"] if "Bacterial Leaf Blight" in p["pathogen"])
    assert blight["contributing_factors"]["humidity_exceeded"] is True
    assert blight["contributing_factors"]["wind_rain_vector_active"] is True
    assert "advisory" in blight


def test_market_intelligence_evaluation():
    impact = market_engine.evaluate_market_impact(
        region="odisha",
        crop="rice",
        hazard="cyclone",
        efi_intensity=0.95,
    )

    assert "mandi_name" in impact
    assert "Odisha" in impact["location"]["state"]
    assert impact["commodity"]["modal_price_inr_quintal"] > 2000.0
    assert impact["weather_shock_forecast"]["projected_arrival_reduction_72h_pct"] >= 40.0
    assert impact["weather_shock_forecast"]["volatility_status"] == "Severe Volatility"
    assert len(impact["weather_shock_forecast"]["impacted_transit_routes"]) >= 1


def test_api_pest_risk_endpoint():
    response = client.get("/weather/agri/pest-risk?crop=rice&stage=vegetative&region=Odisha")
    assert response.status_code == 200
    data = response.json()
    assert "overall_pest_disease_risk" in data
    assert "pathogens_evaluated" in data
    assert "environmental_conditions" in data


def test_api_market_intelligence_endpoint():
    response = client.get("/weather/agri/market-intelligence?region=odisha&crop=rice&hazard=cyclone&efi_intensity=0.92")
    assert response.status_code == 200
    data = response.json()
    assert "mandi_name" in data
    assert "weather_shock_forecast" in data
    assert "fpo_and_procurement_advisory" in data


def test_api_crops_endpoint():
    response = client.get("/weather/agri/crops")
    assert response.status_code == 200
    data = response.json()
    assert len(data["crops"]) >= 3
    assert any(c["id"] == "rice" for c in data["crops"])
