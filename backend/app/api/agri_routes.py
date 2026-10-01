"""
Agri-Tech & Market Intelligence REST API Routes (Modules 14 & 15).
Provides Pest & Disease Risk Assessment and APMC Mandi Market Impact Forecasts.
"""

from fastapi import APIRouter, Query
from app.agri.pest_risk import pest_engine
from app.agri.market_intelligence import market_engine

router = APIRouter(prefix="/weather/agri", tags=["Agri-Tech & Market Intelligence (Modules 14 & 15)"])


@router.get("/pest-risk")
def get_pest_disease_risk(
    crop: str = Query("rice", description="Crop name (rice, cotton, wheat)"),
    stage: str = Query("vegetative", description="Crop growth stage (sowing_nursery, vegetative, flowering, maturity)"),
    region: str = Query("Odisha", description="Target agricultural region or state"),
    temp_c: float = Query(29.5, description="Ambient / forecast temperature (°C)"),
    rh_pct: float = Query(88.0, description="Relative humidity percentage (%)"),
    rain_mm: float = Query(65.0, description="24h precipitation (mm)"),
    wind_kmh: float = Query(45.0, description="Surface wind speed (km/h)"),
):
    """
    Module 14: Evaluates pest and disease infection risk based on atmospheric drivers,
    crop type, and growth stage vulnerabilities.
    """
    return pest_engine.evaluate_risk(
        crop=crop,
        stage=stage,
        temp_c=temp_c,
        rh_pct=rh_pct,
        rain_mm=rain_mm,
        wind_kmh=wind_kmh,
        region=region,
    )


@router.get("/market-intelligence")
def get_market_intelligence(
    region: str = Query("odisha", description="Target region / market basin (odisha, west_bengal, punjab, gujarat)"),
    crop: str = Query("rice", description="Crop commodity name"),
    hazard: str = Query("cyclone", description="Extreme weather hazard driving supply shock"),
    efi_intensity: float = Query(0.92, description="Extreme Forecast Index intensity (0.0 to 1.0)"),
):
    """
    Module 15: Projects APMC mandi modal price fluctuations, arrival deficits,
    and transit corridor risks resulting from the extreme weather hazard.
    """
    return market_engine.evaluate_market_impact(
        region=region,
        crop=crop,
        hazard=hazard,
        efi_intensity=efi_intensity,
    )


@router.get("/crops")
def list_supported_crops():
    """
    Returns supported agricultural crops and their recognized growth stages.
    """
    return {
        "crops": [
            {"id": k, "name": v["name"], "stages": v["stages"], "pathogen_count": len(v["pathogens"])}
            for k, v in pest_engine.CROP_PROFILES.items()
        ]
    }
