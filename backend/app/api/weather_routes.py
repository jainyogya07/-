"""
Weather Forecast, Ensemble, Anomaly, and Spatial REST API Routes.
Exposes endpoints required by the Team Leader (Frontend) and Palak (AI Inference).
"""

from typing import Literal
from fastapi import APIRouter, HTTPException, Query
import numpy as np
from app.config import settings
from app.data_ingestion.loaders import NWPDataLoader
from app.preprocessing.ensemble import compute_ensemble_statistics
from app.preprocessing.pipeline import WeatherPreprocessingPipeline
from app.anomaly.efi import compute_dataset_efi
from app.anomaly.bbox_tracker import track_anomaly_events
from app.spatial.cropper import crop_spatial_bbox
from app.spatial.geojson import events_to_geojson_feature_collection
from app.cache.zarr_cache import cache_manager

router = APIRouter(prefix="/weather", tags=["Weather Data & Anomalies"])

loader = NWPDataLoader()
pipeline = WeatherPreprocessingPipeline()

# Lazy-loaded / cached in-memory benchmark datasets
_FORECAST_DS = None
_CLIMATOLOGY_DS = None


def get_datasets():
    global _FORECAST_DS, _CLIMATOLOGY_DS
    if _FORECAST_DS is None or _CLIMATOLOGY_DS is None:
        _FORECAST_DS, _CLIMATOLOGY_DS = loader.get_or_create_benchmark_datasets()
    return _FORECAST_DS, _CLIMATOLOGY_DS


@router.get("/forecast")
def get_forecast(
    variable: str = Query("wind_speed_10m", description="Variable name (wind_speed_10m, total_precipitation, temperature_2m, mean_sea_level_pressure)"),
    lead_hour: int | None = Query(None, description="Optional lead forecast hour (e.g. 24, 72, 120)"),
):
    """
    Returns forecast metadata, available spatial coordinates, lead times,
    and aggregate summary for the selected variable.
    """
    ds_forecast, _ = get_datasets()
    if variable not in ds_forecast:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found. Available: {list(ds_forecast.data_vars)}")

    da = ds_forecast[variable]
    if lead_hour is not None:
        if "lead_hour" in ds_forecast.coords:
            valid_hours = list(ds_forecast["lead_hour"].values)
            if lead_hour not in valid_hours:
                raise HTTPException(status_code=400, detail=f"Invalid lead_hour. Valid: {valid_hours}")
            da = da.sel(time=(ds_forecast["lead_hour"] == lead_hour))

    lats = [float(da.latitude.min()), float(da.latitude.max())]
    lons = [float(da.longitude.min()), float(da.longitude.max())]
    lead_hours = [int(h) for h in ds_forecast["lead_hour"].values] if "lead_hour" in ds_forecast.coords else []

    return {
        "dataset": "NCMRWF NEPS-G 12km Global Ensemble",
        "variable": variable,
        "units": da.attrs.get("units", ""),
        "lead_hours": lead_hours,
        "ensemble_members": int(ds_forecast.sizes.get("member", 1)),
        "bounding_box": [lons[0], lats[0], lons[1], lats[1]],
        "resolution_deg": 0.5,
        "summary": {
            "min": round(float(da.min()), 2),
            "max": round(float(da.max()), 2),
            "mean": round(float(da.mean()), 2),
        },
    }


@router.get("/ensemble")
def get_ensemble(
    variable: str = Query("wind_speed_10m", description="Variable name"),
    lat: float | None = Query(None, description="Latitude for point probe (5.0 to 35.0)"),
    lon: float | None = Query(None, description="Longitude for point probe (65.0 to 98.0)"),
    lead_hour: int = Query(72, description="Forecast lead hour"),
):
    """
    Returns ensemble metrics (mean, spread/std, and percentiles p10..p99)
    across all perturbation members for a specific location or regional aggregate.
    """
    ds_forecast, _ = get_datasets()
    if variable not in ds_forecast:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found.")

    cache_key = f"ens_{variable}_{lead_hour}_{lat}_{lon}"
    cached = cache_manager.get_memory_cache(cache_key)
    if cached:
        return cached

    # Select lead time
    if "lead_hour" in ds_forecast.coords:
        time_mask = ds_forecast["lead_hour"] == lead_hour
        if not time_mask.any():
            raise HTTPException(status_code=400, detail=f"Lead hour {lead_hour} not in forecast horizon.")
        step_ds = ds_forecast.sel(time=time_mask).squeeze()
    else:
        step_ds = ds_forecast.isel(time=0)

    # Compute statistics across ensemble dimension
    stats = compute_ensemble_statistics(step_ds, variable)

    if lat is not None and lon is not None:
        # Point probe with nearest neighbor interpolation
        pt = stats.sel(latitude=lat, longitude=lon, method="nearest")
        raw_members = step_ds[variable].sel(latitude=lat, longitude=lon, method="nearest").values
        result = {
            "variable": variable,
            "lead_hour": lead_hour,
            "coordinates": {"lat": float(pt.latitude), "lon": float(pt.longitude)},
            "ensemble_mean": round(float(pt[f"{variable}_mean"]), 2),
            "ensemble_spread": round(float(pt[f"{variable}_spread"]), 2),
            "ensemble_median": round(float(pt[f"{variable}_median"]), 2),
            "p10": round(float(pt[f"{variable}_p10"]), 2),
            "p90": round(float(pt[f"{variable}_p90"]), 2),
            "p95": round(float(pt[f"{variable}_p95"]), 2),
            "p99": round(float(pt[f"{variable}_p99"]), 2),
            "member_values": [round(float(v), 2) for v in raw_members],
        }
    else:
        # Spatial regional aggregate
        result = {
            "variable": variable,
            "lead_hour": lead_hour,
            "spatial_aggregate": {
                "max_ensemble_mean": round(float(stats[f"{variable}_mean"].max()), 2),
                "max_ensemble_spread": round(float(stats[f"{variable}_spread"].max()), 2),
                "peak_p95": round(float(stats[f"{variable}_p95"].max()), 2),
                "peak_p99": round(float(stats[f"{variable}_p99"].max()), 2),
            },
        }

    cache_manager.set_memory_cache(cache_key, result)
    return result


@router.get("/anomaly")
def get_anomalies(
    variable: str = Query("wind_speed_10m", description="Variable to evaluate"),
    threshold: float = Query(0.65, description="EFI threshold to flag anomaly (0.5 to 0.95)"),
    format: Literal["json", "geojson"] = Query("json", description="Output format: json or geojson FeatureCollection"),
):
    """
    Evaluates Extreme Forecast Index (EFI) against 30-year ERA5 climatological baseline.
    Isolates moving weather anomalies and generates dynamic spatio-temporal bounding boxes.
    """
    cache_key = f"anom_{variable}_{threshold}_{format}"
    cached = cache_manager.get_memory_cache(cache_key)
    if cached:
        return cached

    ds_forecast, ds_clim = get_datasets()
    if variable not in ds_forecast:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found.")

    # Calculate EFI array
    efi_da = compute_dataset_efi(ds_forecast, ds_clim, variable)

    hazard_label = "tropical_cyclone" if "wind" in variable else "extreme_precipitation"
    events = track_anomaly_events(efi_da, hazard_type=hazard_label, threshold=threshold)

    if format == "geojson":
        result = events_to_geojson_feature_collection(events)
    else:
        result = {
            "hazard_evaluated": hazard_label,
            "variable": variable,
            "efi_threshold": threshold,
            "active_events_count": len(events),
            "events": events,
        }

    cache_manager.set_memory_cache(cache_key, result)
    return result


@router.get("/region/{id}")
def get_region_data(
    id: str,
    variable: str = Query("wind_speed_10m", description="Variable to query"),
    lead_hour: int | None = Query(72, description="Forecast lead hour"),
):
    """
    Returns cropped regional metrics and bounding boxes for predefined Indian meteorological regions
    (e.g., bay_of_bengal, coastal_odisha, north_india, arabian_sea).
    """
    if id not in settings.REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{id}' not found. Available regions: {list(settings.REGIONS.keys())}",
        )

    reg_info = settings.REGIONS[id]
    bbox = reg_info["bbox"]

    ds_forecast, _ = get_datasets()
    cropped = crop_spatial_bbox(ds_forecast, bbox)

    if lead_hour is not None and "lead_hour" in cropped.coords:
        cropped = cropped.sel(time=(cropped["lead_hour"] == lead_hour)).squeeze()

    da = cropped[variable]
    mean_val = float(da.mean(dim="member").mean())
    max_val = float(da.max())

    return {
        "region_id": id,
        "region_name": reg_info["name"],
        "bbox": bbox,
        "lead_hour": lead_hour,
        "variable": variable,
        "regional_mean": round(mean_val, 2),
        "regional_max": round(max_val, 2),
        "status": "alert" if max_val > 30.0 else "normal",
    }


@router.get("/timeline")
def get_anomaly_timeline(
    variable: str = Query("wind_speed_10m", description="Variable name"),
):
    """
    Returns the temporal evolution curve (time series) of anomaly intensity,
    peak wind/rain, and ensemble spread across the 3 to 10-day forecast horizon.
    """
    ds_forecast, ds_clim = get_datasets()
    efi_da = compute_dataset_efi(ds_forecast, ds_clim, variable)

    lead_hours = [int(h) for h in ds_forecast["lead_hour"].values] if "lead_hour" in ds_forecast.coords else []
    timeline = []

    for t_idx, lead_h in enumerate(lead_hours):
        step_efi = efi_da.isel(time=t_idx)
        raw_var = ds_forecast[variable].isel(time=t_idx)

        peak_efi = float(step_efi.max())
        max_physical = float(raw_var.max())
        mean_physical = float(raw_var.mean())
        spread = float(raw_var.std(dim="member").max())

        timeline.append({
            "lead_hour": lead_h,
            "day": round(lead_h / 24, 1),
            "max_efi": round(peak_efi, 3),
            "peak_value": round(max_physical, 2),
            "mean_value": round(mean_physical, 2),
            "max_ensemble_spread": round(spread, 2),
            "threat_level": "extreme" if peak_efi >= 0.85 else "high" if peak_efi >= 0.70 else "moderate",
        })

    return {
        "variable": variable,
        "forecast_horizon_hours": lead_hours[-1] if lead_hours else 240,
        "timeline": timeline,
    }
