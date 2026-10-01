"""
Dataset Registry & Metadata Management.
Tracks metadata, coordinate grids, and specifications for supported NWP datasets:
ERA5, IMDAA, NCUM, and NEPS-G.
"""

from typing import Any
from app.data_ingestion.loaders import NWPDataLoader
from app.data_ingestion.synthetic_data import (
    generate_synthetic_neps_g,
    generate_synthetic_era5_climatology,
)


class DatasetRegistry:
    """Registry of meteorological datasets for MoES extreme weather tracking."""

    DATASETS = {
        "neps_g": {
            "name": "NCMRWF Global Ensemble (NEPS-G)",
            "format": "GRIB2 / NetCDF",
            "spatial_resolution": "12 km (~0.5 deg)",
            "type": "Ensemble (EPS)",
            "ensemble_members": 23,
            "forecast_lead_days": 10,
            "variables": ["wind_speed_10m", "total_precipitation", "temperature_2m", "mean_sea_level_pressure"],
            "institution": "National Centre for Medium Range Weather Forecasting (NCMRWF)",
        },
        "ncum": {
            "name": "NCMRWF Unified Model (NCUM)",
            "format": "GRIB2 / NetCDF",
            "spatial_resolution": "12 km",
            "type": "Deterministic NWP",
            "ensemble_members": 1,
            "forecast_lead_days": 10,
            "variables": ["wind_speed_10m", "total_precipitation", "temperature_2m", "mean_sea_level_pressure"],
            "institution": "NCMRWF / MoES",
        },
        "era5": {
            "name": "ECMWF ERA5 Atmospheric Reanalysis",
            "format": "NetCDF4",
            "spatial_resolution": "31 km / 0.25 deg",
            "type": "30-Year Historical Reanalysis Climatological Baseline (1991-2020)",
            "variables": ["wind_speed_mean", "wind_speed_std", "wind_speed_p95", "total_precipitation_mean", "total_precipitation_std"],
            "institution": "ECMWF / Copernicus Climate Change Service",
        },
        "imdaa": {
            "name": "Indian Monsoon Data Assimilation and Analysis (IMDAA)",
            "format": "NetCDF4 / GRIB2",
            "spatial_resolution": "12 km regional reanalysis over India",
            "type": "Historical Baseline Climatology",
            "variables": ["precipitation", "surface_wind", "air_temperature"],
            "institution": "NCMRWF / MoES / Met Office UK",
        },
    }

    @classmethod
    def get_metadata(cls, dataset_id: str) -> dict[str, Any] | None:
        return cls.DATASETS.get(dataset_id.lower())

    @classmethod
    def list_datasets(cls) -> list[dict[str, Any]]:
        return [{"id": k, **v} for k, v in cls.DATASETS.items()]


__all__ = [
    "DatasetRegistry",
    "NWPDataLoader",
    "generate_synthetic_neps_g",
    "generate_synthetic_era5_climatology",
]
