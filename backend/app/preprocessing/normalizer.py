"""
Meteorological Variable Normalizer.
Standardizes atmospheric variables for downstream Deep Learning models (Spherical GNN & Diffusion).
"""

from typing import Literal
import numpy as np
import xarray as xr


class MeteorologicalNormalizer:
    """
    Handles normalization, scaling, and unit conversions
    for atmospheric variables according to physics constraints.
    """

    # Typical physical bounds for weather variables in South Asia / Global NWP
    DEFAULT_STATS = {
        "wind_speed_10m": {"mean": 8.0, "std": 5.0, "min": 0.0, "max": 75.0},
        "total_precipitation": {"mean": 10.0, "std": 25.0, "min": 0.0, "max": 400.0},
        "temperature_2m": {"mean": 300.0, "std": 8.0, "min": 260.0, "max": 330.0},
        "mean_sea_level_pressure": {"mean": 1010.0, "std": 8.0, "min": 920.0, "max": 1040.0},
    }

    def __init__(self, custom_stats: dict | None = None):
        self.stats = {**self.DEFAULT_STATS, **(custom_stats or {})}

    def z_score_normalize(self, da: xr.DataArray, var_name: str) -> xr.DataArray:
        """Standardizes array to zero mean and unit variance."""
        var_stat = self.stats.get(var_name, {"mean": float(da.mean()), "std": float(da.std()) + 1e-6})
        return (da - var_stat["mean"]) / (var_stat["std"] + 1e-6)

    def min_max_scale(self, da: xr.DataArray, var_name: str, target_range: tuple[float, float] = (0.0, 1.0)) -> xr.DataArray:
        """Scales array to target range (default [0, 1])."""
        var_stat = self.stats.get(var_name, {"min": float(da.min()), "max": float(da.max()) + 1e-6})
        norm = (da - var_stat["min"]) / (var_stat["max"] - var_stat["min"] + 1e-6)
        low, high = target_range
        return norm * (high - low) + low

    @staticmethod
    def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
        """Converts Kelvin to Celsius."""
        return da - 273.15

    @staticmethod
    def ms_to_kmh(da: xr.DataArray) -> xr.DataArray:
        """Converts m/s to km/h."""
        return da * 3.6
