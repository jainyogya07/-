"""
Synthetic NWP & Climatology Data Generator.
Produces physically realistic 4D/5D NetCDF datasets matching NCMRWF NEPS-G (12km ensemble)
and ERA5 historical 30-year baseline for testing extreme weather tracking (e.g., Cyclone Amphan).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def generate_synthetic_neps_g(
    start_date: str = "2026-10-01",
    lead_hours: list[int] = None,
    n_members: int = 10,
    save_path: Path | None = None,
) -> xr.Dataset:
    """
    Generates a synthetic NEPS-G 12km ensemble dataset simulating a moving severe cyclone
    in the Bay of Bengal heading towards Odisha/West Bengal over a 10-day forecast window.
    """
    if lead_hours is None:
        lead_hours = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

    # Grid: India & Indian Ocean Basin (5°N to 35°N, 65°E to 98°E)
    lats = np.arange(5.0, 35.1, 0.5)  # 61 lat points
    lons = np.arange(65.0, 98.1, 0.5)  # 67 lon points

    n_lats = len(lats)
    n_lons = len(lons)
    n_times = len(lead_hours)

    times = [
        pd.Timestamp(start_date) + pd.Timedelta(hours=h) for h in lead_hours
    ]
    members = list(range(n_members))

    # Base climatological fields
    base_t2m = np.zeros((n_times, n_members, n_lats, n_lons), dtype=np.float32)
    base_u10 = np.zeros((n_times, n_members, n_lats, n_lons), dtype=np.float32)
    base_tp = np.zeros((n_times, n_members, n_lats, n_lons), dtype=np.float32)
    base_mslp = (
        np.zeros((n_times, n_members, n_lats, n_lons), dtype=np.float32)
        + 1012.0
    )

    # Ambient conditions
    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")

    # Ambient temperature gradient (colder north, warmer equator)
    temp_profile = 303.0 - (lat_mesh - 5.0) * 0.3
    # Ambient breeze
    wind_profile = 6.0 + np.sin(np.radians(lat_mesh)) * 3.0

    # Cyclone trajectory starting at ~10°N, 88°E moving NW towards Odisha coast ~20°N, 86°E
    track_lats = np.linspace(10.0, 22.0, n_times)
    track_lons = np.linspace(88.0, 85.5, n_times)

    for t_idx in range(n_times):
        c_lat = track_lats[t_idx]
        c_lon = track_lons[t_idx]

        for m in range(n_members):
            # Member perturbation (atmospheric chaos / ensemble spread)
            pert_lat = c_lat + np.random.normal(0, 0.25 + 0.05 * t_idx)
            pert_lon = c_lon + np.random.normal(0, 0.25 + 0.05 * t_idx)
            radius = np.sqrt(
                (lat_mesh - pert_lat) ** 2 + (lon_mesh - pert_lon) ** 2
            )

            # Vortex structure (Rankine-like vortex for cyclone)
            vortex_scale = 3.5  # degrees core
            cyclone_intensity = 35.0 + np.random.uniform(
                -4.0, 6.0
            )  # m/s peak wind
            # Peak intensity at days 3-5 (hours 72-120), then weakens after landfall
            if t_idx in [2, 3, 4]:
                cyclone_intensity += 18.0  # Super cyclonic storm
            elif t_idx > 6:
                cyclone_intensity = max(15.0, cyclone_intensity - 15.0)

            # Wind speed field
            wind_anomaly = cyclone_intensity * np.exp(
                -((radius / vortex_scale) ** 2)
            )
            base_u10[t_idx, m] = wind_profile + wind_anomaly + np.random.normal(0, 1.2, (n_lats, n_lons))

            # Severe precipitation field (mm/24h)
            precip_anomaly = (cyclone_intensity * 3.2) * np.exp(
                -((radius / (vortex_scale * 1.2)) ** 2)
            )
            base_tp[t_idx, m] = np.clip(
                precip_anomaly + np.random.exponential(1.5, (n_lats, n_lons)),
                0,
                380.0,
            )

            # Central low pressure (hPa)
            pressure_drop = (cyclone_intensity * 1.1) * np.exp(
                -((radius / (vortex_scale * 1.5)) ** 2)
            )
            base_mslp[t_idx, m] = 1010.0 - pressure_drop + np.random.normal(0, 0.8, (n_lats, n_lons))

            # Temperature field (slight cooling under storm cloud shield, warmer inland)
            cooling = 4.0 * np.exp(-((radius / vortex_scale) ** 2))
            base_t2m[t_idx, m] = temp_profile - cooling + np.random.normal(0, 0.6, (n_lats, n_lons))

    ds = xr.Dataset(
        data_vars={
            "wind_speed_10m": (
                ["time", "member", "latitude", "longitude"],
                base_u10,
                {"units": "m/s", "long_name": "10m Wind Speed"},
            ),
            "total_precipitation": (
                ["time", "member", "latitude", "longitude"],
                base_tp,
                {"units": "mm", "long_name": "Total 24h Precipitation"},
            ),
            "mean_sea_level_pressure": (
                ["time", "member", "latitude", "longitude"],
                base_mslp,
                {"units": "hPa", "long_name": "Mean Sea Level Pressure"},
            ),
            "temperature_2m": (
                ["time", "member", "latitude", "longitude"],
                base_t2m,
                {"units": "K", "long_name": "2-meter Temperature"},
            ),
        },
        coords={
            "time": times,
            "lead_hour": ("time", lead_hours),
            "member": members,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "NCMRWF NEPS-G 12km Ensemble Forecast (Synthetic Benchmark)",
            "institution": "Ministry of Earth Sciences (MoES) / NCMRWF",
            "source": "AI-Driven Spatio-Temporal Weather Tracking System",
            "spatial_resolution": "0.5 deg (~12 km surrogate)",
            "ensemble_members": n_members,
        },
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(save_path, engine="h5netcdf")

    return ds


def generate_synthetic_era5_climatology(
    save_path: Path | None = None,
) -> xr.Dataset:
    """
    Generates a 30-year climatological baseline dataset (ERA5/IMDAA)
    containing mean, standard deviation, and 95th/99th percentiles
    to compute Extreme Forecast Index (EFI) anomalies.
    """
    lats = np.arange(5.0, 35.1, 0.5)
    lons = np.arange(65.0, 98.1, 0.5)
    n_lats = len(lats)
    n_lons = len(lons)

    lat_mesh, _ = np.meshgrid(lats, lons, indexing="ij")

    # Climatological normal for wind speed (m/s)
    wind_mean = 6.0 + np.sin(np.radians(lat_mesh)) * 3.0
    wind_std = np.full((n_lats, n_lons), 3.2, dtype=np.float32)
    wind_p95 = wind_mean + 1.645 * wind_std  # ~95th percentile
    wind_p99 = wind_mean + 2.326 * wind_std  # ~99th percentile

    # Climatological normal for precipitation (mm)
    precip_mean = np.full((n_lats, n_lons), 8.5, dtype=np.float32)
    precip_std = np.full((n_lats, n_lons), 12.0, dtype=np.float32)
    precip_p95 = np.full((n_lats, n_lons), 38.0, dtype=np.float32)
    precip_p99 = np.full((n_lats, n_lons), 75.0, dtype=np.float32)

    # Climatological normal for temperature (K)
    temp_mean = 302.0 - (lat_mesh - 5.0) * 0.3
    temp_std = np.full((n_lats, n_lons), 2.5, dtype=np.float32)
    temp_p95 = temp_mean + 1.645 * temp_std
    temp_p99 = temp_mean + 2.326 * temp_std

    ds = xr.Dataset(
        data_vars={
            "wind_speed_mean": (["latitude", "longitude"], wind_mean.astype(np.float32)),
            "wind_speed_std": (["latitude", "longitude"], wind_std),
            "wind_speed_p95": (["latitude", "longitude"], wind_p95),
            "wind_speed_p99": (["latitude", "longitude"], wind_p99),
            "total_precipitation_mean": (["latitude", "longitude"], precip_mean),
            "total_precipitation_std": (["latitude", "longitude"], precip_std),
            "total_precipitation_p95": (["latitude", "longitude"], precip_p95),
            "total_precipitation_p99": (["latitude", "longitude"], precip_p99),
            "temperature_mean": (["latitude", "longitude"], temp_mean.astype(np.float32)),
            "temperature_std": (["latitude", "longitude"], temp_std),
            "temperature_p95": (["latitude", "longitude"], temp_p95),
            "temperature_p99": (["latitude", "longitude"], temp_p99),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "ERA5/IMDAA 30-Year Climatological Baseline Distribution (1991-2020)",
            "institution": "MoES / NCMRWF / ECMWF",
            "spatial_resolution": "0.5 deg",
        },
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(save_path, engine="h5netcdf")

    return ds
