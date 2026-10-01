"""Tests for NWP data ingestion and synthetic dataset generation."""

import pytest
import xarray as xr
from app.data_ingestion.synthetic_data import (
    generate_synthetic_neps_g,
    generate_synthetic_era5_climatology,
)
from app.data_ingestion.loaders import NWPDataLoader


def test_synthetic_neps_g_structure(tmp_path):
    save_file = tmp_path / "test_forecast.nc"
    ds = generate_synthetic_neps_g(
        lead_hours=[24, 48, 72],
        n_members=5,
        save_path=save_file,
    )

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.dims
    assert "member" in ds.dims
    assert "latitude" in ds.dims
    assert "longitude" in ds.dims
    assert ds.sizes["member"] == 5
    assert ds.sizes["time"] == 3

    assert "wind_speed_10m" in ds
    assert "total_precipitation" in ds
    assert "temperature_2m" in ds
    assert "mean_sea_level_pressure" in ds

    assert save_file.exists()


def test_synthetic_era5_climatology(tmp_path):
    save_file = tmp_path / "test_clim.nc"
    ds = generate_synthetic_era5_climatology(save_path=save_file)

    assert isinstance(ds, xr.Dataset)
    assert "latitude" in ds.dims
    assert "longitude" in ds.dims
    assert "wind_speed_mean" in ds
    assert "wind_speed_std" in ds
    assert "wind_speed_p95" in ds
    assert "total_precipitation_mean" in ds

    assert save_file.exists()


def test_nwp_loader(tmp_path):
    loader = NWPDataLoader(data_dir=tmp_path)
    ds_forecast, ds_clim = loader.get_or_create_benchmark_datasets()

    assert isinstance(ds_forecast, xr.Dataset)
    assert isinstance(ds_clim, xr.Dataset)
    assert (tmp_path / "neps_g_benchmark.nc").exists()
    assert (tmp_path / "era5_climatology_benchmark.nc").exists()
