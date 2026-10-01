"""
NWP and Climatology Data Loader.
Supports loading GRIB2 files (via cfgrib/ecCodes) and NetCDF4 files (via h5netcdf/netCDF4)
with lazy Dask chunking for parallel out-of-core computation.
"""

from pathlib import Path
import logging
import xarray as xr
from app.config import settings
from app.data_ingestion.synthetic_data import (
    generate_synthetic_neps_g,
    generate_synthetic_era5_climatology,
)

logger = logging.getLogger(__name__)


class NWPDataLoader:
    """
    Unified Loader for NCUM (deterministic 12km), NEPS-G (ensemble 12km),
    and ERA5/IMDAA reanalysis baselines.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.DATA_DIR

    def load_dataset(self, file_path: str | Path, chunks: dict | None = None) -> xr.Dataset:
        """
        Loads a NetCDF or GRIB2 dataset with optional Dask chunking.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Weather dataset file not found at: {path}")

        default_chunks = chunks or {"time": 1, "latitude": 100, "longitude": 100}

        # Check file extension
        suffix = path.suffix.lower()
        if suffix in [".grib", ".grb2", ".grib2"]:
            logger.info("Loading GRIB2 file using cfgrib engine: %s", path)
            try:
                return xr.open_dataset(path, engine="cfgrib", chunks=default_chunks)
            except Exception as e:
                logger.error("Failed to load GRIB with cfgrib: %s. Ensure ecCodes is installed.", e)
                raise
        else:
            # Assume NetCDF
            logger.info("Loading NetCDF file using h5netcdf engine: %s", path)
            return xr.open_dataset(path, engine="h5netcdf", chunks=default_chunks)

    def get_or_create_benchmark_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Retrieves the standard benchmark NEPS-G forecast and ERA5 baseline datasets.
        If files do not exist on disk in DATA_DIR, generates and caches them.
        """
        forecast_path = self.data_dir / "neps_g_benchmark.nc"
        climatology_path = self.data_dir / "era5_climatology_benchmark.nc"

        if forecast_path.exists() and climatology_path.exists():
            ds_forecast = xr.open_dataset(forecast_path, engine="h5netcdf")
            ds_clim = xr.open_dataset(climatology_path, engine="h5netcdf")
        else:
            logger.info("Benchmark datasets not found on disk. Generating synthetic benchmark...")
            ds_forecast = generate_synthetic_neps_g(save_path=forecast_path)
            ds_clim = generate_synthetic_era5_climatology(save_path=climatology_path)

        return ds_forecast, ds_clim
