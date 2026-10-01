"""
Spatial & Temporal Slicing / Cropping Utilities.
Slices multidimensional xarray Datasets along geographic bounding boxes and forecast time horizons.
"""

from typing import Sequence
import xarray as xr


def crop_spatial_bbox(
    ds: xr.Dataset | xr.DataArray,
    bbox: Sequence[float],  # [min_lon, min_lat, max_lon, max_lat]
) -> xr.Dataset | xr.DataArray:
    """
    Crops an xarray object to a geographic bounding box [min_lon, min_lat, max_lon, max_lat].
    Automatically handles descending vs ascending latitude ordering.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Identify lat/lon coordinate names
    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"

    lats = ds.coords[lat_key].values
    lons = ds.coords[lon_key].values

    # Determine latitude slice direction
    if lats[0] > lats[-1]:  # Descending
        lat_slice = slice(max_lat, min_lat)
    else:  # Ascending
        lat_slice = slice(min_lat, max_lat)

    lon_slice = slice(min_lon, max_lon)

    return ds.sel({lat_key: lat_slice, lon_key: lon_slice})


def crop_temporal_window(
    ds: xr.Dataset | xr.DataArray,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Filters dataset to specific lead hour forecast window (e.g. 72h to 240h).
    """
    if "lead_hour" in ds.coords:
        cond = True
        if start_hour is not None:
            cond = cond & (ds["lead_hour"] >= start_hour)
        if end_hour is not None:
            cond = cond & (ds["lead_hour"] <= end_hour)
        return ds.where(cond, drop=True)

    return ds
