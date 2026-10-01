"""
Extreme Forecast Index (EFI) & Climatological Anomaly Computation.
Computes EFI by evaluating the cumulative distribution function (CDF) of the ensemble forecast
against the long-term ERA5/IMDAA climatological distribution (M-Climate).
"""

import numpy as np
import xarray as xr


def compute_efi_integral(
    ensemble_values: np.ndarray,
    clim_mean: np.ndarray,
    clim_std: np.ndarray,
    p_steps: int = 20,
) -> np.ndarray:
    """
    Numerically integrates the ECMWF Extreme Forecast Index (EFI) equation across probability quantiles:
        EFI = (2 / pi) * Integral_0^1 [ (p - F_f(Q_c(p))) / sqrt(p * (1 - p)) ] dp

    Args:
        ensemble_values: Shape (n_members, n_lats, n_lons)
        clim_mean: Shape (n_lats, n_lons)
        clim_std: Shape (n_lats, n_lons)
        p_steps: Number of integration steps between (0, 1)

    Returns:
        EFI array of shape (n_lats, n_lons), normalized between -1.0 and 1.0
    """
    n_members, n_lats, n_lons = ensemble_values.shape
    p_vals = np.linspace(0.02, 0.98, p_steps)
    dp = p_vals[1] - p_vals[0]

    integral_accum = np.zeros((n_lats, n_lons), dtype=np.float32)

    # Sort ensemble values along member dimension for empirical CDF
    sorted_ens = np.sort(ensemble_values, axis=0)

    for p in p_vals:
        weight = 1.0 / np.sqrt(p * (1.0 - p))
        # Quantile of climatology assuming normal distribution
        # For precipitation/wind, positive tail is key
        from scipy.special import ndtri
        z = ndtri(p)
        q_c = clim_mean + z * (clim_std + 1e-6)

        # F_f(q_c): Empirical proportion of ensemble members <= q_c
        # Broadcast comparison: (n_members, n_lats, n_lons) <= (n_lats, n_lons)
        count_less = np.sum(sorted_ens <= q_c[np.newaxis, :, :], axis=0)
        f_f = count_less / float(n_members)

        # Integrand: (p - F_f(q_c))
        integrand = (p - f_f) * weight
        integral_accum += integrand * dp

    efi = (2.0 / np.pi) * integral_accum
    return np.clip(efi, -1.0, 1.0)


def compute_dataset_efi(
    forecast_ds: xr.Dataset,
    climatology_ds: xr.Dataset,
    var_name: str,
) -> xr.DataArray:
    """
    Computes EFI across all forecast lead times for a given variable.
    """
    # Resolve climatology variable names with alias fallbacks (e.g. wind_speed_10m -> wind_speed)
    def resolve_clim_var(stat: str):
        candidates = [
            f"{var_name}_{stat}",
            f"{var_name.replace('_10m', '').replace('_2m', '')}_{stat}",
        ]
        for c in candidates:
            if c in climatology_ds:
                return climatology_ds[c].values
        # Fallback search
        base = var_name.replace("_10m", "").replace("_2m", "")
        for k in climatology_ds.data_vars:
            if base in k and stat in k:
                return climatology_ds[k].values
        raise KeyError(f"Could not find climatology field for {var_name} ({stat}). Available: {list(climatology_ds.data_vars)}")

    clim_mean = resolve_clim_var("mean")
    clim_std = resolve_clim_var("std")

    times = forecast_ds["time"].values
    lats = forecast_ds["latitude"].values
    lons = forecast_ds["longitude"].values

    efi_time_series = []

    for t_idx in range(len(times)):
        # Extract ensemble slice for this time step: shape (n_members, n_lats, n_lons)
        ens_slice = forecast_ds[var_name].isel(time=t_idx).values
        efi_grid = compute_efi_integral(ens_slice, clim_mean, clim_std)
        efi_time_series.append(efi_grid)

    efi_array = np.stack(efi_time_series, axis=0)

    return xr.DataArray(
        efi_array,
        coords={
            "time": times,
            "lead_hour": ("time", forecast_ds["lead_hour"].values if "lead_hour" in forecast_ds else list(range(len(times)))),
            "latitude": lats,
            "longitude": lons,
        },
        dims=["time", "latitude", "longitude"],
        attrs={
            "long_name": f"Extreme Forecast Index ({var_name})",
            "units": "dimensionless (-1 to 1)",
            "description": "EFI measure against ERA5 30-year climatology baseline",
        },
    )
