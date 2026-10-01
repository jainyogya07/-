"""
Ensemble Dimension Handling & Statistical Operations.
Processes multi-member ensemble predictions (NEPS-G) along the ensemble axis ('member').
"""

import numpy as np
import xarray as xr


def get_ensemble_dim_name(ds: xr.Dataset | xr.DataArray) -> str:
    """Finds the ensemble dimension name in common NWP conventions."""
    for dim_name in ["member", "realization", "number", "ensemble"]:
        if dim_name in ds.dims:
            return dim_name
    raise ValueError(f"No ensemble dimension found in dimensions: {list(ds.dims)}")


def compute_ensemble_statistics(
    ds: xr.Dataset,
    var_name: str,
    quantiles: list[float] | None = None,
) -> xr.Dataset:
    """
    Computes ensemble mean, spread (std), median, and specified quantiles (e.g. 0.90, 0.95).
    """
    if quantiles is None:
        quantiles = [0.10, 0.50, 0.90, 0.95, 0.99]

    ens_dim = get_ensemble_dim_name(ds)
    da = ds[var_name]

    mean_val = da.mean(dim=ens_dim)
    std_val = da.std(dim=ens_dim)
    median_val = da.median(dim=ens_dim)

    stat_vars = {
        f"{var_name}_mean": mean_val,
        f"{var_name}_spread": std_val,
        f"{var_name}_median": median_val,
    }

    if quantiles:
        q_da = da.quantile(quantiles, dim=ens_dim)
        for idx, q in enumerate(quantiles):
            q_label = int(q * 100)
            stat_vars[f"{var_name}_p{q_label}"] = q_da.isel(quantile=idx).drop_vars("quantile", errors="ignore")

    return xr.Dataset(stat_vars)


def compute_probability_of_exceedance(
    ds: xr.Dataset,
    var_name: str,
    threshold: float,
) -> xr.DataArray:
    """
    Computes the fraction of ensemble members that exceed a critical physical threshold
    (e.g., wind > 30 m/s or rainfall > 100 mm). Returns values between 0.0 and 1.0.
    """
    ens_dim = get_ensemble_dim_name(ds)
    da = ds[var_name]
    exceeded = (da >= threshold).astype(np.float32)
    return exceeded.mean(dim=ens_dim)
