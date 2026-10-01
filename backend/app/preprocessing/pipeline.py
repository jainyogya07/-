"""
Reusable Weather Preprocessing Pipeline.
Combines ensemble metrics computation, climatology alignment, normalization,
and anomaly extraction into a single, production-ready pipeline.
"""

from dataclasses import dataclass
import xarray as xr
from app.preprocessing.ensemble import compute_ensemble_statistics, compute_probability_of_exceedance
from app.preprocessing.normalizer import MeteorologicalNormalizer


@dataclass
class ProcessedWeatherResult:
    ensemble_stats: xr.Dataset
    normalized_forecast: xr.Dataset
    exceedance_probabilities: dict[str, xr.DataArray]


class WeatherPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for multivariable 4D NWP ensemble datasets.
    """

    def __init__(self, normalizer: MeteorologicalNormalizer | None = None):
        self.normalizer = normalizer or MeteorologicalNormalizer()

    def process(
        self,
        forecast_ds: xr.Dataset,
        variables: list[str] | None = None,
        exceedance_thresholds: dict[str, float] | None = None,
    ) -> ProcessedWeatherResult:
        """
        Runs the full preprocessing pipeline:
        1. Computes ensemble statistics across realization members.
        2. Calculates probability of exceedance for severe thresholds.
        3. Normalizes variables for ML tensor feeding.
        """
        if variables is None:
            variables = ["wind_speed_10m", "total_precipitation", "temperature_2m", "mean_sea_level_pressure"]

        if exceedance_thresholds is None:
            exceedance_thresholds = {
                "wind_speed_10m": 25.0,  # ~90 km/h gale force
                "total_precipitation": 70.0,  # Heavy rainfall (mm/24h)
            }

        # 1. Ensemble statistics
        stat_datasets = []
        for var in variables:
            if var in forecast_ds:
                stat_ds = compute_ensemble_statistics(forecast_ds, var)
                stat_datasets.append(stat_ds)
        combined_stats = xr.merge(stat_datasets) if stat_datasets else xr.Dataset()

        # 2. Probability of exceedance
        probs = {}
        for var, thresh in exceedance_thresholds.items():
            if var in forecast_ds:
                probs[var] = compute_probability_of_exceedance(forecast_ds, var, thresh)

        # 3. Normalization (using ensemble mean)
        normalized_vars = {}
        for var in variables:
            mean_var_name = f"{var}_mean"
            if mean_var_name in combined_stats:
                da = combined_stats[mean_var_name]
                normalized_vars[f"{var}_zscore"] = self.normalizer.z_score_normalize(da, var)
                normalized_vars[f"{var}_minmax"] = self.normalizer.min_max_scale(da, var)
        normalized_ds = xr.Dataset(normalized_vars)

        return ProcessedWeatherResult(
            ensemble_stats=combined_stats,
            normalized_forecast=normalized_ds,
            exceedance_probabilities=probs,
        )
