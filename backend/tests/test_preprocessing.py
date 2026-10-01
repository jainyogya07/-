"""Tests for ensemble operations, normalization, and preprocessing pipeline."""

import pytest
import numpy as np
from app.data_ingestion.synthetic_data import generate_synthetic_neps_g
from app.preprocessing.ensemble import (
    compute_ensemble_statistics,
    compute_probability_of_exceedance,
    get_ensemble_dim_name,
)
from app.preprocessing.normalizer import MeteorologicalNormalizer
from app.preprocessing.pipeline import WeatherPreprocessingPipeline


@pytest.fixture(scope="module")
def sample_dataset():
    return generate_synthetic_neps_g(lead_hours=[24, 48], n_members=4)


def test_ensemble_dim_detection(sample_dataset):
    dim_name = get_ensemble_dim_name(sample_dataset)
    assert dim_name == "member"


def test_compute_ensemble_statistics(sample_dataset):
    stats = compute_ensemble_statistics(sample_dataset, "wind_speed_10m")
    assert "wind_speed_10m_mean" in stats
    assert "wind_speed_10m_spread" in stats
    assert "wind_speed_10m_p95" in stats
    assert "member" not in stats.dims


def test_compute_probability_of_exceedance(sample_dataset):
    prob = compute_probability_of_exceedance(sample_dataset, "wind_speed_10m", threshold=10.0)
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0


def test_normalizer(sample_dataset):
    normalizer = MeteorologicalNormalizer()
    da = sample_dataset["wind_speed_10m"].isel(time=0, member=0)

    z = normalizer.z_score_normalize(da, "wind_speed_10m")
    assert isinstance(z.values, np.ndarray)

    mm = normalizer.min_max_scale(da, "wind_speed_10m")
    assert float(mm.min()) >= -0.5
    assert float(mm.max()) <= 2.0


def test_preprocessing_pipeline(sample_dataset):
    pipeline = WeatherPreprocessingPipeline()
    result = pipeline.process(sample_dataset)

    assert "wind_speed_10m_mean" in result.ensemble_stats
    assert "wind_speed_10m" in result.exceedance_probabilities
    assert "wind_speed_10m_zscore" in result.normalized_forecast
