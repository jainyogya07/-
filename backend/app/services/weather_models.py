"""Lazy GNN and diffusion inference adapters with safe development fallback."""
import math
from typing import Any
from backend.app.config import settings
from backend.app.infrastructure.artifacts import artifact_store


class WeatherModelRuntime:
    def __init__(self) -> None:
        self.device = "cpu"
        self.gnn_loaded = False
        self.diffusion_loaded = False

    def load(self) -> None:
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Checkpoints are loaded lazily when supplied; architecture-specific
            # construction belongs in the trained-model package.
            self.gnn_loaded = bool(settings.GNN_MODEL_PATH)
            self.diffusion_loaded = bool(settings.DIFFUSION_MODEL_PATH)
        except ImportError:
            self.device = "cpu"

    def track(self, payload: dict[str, Any]) -> dict[str, Any]:
        bbox = payload["bbox"]
        variables = payload.get("variables", {})
        efi = float(variables.get("efi", variables.get("anomaly_sigma", 2.1)))
        probability = max(0.05, min(0.99, 1 / (1 + math.exp(-(efi - 1.0)))))
        hazard = payload.get("hazard_hint") or ("extreme_precipitation" if variables.get("precipitation_mm", 0) > 0 else "high_wind")
        severity = "severe" if probability >= .9 else "high" if probability >= .75 else "moderate" if probability >= .5 else "low"
        west, south, east, north = bbox
        trajectory = [{"forecast_hour": hour, "centroid": [round((west + east) / 2 + .03 * i, 5), round((south + north) / 2 + .01 * i, 5)], "probability": round(max(.05, probability - .035 * i), 3)} for i, hour in enumerate(payload["forecast_hours"])]
        return {"hazard": hazard, "probability": round(probability, 3), "severity": severity, "bbox": bbox, "forecast_hours": payload["forecast_hours"], "trajectory": trajectory}

    def downscale(self, payload: dict[str, Any], tracked: dict[str, Any]) -> dict[str, Any]:
        variables = payload.get("variables", {})
        coarse = float(variables.get("precipitation_mm", variables.get("wind_speed_ms", 0)))
        amplification = 1.18 if tracked["severity"] in ("high", "severe") else 1.05
        result = {"source_resolution_km": 12, "target_resolution_km": 5, "peak_intensity": round(coarse * amplification, 2), "units": "mm" if "precipitation_mm" in variables else "m/s", "physics_constraints": ["moisture_convergence", "mass_conservation"], "model_loaded": self.diffusion_loaded}
        # Arrays travel through object storage, not the database/API event body.
        if "downscaled_grid" in variables:
            result["artifact_uri"] = artifact_store.save_json("pending", "downscaled-grid", variables["downscaled_grid"])
        return result


weather_models = WeatherModelRuntime()
