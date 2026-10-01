"""
Anomaly Bounding Box Extractor & Trajectory Tracker.
Detects spatially contiguous extreme anomaly clusters using morphological labeling,
computes 4D spatio-temporal bounding boxes, and tracks centroids across forecast horizons.
"""

from typing import Any
import numpy as np
from scipy.ndimage import label
import xarray as xr


def classify_severity(efi_score: float, probability: float) -> str:
    """Classifies weather hazard severity based on EFI and ensemble probability."""
    if efi_score >= 0.85 or probability >= 0.80:
        return "extreme"
    elif efi_score >= 0.70 or probability >= 0.60:
        return "severe"
    elif efi_score >= 0.50 or probability >= 0.40:
        return "moderate"
    else:
        return "low"


def extract_spatial_features_at_step(
    efi_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    threshold: float = 0.60,
    min_cells: int = 4,
) -> list[dict[str, Any]]:
    """
    Finds contiguous anomaly regions exceeding the threshold in a 2D spatial grid.
    """
    mask = efi_grid >= threshold
    labeled_array, num_features = label(mask)
    features = []

    for feature_id in range(1, num_features + 1):
        y_idxs, x_idxs = np.where(labeled_array == feature_id)
        if len(y_idxs) < min_cells:
            continue

        selected_lats = lats[y_idxs]
        selected_lons = lons[x_idxs]

        min_lat, max_lat = float(selected_lats.min()), float(selected_lats.max())
        min_lon, max_lon = float(selected_lons.min()), float(selected_lons.max())

        # Buffer bbox slightly (0.25 deg) for contextual downscaling border
        buffer = 0.25
        min_lat = max(float(lats.min()), min_lat - buffer)
        max_lat = min(float(lats.max()), max_lat + buffer)
        min_lon = max(float(lons.min()), min_lon - buffer)
        max_lon = min(float(lons.max()), max_lon + buffer)

        # Center of mass / peak
        cluster_efi = efi_grid[y_idxs, x_idxs]
        peak_efi = float(cluster_efi.max())
        mean_efi = float(cluster_efi.mean())

        # Weighted centroid by EFI intensity
        weights = cluster_efi / (cluster_efi.sum() + 1e-6)
        c_lat = float(np.sum(selected_lats * weights))
        c_lon = float(np.sum(selected_lons * weights))

        # Approximate area in km2 (1 deg lat ~ 111 km)
        d_lat_km = (max_lat - min_lat) * 111.0
        d_lon_km = (max_lon - min_lon) * 111.0 * np.cos(np.radians(c_lat))
        area_km2 = float(max(10.0, d_lat_km * d_lon_km))

        features.append({
            "bbox": [round(min_lon, 3), round(min_lat, 3), round(max_lon, 3), round(max_lat, 3)],
            "centroid": [round(c_lon, 3), round(c_lat, 3)],
            "peak_efi": round(peak_efi, 3),
            "mean_efi": round(mean_efi, 3),
            "area_km2": round(area_km2, 1),
            "grid_cells": int(len(y_idxs)),
        })

    # Sort largest/strongest features first
    features.sort(key=lambda x: x["peak_efi"], reverse=True)
    return features


def track_anomaly_events(
    efi_da: xr.DataArray,
    hazard_type: str = "severe_weather",
    threshold: float = 0.60,
) -> list[dict[str, Any]]:
    """
    Tracks dynamic anomalies across the full temporal forecast sequence (3 to 10 days).
    Produces consolidated 4D event records ready for Palak's GNN/Diffusion pipeline.
    """
    times = efi_da["time"].values
    lead_hours = efi_da["lead_hour"].values if "lead_hour" in efi_da.coords else [int(i * 24) for i in range(len(times))]
    lats = efi_da["latitude"].values
    lons = efi_da["longitude"].values

    step_features = []
    for t_idx in range(len(times)):
        grid = efi_da.isel(time=t_idx).values
        feats = extract_spatial_features_at_step(grid, lats, lons, threshold=threshold)
        step_features.append(feats)

    # Simple nearest-centroid temporal tracker to assemble unified events
    active_events = []
    event_counter = 1

    for t_idx, feats in enumerate(step_features):
        lead_h = int(lead_hours[t_idx])
        for feat in feats:
            c_lon, c_lat = feat["centroid"]
            # Look for existing track matching within 4.0 degrees distance (~400 km in 24h)
            matched_event = None
            for ev in active_events:
                last_pt = ev["trajectory"][-1]
                dist = np.hypot(last_pt["centroid"][0] - c_lon, last_pt["centroid"][1] - c_lat)
                if dist < 4.5:
                    matched_event = ev
                    break

            if matched_event:
                matched_event["trajectory"].append({
                    "lead_hour": lead_h,
                    "centroid": [c_lon, c_lat],
                    "bbox": feat["bbox"],
                    "peak_efi": feat["peak_efi"],
                })
                matched_event["forecast_hours"].append(lead_h)
                matched_event["max_efi"] = max(matched_event["max_efi"], feat["peak_efi"])
                # Expand global event bbox
                gb = matched_event["global_bbox"]
                matched_event["global_bbox"] = [
                    min(gb[0], feat["bbox"][0]),
                    min(gb[1], feat["bbox"][1]),
                    max(gb[2], feat["bbox"][2]),
                    max(gb[3], feat["bbox"][3]),
                ]
            else:
                new_event = {
                    "event_id": f"EVT-{event_counter:03d}",
                    "hazard": hazard_type,
                    "max_efi": feat["peak_efi"],
                    "global_bbox": feat["bbox"],
                    "forecast_hours": [lead_h],
                    "trajectory": [{
                        "lead_hour": lead_h,
                        "centroid": [c_lon, c_lat],
                        "bbox": feat["bbox"],
                        "peak_efi": feat["peak_efi"],
                    }],
                }
                active_events.append(new_event)
                event_counter += 1

    # Enrich events with severity and probability
    final_events = []
    for ev in active_events:
        # Require event to persist for at least 2 forecast steps to filter transient noise
        if len(ev["trajectory"]) >= 2 or len(step_features) <= 2:
            prob = float(min(0.98, max(0.40, ev["max_efi"] * 0.95)))
            ev["probability"] = round(prob, 2)
            ev["severity"] = classify_severity(ev["max_efi"], prob)
            ev["bbox"] = ev.pop("global_bbox")
            final_events.append(ev)

    return final_events
