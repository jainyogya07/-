"""
GeoJSON Serializers for Weather Anomalies, Bounding Boxes, and Trajectories.
Outputs RFC 7946 compliant GeoJSON FeatureCollections for frontend mapping (Leaflet/Mapbox/OpenLayers).
"""

from typing import Any


def bbox_to_geojson_polygon(bbox: list[float], properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Converts [min_lon, min_lat, max_lon, max_lat] to GeoJSON Polygon Feature.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    coordinates = [[
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]]
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates,
        },
        "properties": properties or {},
    }


def trajectory_to_geojson_linestring(trajectory: list[dict[str, Any]], properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Converts a sequence of trajectory points into a GeoJSON LineString Feature.
    """
    coordinates = [point["centroid"] for point in trajectory]
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": properties or {},
    }


def event_to_geojson_features(event: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Translates a 4D anomaly event into a set of GeoJSON features:
    1. Overall Bounding Box Polygon
    2. Dynamic Trajectory LineString
    3. Step Centroid Points with individual timestamps and EFI intensities
    """
    features = []

    # 1. Bounding box feature
    bbox_feat = bbox_to_geojson_polygon(
        event["bbox"],
        properties={
            "feature_type": "event_bounding_box",
            "event_id": event["event_id"],
            "hazard": event["hazard"],
            "severity": event["severity"],
            "probability": event["probability"],
            "max_efi": event["max_efi"],
        },
    )
    features.append(bbox_feat)

    # 2. Trajectory line feature
    if "trajectory" in event and len(event["trajectory"]) > 1:
        line_feat = trajectory_to_geojson_linestring(
            event["trajectory"],
            properties={
                "feature_type": "trajectory_path",
                "event_id": event["event_id"],
                "hazard": event["hazard"],
                "forecast_hours": event.get("forecast_hours", []),
            },
        )
        features.append(line_feat)

    # 3. Trajectory waypoint points
    for pt in event.get("trajectory", []):
        pt_feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": pt["centroid"],
            },
            "properties": {
                "feature_type": "trajectory_waypoint",
                "event_id": event["event_id"],
                "lead_hour": pt["lead_hour"],
                "peak_efi": pt.get("peak_efi"),
                "bbox": pt.get("bbox"),
            },
        }
        features.append(pt_feat)

    return features


def events_to_geojson_feature_collection(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Converts a list of weather anomaly events into a single GeoJSON FeatureCollection.
    """
    all_features = []
    for ev in events:
        all_features.extend(event_to_geojson_features(ev))

    return {
        "type": "FeatureCollection",
        "features": all_features,
    }
