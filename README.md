# AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies
### Ministry of Earth Sciences (MoES) — Problem Statement 26078

Production-ready backend for Numerical Weather Prediction (NWP) ensemble ingestion, climatological baseline comparison, Extreme Forecast Index (EFI) anomaly detection, dynamic spatio-temporal bounding box generation, and live extreme weather disaster news feed integration.

---

## 🏗 System Architecture & Team Interfaces

```
                                [Raw Global Ensemble Data]
                     NCUM (12km deterministic) | NEPS-G (12km ensemble)
                     ERA5 / IMDAA (30-yr Reanalysis Climatological Baseline)
                                             │
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           YASHVARDHAN'S DATA & ANOMALY CORE                            │
  │  • Multidimensional Ingestion: xarray + Dask lazy chunking (GRIB2 / NetCDF4 / Zarr)   │
  │  • Ensemble Processing: Mean, spread, and vectorized quantiles along 'member' axis      │
  │  • EFI Anomaly Engine: Numerical integration of ECMWF EFI CDF vs ERA5 M-Climate       │
  │  • Spatio-Temporal Tracker: Contiguous cluster labeling, centroid & 4D Bounding Boxes  │
  │  • Geospatial Serializer: RFC 7946 compliant GeoJSON Polygons, LineStrings, and Points  │
  │  • Live Media Intelligence: Real-time disaster news integration (NewsAPI + GDELT)     │
  └───────────────────┬───────────────────────────────────────────────┬────────────────────┘
                      │ Clean Array Slices & 4D BBoxes                │ GeoJSON & News APIs
                      ▼                                               ▼
  ┌───────────────────────────────────────┐       ┌────────────────────────────────────────┐
  │         PALAK'S AI INFERENCE          │       │          TEAM LEAD'S FRONTEND          │
  │  • Stage 1: Spherical Icosahedral GNN │       │  • Interactive Leaflet / Mapbox Layers │
  │    (3-10 day anomaly trajectory)      │       │  • NDRF 5km Impact Zone Visualizer     │
  │  • Stage 2: Amplitude-Preserving      │       │  • Real-time Anomaly Timeline Graphs   │
  │    Diffusion Downscaling (12km -> 5km)│       │  • Correlated Live News Ticker         │
  └───────────────────────────────────────┘       └────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Prerequisites & Virtual Environment
Python 3.10+ is supported. A virtual environment has been configured.

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies (if setting up on a new node)
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
# Standard Launch (with hot-reload):
python run.py

# High-Performance Multi-Worker Launch (e.g. 4 parallel workers):
WORKERS=4 python run.py

# Server will launch on http://0.0.0.0:8000
# Interactive Swagger Documentation: http://127.0.0.1:8000/docs
# Interactive Redoc: http://127.0.0.1:8000/redoc
```

### 3. Run Automated Test Suite
```bash
PYTHONPATH=backend pytest backend/tests -v
# 46 tests covering data ingestion, preprocessing, EFI, spatial bboxes, GeoJSON, datasets, parallel workers, pest risk (Mod 14), and market intelligence (Mod 15)
```

---

## 📡 REST API Reference

### 1. Forecast Metadata & Spatial Bounds
* **Endpoint**: `GET /weather/forecast`
* **Query Parameters**:
  * `variable`: `wind_speed_10m` | `total_precipitation` | `temperature_2m` | `mean_sea_level_pressure`
  * `lead_hour`: (optional, e.g. `72`)
* **Response**:
```json
{
  "dataset": "NCMRWF NEPS-G 12km Global Ensemble",
  "variable": "wind_speed_10m",
  "units": "m/s",
  "lead_hours": [24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
  "ensemble_members": 10,
  "bounding_box": [65.0, 5.0, 98.0, 35.0],
  "resolution_deg": 0.5,
  "summary": { "min": 1.33, "max": 66.45, "mean": 8.39 }
}
```

---

### 2. Ensemble Member Distribution & Point Probe
* **Endpoint**: `GET /weather/ensemble`
* **Query Parameters**:
  * `variable`: `wind_speed_10m` (default)
  * `lat`, `lon`: Geographic probe coordinate (e.g. `lat=18.5&lon=85.2`)
  * `lead_hour`: Forecast lead hour (default `72`)
* **Response**:
```json
{
  "variable": "wind_speed_10m",
  "lead_hour": 72,
  "coordinates": { "lat": 18.5, "lon": 85.0 },
  "ensemble_mean": 38.45,
  "ensemble_spread": 4.12,
  "ensemble_median": 38.10,
  "p10": 33.2,
  "p90": 43.8,
  "p95": 45.6,
  "p99": 48.1,
  "member_values": [35.2, 42.1, 38.4, 40.0, 33.9, 44.5, 37.8, 36.5, 41.2, 34.9]
}
```

---

### 3. Extreme Forecast Index (EFI) Anomaly & 4D Bounding Boxes
* **Endpoint**: `GET /weather/anomaly`
* **Query Parameters**:
  * `variable`: `wind_speed_10m` | `total_precipitation`
  * `threshold`: EFI cutoff (default `0.65`, values $>0.8$ flag severe events)
  * `format`: `json` (default) or `geojson`

#### JSON Format (Used by Palak's GNN / Diffusion):
```json
{
  "hazard_evaluated": "tropical_cyclone",
  "variable": "wind_speed_10m",
  "efi_threshold": 0.65,
  "active_events_count": 1,
  "events": [
    {
      "event_id": "EVT-001",
      "hazard": "tropical_cyclone",
      "max_efi": 0.954,
      "severity": "extreme",
      "probability": 0.91,
      "bbox": [81.75, 5.0, 93.25, 23.5],
      "forecast_hours": [24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
      "trajectory": [
        { "lead_hour": 24, "centroid": [87.859, 9.907], "bbox": [82.75, 5.0, 93.25, 15.25], "peak_efi": 0.954 },
        { "lead_hour": 48, "centroid": [87.649, 11.381], "bbox": [82.25, 6.25, 92.75, 16.75], "peak_efi": 0.954 },
        { "lead_hour": 72, "centroid": [87.307, 12.684], "bbox": [81.75, 6.75, 92.25, 17.25], "peak_efi": 0.954 }
      ]
    }
  ]
}
```

#### GeoJSON Format (`?format=geojson` - Used directly by Frontend Mapbox/Leaflet):
Returns an RFC 7946 `FeatureCollection` with:
* Bounding Box Polygons
* Trajectory LineString
* Waypoint Points with lead hour and peak EFI properties.

---

### 4. Regional Subsets & Alerts
* **Endpoint**: `GET /weather/region/{id}`
* **Available IDs**: `bay_of_bengal`, `coastal_odisha`, `north_india`, `arabian_sea`, `western_ghats`
* **Response**:
```json
{
  "region_id": "coastal_odisha",
  "region_name": "Odisha Coastal Zone",
  "bbox": [83.0, 18.0, 87.5, 22.5],
  "lead_hour": 72,
  "variable": "wind_speed_10m",
  "regional_mean": 32.14,
  "regional_max": 54.82,
  "status": "alert"
}
```

---

### 5. Anomaly Evolution Timeline
* **Endpoint**: `GET /weather/timeline`
* **Query Parameters**: `variable` (default `wind_speed_10m`)
* **Response**:
```json
{
  "variable": "wind_speed_10m",
  "forecast_horizon_hours": 240,
  "timeline": [
    { "lead_hour": 24, "day": 1.0, "max_efi": 0.954, "peak_value": 48.2, "mean_value": 8.4, "max_ensemble_spread": 4.8, "threat_level": "extreme" },
    { "lead_hour": 72, "day": 3.0, "max_efi": 0.954, "peak_value": 64.5, "mean_value": 9.1, "max_ensemble_spread": 5.2, "threat_level": "extreme" }
  ]
}
```

---

### 6. Live Weather Disaster News Integration
* **Endpoint**: `GET /weather/news`
* **Query Parameters**:
  * `hazard`: `cyclone` | `heatwave` | `flood` | `cloudburst`
  * `region`: Target state/region (e.g. `Odisha`, `Rajasthan`, `India`)
  * `limit`: Number of articles (default `5`)
* **Response**:
```json
{
  "hazard": "cyclone",
  "region": "Odisha",
  "articles_count": 2,
  "active_provider": "GDELT Project (Free Live Feed)",
  "articles": [
    {
      "title": "IMD Issues Red Alert for Severe Cyclone in Odisha",
      "description": "National Disaster Response Force (NDRF) teams pre-deployed across vulnerable districts following extreme medium-range forecast models.",
      "source": "India Meteorological Department / MoES Bulletin",
      "url": "https://mausam.imd.gov.in",
      "published_at": "2026-10-01T12:00:00Z",
      "provider": "MoES Emergency Fallback Feed"
    }
  ]
}
```

* **News API Key Management**:
  * `GET /weather/news/status`: Checks if an external NewsAPI key is configured.
  * `POST /weather/news/key`: Update key at runtime without restarting the server:
    ```json
    { "api_key": "YOUR_NEWSAPI_KEY" }
    ```

---

## 🔑 News API Keys & Zero-Config Fallback

1. **Zero-Configuration Mode (Default)**:
   * The backend automatically queries **GDELT Project 2.0 API** and **UN OCHA ReliefWeb**.
   * **No API key is required.** Real-time global and Indian regional press coverage works out-of-the-box.
2. **NewsAPI.org Integration**:
   * If you have a NewsAPI key, set it in `.env`:
     ```bash
     NEWS_API_KEY=your_key_here
     ```
   * Or pass it dynamically via `POST /weather/news/key`.

---

### 7. Module 14: Pest & Disease Risk Engine
* **Endpoint**: `GET /weather/agri/pest-risk`
* **Query Parameters**:
  * `crop`: `rice` | `cotton` | `wheat`
  * `stage`: `vegetative` | `flowering` | `maturity`
  * `region`: Target region (e.g. `Odisha`, `Punjab`)
* **Response**:
```json
{
  "crop": "Rice (Paddy)",
  "region": "Odisha",
  "growth_stage": "Vegetative",
  "overall_pest_disease_risk": 0.82,
  "threat_level": "critical",
  "action_urgency": "Immediate preventive spray within 24-48 hours",
  "pathogens_evaluated": [
    {
      "pathogen": "Bacterial Leaf Blight (Xanthomonas oryzae)",
      "type": "bacterial",
      "risk_probability": 0.88,
      "severity": "critical",
      "contributing_factors": {
        "temperature_match": true,
        "humidity_exceeded": true,
        "wind_rain_vector_active": true,
        "stage_susceptible": true
      },
      "advisory": "Avoid nitrogen top-dressing; apply Streptocycline (0.01%) + Copper Oxychloride (0.25%)."
    }
  ]
}
```

---

### 8. Module 15: Mandi & Market Intelligence
* **Endpoint**: `GET /weather/agri/market-intelligence`
* **Query Parameters**:
  * `region`: `odisha` | `west_bengal` | `punjab` | `gujarat`
  * `crop`: `rice` | `wheat` | `cotton`
  * `hazard`: `cyclone` | `heatwave` | `flood`
* **Response**:
```json
{
  "mandi_id": "MND-OD-001",
  "mandi_name": "Bhubaneswar APMC Agricultural Market",
  "commodity": {
    "crop_name": "Rice / Paddy (Common & Grade A)",
    "modal_price_inr_quintal": 2260.0,
    "msp_benchmark_inr": 2183.0,
    "premium_over_msp_pct": 3.53
  },
  "weather_shock_forecast": {
    "hazard_driving_shock": "Cyclone",
    "projected_arrival_reduction_72h_pct": 55.2,
    "projected_price_surge_pct": 16.6,
    "volatility_status": "Severe Volatility",
    "supply_corridor_risk": "Inundation & Transport Blockage Expected",
    "impacted_transit_routes": ["NH-16 (Coastal Highway)", "Bhubaneswar-Puri Link"]
  },
  "fpo_and_procurement_advisory": "Expedite pre-landfall procurement; redirect truck freight away from coastal routes."
}
```

---

## 📦 Directory Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application & lifespan management
│   │   ├── config.py                   # App configuration & geographical basins
│   │   ├── api/
│   │   │   ├── weather_routes.py       # Forecast, ensemble, anomaly, and timeline endpoints
│   │   │   ├── news_routes.py          # Live disaster news endpoints
│   │   │   └── agri_routes.py          # Modules 14 & 15: Pest risk & Mandi market intelligence
│   │   ├── agri/
│   │   │   ├── pest_risk.py            # Module 14: Pathogen & insect pest risk engine
│   │   │   └── market_intelligence.py  # Module 15: Mandi pricing & weather shock model
│   │   ├── data_ingestion/
│   │   │   ├── loaders.py              # GRIB2 & NetCDF4 Dask loader
│   │   │   └── synthetic_data.py       # 4D NEPS-G & 30-year ERA5 climatology generator
│   │   ├── preprocessing/
│   │   │   ├── ensemble.py             # Vectorized ensemble mean, spread, quantiles
│   │   │   ├── normalizer.py           # Z-score and physical unit transforms
│   │   │   └── pipeline.py             # Reusable end-to-end preprocessing pipeline
│   │   ├── anomaly/
│   │   │   ├── efi.py                  # ECMWF numerical integral EFI calculation
│   │   │   └── bbox_tracker.py         # Morphological labeling & trajectory tracking
│   │   ├── spatial/
│   │   │   ├── cropper.py              # Spatial BBox & temporal window slicer
│   │   │   └── geojson.py              # GeoJSON RFC 7946 feature collections
│   │   ├── cache/
│   │   │   └── zarr_cache.py           # Chunked Zarr & memory cache manager
│   │   └── news/
│   │       └── news_service.py         # NewsAPI + GDELT + ReliefWeb live service
│   ├── data/
│   │   ├── raw/                        # NetCDF / GRIB2 weather stores
│   │   └── cache/                      # Zarr stores and computed matrices
│   └── tests/                          # Complete automated test suite (34 tests)
├── run.py                              # Clean one-click application launcher
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```
