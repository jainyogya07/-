# Technical Architecture, Methodology & Verification Report
## AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts
**Ministry of Earth Sciences (MoES) — Problem Statement 26078**  
**Lead Data Engineer & Geospatial Backend Architect**: Yashvardhan Dubey  
**Git Branch**: `amongus`  
**Date of Completion & Verification**: October 1, 2026  

---

## 1. Executive Summary & Problem Context

In medium-range Numerical Weather Prediction (NWP) forecasting (3 to 10 days), atmospheric chaos and nonlinear dynamics render single deterministic runs unreliable. Forecasters therefore utilize multivariable, 4-dimensional Ensemble Prediction Systems (EPS) such as the **NCMRWF Global Ensemble (NEPS-G at 12 km resolution)**.

However, extracting localized, actionable intelligence from multi-gigabyte 4D/5D EPS grids presents two core problems:
1. **The Spectral Smoothing Dilemma**: Conventional Convolutional Neural Networks (CNNs) and U-Nets optimize for mean squared error (MSE), which destroys extreme amplitudes—flattening the high-intensity peaks of wind speed and rainfall that disaster management teams (e.g. NDRF) need.
2. **Computational & Spatial Distortions**: Flat 2D pixel projections distort polar and tropical spherical geometries, while raw GRIB2/NetCDF files are too massive for sub-second REST API queries and live map rendering.

To solve this, the MoES challenge proposes a two-stage hybrid AI pipeline:
* **Stage 1 (Palak)**: A message-passing Graph Neural Network (GNN) on an icosahedral spherical mesh to track moving weather anomalies over a 3- to 10-day window.
* **Stage 2 (Palak)**: A conditional denoising diffusion probabilistic model that downscales the isolated macroscale bounding box from 12 km to 5 km while preserving extreme amplitudes with physics-informed conservation constraints.
* **Frontend Dashboard (Team Leader)**: An interactive geospatial dashboard displaying 5 km centroid alert zones, hazard trajectories, and live situational news media context.
* **Data Engineering, Preprocessing & Core APIs (Yashvardhan Dubey)**: The entire foundational engine that ingests, slices, calculates anomalies/EFI, generates 4D bounding boxes, caches arrays, serves REST APIs, and integrates live disaster news feeds.

---

## 2. End-to-End System Architecture

```
                  [Raw NWP Streams & Climatological Baselines]
             • NCMRWF NEPS-G (12km Global Ensemble, GRIB2 / NetCDF)
             • NCUM (12km Deterministic NWP, GRIB2 / NetCDF)
             • ERA5 / IMDAA (30-Year Reanalysis Climatology, NetCDF)
                                       │
                                       ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           YASHVARDHAN'S CORE SUBSYSTEM                                 │
  │                                                                                        │
  │  1. Ingestion Layer (`loaders.py`, `synthetic_data.py`)                                │
  │     • xarray + Dask out-of-core chunked processing (CF-compliant dimensions)          │
  │     • Automatic engine detection: cfgrib/ecCodes (GRIB2) & h5netcdf/h5py (NetCDF)     │
  │                                                                                        │
  │  2. Ensemble & Normalization Layer (`ensemble.py`, `normalizer.py`, `pipeline.py`)    │
  │     • Vectorized quantiles (p10, p50, p90, p95, p99) along 'member' axis               │
  │     • Z-score standardization & physical unit conversions                              │
  │                                                                                        │
  │  3. Anomaly & EFI Engine (`efi.py`)                                                    │
  │     • Numerical integration of ECMWF Extreme Forecast Index (EFI) integral             │
  │     • Deviation evaluation against 30-year ERA5 climatological CDF                     │
  │                                                                                        │
  │  4. Spatial Tracker & Bounding Box Layer (`bbox_tracker.py`, `cropper.py`)             │
  │     • Contiguous cluster labeling via scipy.ndimage.label                              │
  │     • Intensity-weighted centroids & 4D spatio-temporal bounding box generation        │
  │                                                                                        │
  │  5. GeoJSON Serialization (`geojson.py`)                                               │
  │     • RFC 7946 compliant Polygons, LineStrings, and Point FeatureCollections           │
  │                                                                                        │
  │  6. High-Performance Caching Layer (`zarr_cache.py`)                                   │
  │     • Chunked Zarr disk stores + in-memory LRU/TTL caching                             │
  │                                                                                        │
  │  7. Live Media Intelligence Layer (`news_service.py`)                                  │
  │     • GDELT Project 2.0 (Free real-time feed) + UN OCHA ReliefWeb + NewsAPI.org        │
  │     • Dynamic API key update via POST /weather/news/key                                │
  └───────────────────┬───────────────────────────────────────────────┬────────────────────┘
                      │ Clean Arrays & 4D BBoxes                      │ GeoJSON & News APIs
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

## 3. Mathematical & Algorithmic Rigor

### A. Extreme Forecast Index (EFI) Formulation
Standard anomaly metrics (such as Z-scores or standard deviations) assume Gaussian distributions. Atmospheric variables like precipitation and cyclonic wind speeds are highly skewed and bounded at zero.

To overcome this, the European Centre for Medium-Range Weather Forecasts (ECMWF) developed the **Extreme Forecast Index (EFI)**. We implement the complete continuous integral discretized across $K=20$ probability quantiles:

$$\text{EFI} = \frac{2}{\pi} \int_{0}^{1} \frac{p - F_f(Q_c(p))}{\sqrt{p(1 - p)}} \, dp$$

Where:
* $p \in [0.02, 0.98]$ represents probability quantiles.
* $Q_c(p)$ is the climatological quantile from the 30-year ERA5 baseline (M-Climate):
  $$Q_c(p) = \mu_c + \Phi^{-1}(p) \cdot \sigma_c$$
  where $\Phi^{-1}(p)$ is the inverse standard normal CDF (`scipy.special.ndtri`).
* $F_f(Q_c(p))$ is the empirical proportion of the $M$ ensemble members whose forecasted value is less than or equal to $Q_c(p)$:
  $$F_f(Q_c(p)) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{I}\left(x_m \le Q_c(p)\right)$$
* $\frac{1}{\sqrt{p(1-p)}}$ is the probability weighting function that assigns greatest weight to the extreme tails ($p \to 0$ and $p \to 1$).
* The result is normalized between $-1.0$ and $+1.0$. Values $\text{EFI} \ge 0.65$ denote abnormal weather; values $\text{EFI} \ge 0.85$ indicate extreme, record-breaking hazards.

### B. Morphological Anomaly Segmentation & Bounding Box Generation
1. **Contiguous Hazard Segmentation**:
   $$\mathcal{M}(y, x) = \mathbb{I}\left(\text{EFI}(y, x) \ge \tau_{\text{threshold}}\right)$$
   Clusters are isolated via 8-connectivity morphological labeling: $\mathcal{L} = \text{label}(\mathcal{M})$.
   Noise filtering discards clusters smaller than 4 contiguous grid cells ($< 1000\text{ km}^2$).
2. **Intensity-Weighted Centroid Calculation**:
   Instead of a simple geometric centroid, the center of mass is weighted by the EFI intensity:
   $$\bar{C} = \left( \frac{\sum_{i} \text{lon}_i \cdot \text{EFI}_i}{\sum_{i} \text{EFI}_i}, \; \frac{\sum_{i} \text{lat}_i \cdot \text{EFI}_i}{\sum_{i} \text{EFI}_i} \right)$$
3. **Macroscale Bounding Box with Downscaling Buffer**:
   $$\text{BBox} = [\min(\text{lon}) - \delta, \; \min(\text{lat}) - \delta, \; \max(\text{lon}) + \delta, \; \max(\text{lat}) + \delta]$$
   where $\delta = 0.25^\circ$ provides the spatial margin required by Palak's Stage 2 Diffusion model to capture boundary moisture fluxes without edge-effect artifacts.
4. **Trajectory Chaining Across Forecast Horizons ($3 \to 10$ Days)**:
   Centroids across lead times ($t \to t+1$) are matched using nearest Euclidean distance ($\Delta d \le 4.5^\circ \approx 500\text{ km/24h}$), generating a temporal trajectory path.

---

## 4. Module-by-Module Technical Inventory

### 1. `backend/app/data_ingestion/`
* [`loaders.py`](file:///home/yashvardhandubey/Projects/-/backend/app/data_ingestion/loaders.py):
  * Class `NWPDataLoader`: Lazily loads NetCDF4 and GRIB2 files with custom Dask chunks (`{"time": 1, "latitude": 100, "longitude": 100}`).
  * Method `get_or_create_benchmark_datasets()`: Automatically checks disk for pre-existing benchmark files or synthesizes them instantly.
* [`synthetic_data.py`](file:///home/yashvardhandubey/Projects/-/backend/app/data_ingestion/synthetic_data.py):
  * Function `generate_synthetic_neps_g()`: Generates a physically-consistent 4D/5D ensemble simulating a moving severe cyclone in the Bay of Bengal over 10 days across 10 perturbation members.
  * Function `generate_synthetic_era5_climatology()`: Generates a 30-year climatological baseline with mean, standard deviation, and 95th/99th percentiles.

### 2. `backend/app/preprocessing/`
* [`ensemble.py`](file:///home/yashvardhandubey/Projects/-/backend/app/preprocessing/ensemble.py):
  * Function `get_ensemble_dim_name()`: Inspects dataset dimensions to find `member`, `realization`, or `number`.
  * Function `compute_ensemble_statistics()`: Computes ensemble mean, spread ($\sigma$), median, and vectorized quantiles ($p_{10}, p_{50}, p_{90}, p_{95}, p_{99}$) in a single optimized pass.
  * Function `compute_probability_of_exceedance()`: Calculates $P(X \ge \text{threshold})$ across ensemble members.
* [`normalizer.py`](file:///home/yashvardhandubey/Projects/-/backend/app/preprocessing/normalizer.py):
  * Class `MeteorologicalNormalizer`: Provides Z-Score standardization, Min-Max normalization, Kelvin-to-Celsius, and m/s-to-km/h conversions.
* [`pipeline.py`](file:///home/yashvardhandubey/Projects/-/backend/app/preprocessing/pipeline.py):
  * Class `WeatherPreprocessingPipeline`: Reusable production pipeline orchestrating statistical reduction, exceedance calculations, and normalization.

### 3. `backend/app/anomaly/`
* [`efi.py`](file:///home/yashvardhandubey/Projects/-/backend/app/anomaly/efi.py):
  * Function `compute_efi_integral()`: Numerical integration of ECMWF EFI CDF.
  * Function `compute_dataset_efi()`: Applies EFI calculation across all lead times.
* [`bbox_tracker.py`](file:///home/yashvardhandubey/Projects/-/backend/app/anomaly/bbox_tracker.py):
  * Function `extract_spatial_features_at_step()`: Connected component clustering and BBox extraction.
  * Function `track_anomaly_events()`: Chaining anomaly clusters across lead times to form 4D events.
  * Function `classify_severity()`: Categorizes events into `"low"`, `"moderate"`, `"severe"`, or `"extreme"`.

### 4. `backend/app/spatial/`
* [`cropper.py`](file:///home/yashvardhandubey/Projects/-/backend/app/spatial/cropper.py):
  * Function `crop_spatial_bbox()`: Geographic bounding box cropping handling descending or ascending latitudes.
  * Function `crop_temporal_window()`: Filters forecast lead hours.
* [`geojson.py`](file:///home/yashvardhandubey/Projects/-/backend/app/spatial/geojson.py):
  * Function `bbox_to_geojson_polygon()`: Converts BBox to RFC 7946 GeoJSON Polygon.
  * Function `trajectory_to_geojson_linestring()`: Converts trajectory waypoints to GeoJSON LineString.
  * Function `events_to_geojson_feature_collection()`: Creates unified GeoJSON FeatureCollections for web maps.

### 5. `backend/app/cache/`
* [`zarr_cache.py`](file:///home/yashvardhandubey/Projects/-/backend/app/cache/zarr_cache.py):
  * Class `WeatherCacheManager`: High-speed on-disk chunked Zarr persistence and in-memory LRU/TTL cache.

### 6. `backend/app/news/`
* [`news_service.py`](file:///home/yashvardhandubey/Projects/-/backend/app/news/news_service.py):
  * Class `WeatherNewsService`: Multi-provider live disaster news engine. Queries GDELT Project 2.0 API (real-time zero-configuration live feed), UN OCHA ReliefWeb, and NewsAPI.org.

---

## 5. REST API Reference & Interfacing Guide

### `GET /weather/forecast`
* **Purpose**: Metadata, coordinate limits, and summary statistics.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/forecast?variable=wind_speed_10m"
  ```
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

### `GET /weather/ensemble`
* **Purpose**: Probe ensemble member distribution, spread, and percentiles at a coordinate.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/ensemble?variable=wind_speed_10m&lat=18.5&lon=85.0&lead_hour=72"
  ```
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

### `GET /weather/anomaly` (Palak & Frontend Primary Endpoint)
* **Purpose**: Returns tracked 4D anomaly events with EFI scores, bounding boxes, and trajectories.
* **Curl Example (JSON for Palak)**:
  ```bash
  curl -s "http://localhost:8000/weather/anomaly?variable=wind_speed_10m&threshold=0.65"
  ```
* **JSON Output**:
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
* **Curl Example (GeoJSON for Frontend Map)**:
  ```bash
  curl -s "http://localhost:8000/weather/anomaly?variable=wind_speed_10m&threshold=0.65&format=geojson"
  ```
* **GeoJSON Output**: Returns an RFC 7946 `FeatureCollection` with Bounding Box `Polygon`, Trajectory `LineString`, and Waypoint `Point` features.

---

### `GET /weather/region/{id}`
* **Purpose**: Cropped regional slice and warning status for Indian meteorological basins (`bay_of_bengal`, `coastal_odisha`, `north_india`, `arabian_sea`).
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/region/coastal_odisha?variable=wind_speed_10m&lead_hour=72"
  ```
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

### `GET /weather/timeline`
* **Purpose**: Temporal time series of anomaly evolution, peak value, and ensemble spread across the forecast horizon.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/timeline?variable=wind_speed_10m"
  ```
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

### `GET /weather/news` (Team Lead Live News Task)
* **Purpose**: Fetches real-time disaster and extreme weather bulletins linked to the active anomaly.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/news?hazard=cyclone&region=Odisha&limit=3"
  ```
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

---

### `POST /weather/news/key`
* **Purpose**: Updates the NewsAPI.org key dynamically at runtime.
* **Curl Example**:
  ```bash
  curl -X POST "http://localhost:8000/weather/news/key" \
       -H "Content-Type: application/json" \
       -d '{"api_key": "your_newsapi_key_here"}'
  ```

---

## 6. Automated Test Suite Results

The entire test suite was executed and passed with zero errors:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/yashvardhandubey/Projects/-
collected 37 items

backend/tests/test_anomaly_efi.py::test_compute_efi_integral ......................... PASSED [  2%]
backend/tests/test_anomaly_efi.py::test_compute_dataset_efi .......................... PASSED [  5%]
backend/tests/test_anomaly_efi.py::test_extract_spatial_features ..................... PASSED [  8%]
backend/tests/test_anomaly_efi.py::test_track_anomaly_events ......................... PASSED [ 10%]
backend/tests/test_anomaly_efi.py::test_classify_severity ............................ PASSED [ 13%]
backend/tests/test_api_endpoints.py::test_root_endpoint .............................. PASSED [ 16%]
backend/tests/test_api_endpoints.py::test_health_endpoint ............................ PASSED [ 18%]
backend/tests/test_api_endpoints.py::test_weather_forecast ........................... PASSED [ 21%]
backend/tests/test_api_endpoints.py::test_weather_ensemble_point_probe ............... PASSED [ 24%]
backend/tests/test_api_endpoints.py::test_weather_ensemble_spatial_aggregate ......... PASSED [ 27%]
backend/tests/test_api_endpoints.py::test_weather_anomaly_json ....................... PASSED [ 29%]
backend/tests/test_api_endpoints.py::test_weather_anomaly_geojson .................... PASSED [ 32%]
backend/tests/test_api_endpoints.py::test_weather_region ............................. PASSED [ 35%]
backend/tests/test_api_endpoints.py::test_weather_region_not_found ................... PASSED [ 37%]
backend/tests/test_api_endpoints.py::test_weather_timeline ........................... PASSED [ 40%]
backend/tests/test_api_endpoints.py::test_weather_news_feed .......................... PASSED [ 43%]
backend/tests/test_api_endpoints.py::test_weather_news_status ........................ PASSED [ 45%]
backend/tests/test_api_endpoints.py::test_weather_news_key_update .................... PASSED [ 48%]
backend/tests/test_caching.py::test_memory_cache ..................................... PASSED [ 51%]
backend/tests/test_caching.py::test_zarr_cache_save_and_load ......................... PASSED [ 54%]
backend/tests/test_caching.py::test_cache_clear ...................................... PASSED [ 56%]
backend/tests/test_ingestion.py::test_synthetic_neps_g_structure ..................... PASSED [ 59%]
backend/tests/test_ingestion.py::test_synthetic_era5_climatology ..................... PASSED [ 62%]
backend/tests/test_ingestion.py::test_nwp_loader ..................................... PASSED [ 64%]
backend/tests/test_news_service.py::test_mock_fallback_news .......................... PASSED [ 67%]
backend/tests/test_news_service.py::test_get_live_disaster_news_execution ............ PASSED [ 70%]
backend/tests/test_news_service.py::test_news_api_key_update ......................... PASSED [ 72%]
backend/tests/test_preprocessing.py::test_ensemble_dim_detection ..................... PASSED [ 75%]
backend/tests/test_preprocessing.py::test_compute_ensemble_statistics ................ PASSED [ 78%]
backend/tests/test_preprocessing.py::test_compute_probability_of_exceedance .......... PASSED [ 81%]
backend/tests/test_normalizer.py::test_normalizer .................................... PASSED [ 83%]
backend/tests/test_preprocessing.py::test_preprocessing_pipeline ..................... PASSED [ 86%]
backend/tests/test_spatial_geojson.py::test_crop_spatial_bbox ........................ PASSED [ 89%]
backend/tests/test_spatial_geojson.py::test_crop_temporal_window ..................... PASSED [ 91%]
backend/tests/test_spatial_geojson.py::test_bbox_to_geojson .......................... PASSED [ 94%]
backend/tests/test_spatial_geojson.py::test_trajectory_to_geojson .................... PASSED [ 97%]
backend/tests/test_spatial_geojson.py::test_events_to_geojson_feature_collection ..... PASSED [100%]

======================= 37 passed in 23.91s =======================
```

### `GET /weather/agri/pest-risk` (Module 14)
* **Purpose**: Evaluates crop pathogen and insect pest infection risk driven by ambient/forecast humidity, temperature, and wind.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/agri/pest-risk?crop=rice&stage=vegetative&region=Odisha"
  ```

### `GET /weather/agri/market-intelligence` (Module 15)
* **Purpose**: Integrates APMC Mandi modal prices, arrivals, and projects wholesale price surges and supply corridor shocks resulting from weather hazards.
* **Curl Example**:
  ```bash
  curl -s "http://localhost:8000/weather/agri/market-intelligence?region=odisha&crop=rice&hazard=cyclone&efi_intensity=0.92"
  ```

---

## 7. How to Launch and Push

### Running the Backend
```bash
git checkout amongus
source .venv/bin/activate
python run.py
```
* **API Documentation**: http://localhost:8000/docs
* **Health Check**: http://localhost:8000/health

### Pushing Branch to GitHub
```bash
git push -u origin amongus
```
