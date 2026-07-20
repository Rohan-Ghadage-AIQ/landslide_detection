"""
ladakh_config.py
-----------------
Configuration overrides specific to the Ladakh Union Territory pipeline.

Ladakh UT (~59,146 km²) spans two districts — Leh and Kargil — with
extreme elevation (2,500m to 7,000m+), arid terrain, and landslide
triggers dominated by cloudbursts, snowmelt, and freeze-thaw cycles
rather than sustained monsoon rainfall.

These overrides replace the Pune-district defaults in config.py for the
Ladakh MVP pipeline (main_ladakh.py).
"""

import os
import config

# ---------------------------------------------------------------------------
# Paths (separate from Pune model outputs)
# ---------------------------------------------------------------------------
LADAKH_MODEL_PATH = os.path.join(config.MODEL_DIR, "landslide_model_ladakh.joblib")
LADAKH_TRAINING_TABLE_PATH = os.path.join(config.PROCESSED_DIR, "ladakh_training_table.csv")
LADAKH_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "ladakh")
os.makedirs(LADAKH_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial reference / grid
# ---------------------------------------------------------------------------
# Ladakh spans ~75°E–80°E → entirely within UTM zone 43N (same as Pune
# by coincidence, but we use get_utm_epsg per-site for correctness).
# For the UT-wide grid, EPSG:32643 is the primary zone.
TARGET_CRS = "EPSG:32643"

# 30m resolution for UT-wide coverage.  10m would yield ~65 billion
# pixels for all of Ladakh — impractical for MVP.  30m is the
# Copernicus GLO-30 DEM native resolution and strikes a good balance.
TARGET_RESOLUTION_M = 30

# ---------------------------------------------------------------------------
# Six landslide-prone sub-regions across both districts
# ---------------------------------------------------------------------------
# Rough (lon, lat) anchors for known high-risk zones. These are
# illustrative geographic centers for demo-tile placement — NOT exact
# historical event coordinates.  Replace with real inventory locations
# for production use.
LADAKH_REGION_ANCHORS = {
    "leh_town": {
        "lon": 77.58, "lat": 34.15,
        "description": "Leh town - cloudburst-triggered debris flows (2010 event)",
        "ruggedness": 1.8,
        "elevation_base": 3500,
    },
    "khardung_la": {
        "lon": 77.60, "lat": 34.28,
        "description": "Khardung La pass - extreme elevation, freeze-thaw",
        "ruggedness": 2.2,
        "elevation_base": 4500,
    },
    "zanskar_valley": {
        "lon": 76.85, "lat": 33.50,
        "description": "Zanskar Valley - steep gorges, river-cut slopes",
        "ruggedness": 2.0,
        "elevation_base": 3800,
    },
    "kargil_town": {
        "lon": 76.13, "lat": 34.55,
        "description": "Kargil town - unstable slopes, seismic zone V",
        "ruggedness": 1.6,
        "elevation_base": 2700,
    },
    "nubra_valley": {
        "lon": 77.56, "lat": 34.70,
        "description": "Nubra Valley - GLOF / debris flow corridor",
        "ruggedness": 2.5,
        "elevation_base": 3100,
    },
    "drass_sector": {
        "lon": 75.75, "lat": 34.43,
        "description": "Drass sector - heavy snowfall, avalanche-to-slide transitions",
        "ruggedness": 1.9,
        "elevation_base": 3200,
    },
}

# ---------------------------------------------------------------------------
# Feature engineering — Ladakh-specific parameters
# ---------------------------------------------------------------------------
# API decay and window (same approach, different calibration)
API_DECAY_K = 0.85           # faster decay in arid climate (less soil moisture retention)
API_WINDOW_DAYS = 10         # shorter lookback — events are cloudburst-driven

# Early-warning lead time
FORECAST_LEAD_DAYS = 5       # shorter lead for cloudburst events

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
PSEUDO_ABSENCE_MIN_DIST_M = 200     # wider buffer for 30m resolution
PSEUDO_ABSENCE_RATIO = 2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------
SPATIAL_BLOCK_SIZE_M = 1000         # larger blocks for 30m grid
N_CV_SPLITS = 6                     # one per sub-region (leave-one-out)

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
MODEL_TYPE = "xgboost"
MIN_PRECISION_FOR_THRESHOLD = 0.3

FEATURE_COLUMNS = [
    "elevation", "slope", "aspect", "curvature", "twi", "spi",
    "dist_to_drainage", "ndvi", "snowmelt_proxy",
]

# ---------------------------------------------------------------------------
# Early-warning risk fusion thresholds — Ladakh calibration
# ---------------------------------------------------------------------------
# Ladakh is arid: a 30mm cloudburst is catastrophic here (vs routine
# in Kerala).  Trigger thresholds are MUCH lower than monsoon-belt India.
RISK_LEVELS = [
    (0.00, 0.15, "Low"),
    (0.15, 0.30, "Moderate"),
    (0.30, 0.50, "High"),
    (0.50, 1.01, "Very High"),
]

# API saturation reference — much lower for Ladakh's dry soils
API_SATURATION_REFERENCE_MM = 50.0

# Forecast trigger reference — a 30mm forecast over 5 days is serious here
FORECAST_TRIGGER_REFERENCE_MM = 30.0

# ---------------------------------------------------------------------------
# Snowmelt proxy configuration
# ---------------------------------------------------------------------------
# Day-of-year range for the primary snowmelt window in Ladakh
# (roughly late April through mid-July when freeze-thaw is most active).
SNOWMELT_WINDOW_START_DOY = 110   # ~April 20
SNOWMELT_WINDOW_END_DOY = 200    # ~July 19
SNOWMELT_PEAK_DOY = 160          # ~June 9 (peak melt season)
