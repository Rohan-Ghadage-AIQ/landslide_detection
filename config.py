"""
config.py
---------
Central configuration for the landslide early-warning pipeline.

Edit these values for your real AOI / data before running main.py against
real data. Defaults here are reasonable starting points, NOT validated
facts about Pune district -- calibrate against your own inventory.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

DEM_PATH = os.path.join(RAW_DIR, "real_leh_dem.tif")
NDVI_BASELINE_PATH = os.path.join(RAW_DIR, "ndvi_baseline.tif")
LITHOLOGY_PATH = os.path.join(RAW_DIR, "lithology.tif")
SOIL_PATH = os.path.join(RAW_DIR, "soil.tif")
LANDCOVER_PATH = os.path.join(RAW_DIR, "landcover.tif")
ROADS_PATH = os.path.join(RAW_DIR, "roads.shp")
INVENTORY_PATH = os.path.join(RAW_DIR, "real_leh_inventory.shp")
RAINFALL_CSV_PATH = os.path.join(RAW_DIR, "real_leh_weather.csv")

SUSCEPTIBILITY_RASTER_PATH = os.path.join(OUTPUT_DIR, "real_leh_susceptibility_map.tif")
MODEL_PATH = os.path.join(MODEL_DIR, "real_leh_model.joblib")
TRAINING_TABLE_PATH = os.path.join(PROCESSED_DIR, "real_leh_training_table.csv")

# Ladakh Union Territory paths (separate from Pune model)
LADAKH_MODEL_PATH = os.path.join(MODEL_DIR, "landslide_model_ladakh.joblib")
LADAKH_TRAINING_TABLE_PATH = os.path.join(PROCESSED_DIR, "ladakh_training_table.csv")
LADAKH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ladakh")
os.makedirs(LADAKH_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial reference / grid
# ---------------------------------------------------------------------------
# EPSG:32643 = WGS84 / UTM zone 43N -- a reasonable projected CRS for Pune
# district (keeps distances in metres). Verify this is correct for your
# exact AOI before using it in production.
TARGET_CRS = "EPSG:32643"
TARGET_RESOLUTION_M = 10  # metres per pixel

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
# Antecedent Precipitation Index decay constant and lookback window.
# These are common literature starting points -- NOT calibrated to Pune.
API_DECAY_K = 0.9
API_WINDOW_DAYS = 15

# Early-warning lead time
FORECAST_LEAD_DAYS = 7

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
PSEUDO_ABSENCE_MIN_DIST_M = 100      # exclusion buffer around known events
PSEUDO_ABSENCE_RATIO = 2             # negatives per positive
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------
SPATIAL_BLOCK_SIZE_M = 500          # size of each CV block, in metres
N_CV_SPLITS = 5

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
MODEL_TYPE = "xgboost"               # "xgboost" or "random_forest"
MIN_PRECISION_FOR_THRESHOLD = 0.3    # business-set minimum acceptable precision

FEATURE_COLUMNS = [
    "elevation", "slope", "aspect", "curvature", "twi", "spi",
    "dist_to_drainage", "ndvi",
]

# ---------------------------------------------------------------------------
# Early-warning risk fusion thresholds
# ---------------------------------------------------------------------------
# combined_score = susceptibility_prob * trigger_score  (see early_warning.py)
# These cut points are starting defaults -- must be recalibrated once you
# have a season of real forecast-vs-outcome data.
RISK_LEVELS = [
    (0.00, 0.15, "Low"),
    (0.15, 0.35, "Moderate"),
    (0.35, 0.60, "High"),
    (0.60, 1.01, "Very High"),
]

# API value (mm, weighted) above which ground is considered "saturated"
# for trigger-scoring purposes. This MUST be tuned against real local data.
API_SATURATION_REFERENCE_MM = 150.0

# Forecast rainfall (mm, over the 7-day lead window) considered a strong
# trigger signal. Also a starting default, not a validated threshold.
FORECAST_TRIGGER_REFERENCE_MM = 100.0
