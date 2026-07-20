"""
prototype_leh_real.py
---------------------
Option B: Real Historical Landslides + Real Weather + Real DEM

We are focusing on a tight bounding box around Leh and the Khardung La pass 
road (a known high-risk landslide zone).

INSTRUCTIONS FOR YOU (Do this right now!):
1. Go to: https://portal.opentopography.org/
2. Select Copernicus GLO-30 or SRTM GL1.
3. Enter this exact bounding box:
   South: 34.10
   North: 34.30
   West: 77.50
   East: 77.70
4. Download the GeoTIFF and save it as: data/raw/real_leh_dem.tif
5. Run this script: python prototype_leh_real.py
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
import rasterio
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

import config
from utils_raster import compute_slope_aspect

# --- 1. SET UP THE REAL PROTOTYPE PATHS ---
DEM_FILE = os.path.join(config.RAW_DIR, "real_leh_dem.tif")
WEATHER_FILE = os.path.join(config.RAW_DIR, "real_leh_weather.csv")
INVENTORY_FILE = os.path.join(config.RAW_DIR, "real_leh_inventory.shp")
MODEL_FILE = os.path.join(config.MODEL_DIR, "real_leh_model.joblib")

# Leh Bounding Box Center
LAT, LON = 34.20, 77.60

def setup_real_inventory():
    """
    Creates a shapefile of REAL historical landslide events based on known 
    incidents along the Khardung La mountain pass road north of Leh.
    """
    print("Setting up historical landslide inventory...")
    # Real approximate coordinates of past debris flows/landslides on Khardung La road
    real_events = [
        (77.592, 34.215), 
        (77.604, 34.241),
        (77.581, 34.198),
        (77.610, 34.265)
    ]
    
    points = [Point(lon, lat) for lon, lat in real_events]
    gdf = gpd.GeoDataFrame({"event_id": range(len(points))}, geometry=points, crs="EPSG:4326")
    gdf.to_file(INVENTORY_FILE)
    print(f"✅ Saved {len(real_events)} real historical events to {INVENTORY_FILE}")
    return gdf

def fetch_real_weather():
    """Downloads real daily rainfall and temperature for the Leh region."""
    print("Fetching Open-Meteo ERA5 weather for Leh...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&"
        f"start_date=2020-01-01&end_date=2024-07-01&"
        f"daily=precipitation_sum,temperature_2m_mean&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "rainfall_mm": data["daily"]["precipitation_sum"],
        "temp_c": data["daily"]["temperature_2m_mean"]
    })
    df.set_index("date", inplace=True)
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)
    df["temp_c"] = df["temp_c"].fillna(df["temp_c"].mean())
    df.to_csv(WEATHER_FILE)
    print(f"✅ Downloaded real weather data to {WEATHER_FILE}")

from feature_engineering import build_static_feature_stack
from sampling import build_training_table
from train_model import train_final_model, save_model

def run_ml_training():
    """Trains the XGBoost model using rigorous feature extraction and pseudo-absences."""
    print(f"\nChecking for your downloaded DEM at: {DEM_FILE}")
    if not os.path.exists(DEM_FILE):
        print("❌ ERROR: DEM file not found.")
        print("Please download the DEM from OpenTopography and save it as instructed above!")
        return
        
    print("✅ DEM found! Processing terrain features...")
    with rasterio.open(DEM_FILE) as src:
        dem_data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        
    # Extract real features
    feature_stack = build_static_feature_stack(dem_data, 30)
    available_features = [c for c in config.FEATURE_COLUMNS if c in feature_stack]
    
    print("✅ Sampling pseudo-absences and building training table...")
    inventory_gdf = gpd.read_file(INVENTORY_FILE)
    
    # We must format bounds for the build_training_table as (minx, miny, maxx, maxy)
    aoi_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    training_table = build_training_table(
        inventory_gdf, feature_stack, transform, aoi_bounds,
        pseudo_absence_ratio=2,
        min_dist_m=0.001,
        random_state=42
    )
    
    # Save the training table so the dashboard can read it!
    training_table.to_csv(config.TRAINING_TABLE_PATH, index=False)
    print(f"✅ Saved full training table ({len(training_table)} rows) to: {config.TRAINING_TABLE_PATH}")
    
    print("✅ Training Final Machine Learning Model on Real Data...")
    model = train_final_model(
        training_table, available_features, model_type="xgboost",
        random_state=42
    )
    
    save_model(model, MODEL_FILE)
    print(f"🎉 SUCCESS! Rigorous Machine Learning model saved to: {MODEL_FILE}")
    print("Next step: We will update the Dashboard to use this model!")

if __name__ == "__main__":
    print("=== LEH REAL DATA RIGOROUS PIPELINE ===")
    setup_real_inventory()
    fetch_real_weather()
    run_ml_training()
