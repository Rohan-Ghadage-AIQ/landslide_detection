"""
prototype_leh.py
----------------
Leh Town Micro-Site Prototype
This script sets up a fully working, real-time prototype for Leh Town.
It uses real weather data (Open-Meteo ERA5) and the real DEM you download.

INSTRUCTIONS FOR USER:
1. Go to OpenTopography (https://portal.opentopography.org/)
2. Select SRTM GL1 (30m) or Copernicus GLO-30.
3. Enter these exact coordinates for the Bounding Box:
   South: 34.10
   North: 34.20
   West: 77.53
   East: 77.63
4. Download the GeoTIFF and save it as: data/raw/leh_dem.tif
5. Run this script: python prototype_leh.py
"""

import os
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

import config

LEH_LAT = 34.15
LEH_LON = 77.58
DEM_FILE = os.path.join(config.RAW_DIR, "leh_dem.tif")
WEATHER_FILE = os.path.join(config.RAW_DIR, "leh_weather.csv")
INVENTORY_FILE = os.path.join(config.RAW_DIR, "leh_inventory.shp")

def fetch_real_weather():
    """Downloads real daily rainfall and temperature for Leh Town."""
    print(f"Fetching Open-Meteo ERA5 weather for Leh Town ({LEH_LAT}, {LEH_LON})...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LEH_LAT}&longitude={LEH_LON}&"
        f"start_date=2020-01-01&end_date=2024-07-01&"
        f"daily=precipitation_sum,temperature_2m_mean&timezone=auto"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Weather API failed: {response.text}")
        
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
    print(f"✅ Downloaded {len(df)} days of real weather data to {WEATHER_FILE}")

def generate_synthetic_inventory():
    """
    Reads the real DEM you downloaded, calculates the actual slopes,
    and drops 15 fake landslides on the most dangerous cliffs so we can train the ML model.
    """
    print(f"\nLooking for real DEM at: {DEM_FILE}")
    if not os.path.exists(DEM_FILE):
        print(f"❌ ERROR: Cannot find {DEM_FILE}.")
        print("Please download the DEM from OpenTopography and save it to the path above.")
        return
        
    print("✅ Found DEM! Analyzing terrain to place synthetic landslides...")
    from utils_raster import compute_slope_aspect
    
    with rasterio.open(DEM_FILE) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        
    # Calculate real slopes from your real DEM
    slope, _ = compute_slope_aspect(dem_data, cell_size=30)
    
    # Find the top 2% steepest cliffs in Leh
    steep_rows, steep_cols = np.where(slope > np.percentile(slope, 98))
    
    rng = np.random.default_rng(config.RANDOM_STATE)
    chosen = rng.choice(len(steep_rows), size=min(15, len(steep_rows)), replace=False)
    
    points = []
    for i in chosen:
        r, c = steep_rows[i], steep_cols[i]
        x, y = transform * (c + 0.5, r + 0.5)
        points.append(Point(x, y))
        
    inventory_gdf = gpd.GeoDataFrame({"event_id": range(len(points))}, geometry=points, crs=crs)
    inventory_gdf.to_file(INVENTORY_FILE)
    
    print(f"✅ Generated 15 synthetic landslides on the steepest slopes!")
    print(f"✅ Saved inventory to {INVENTORY_FILE}")
    print("\n🎉 PROTOTYPE DATA READY! You can now run the ML pipeline.")

if __name__ == "__main__":
    print("=== LEH TOWN MICRO-SITE PROTOTYPE SETUP ===")
    fetch_real_weather()
    generate_synthetic_inventory()
