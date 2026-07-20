"""
ladakh_real_data.py
--------------------
Phase 2: Real Data Integration.
This script reads the real user-provided shapefile for Ladakh, extracts its
geographic center, and connects to the Open-Meteo ERA5 API to download 
real historical rainfall and temperature data (for the Snowmelt Proxy).

NOTE on DEM (Elevation Data): 
Downloading a 30m resolution DEM for the entire 45,000 sq km of Ladakh is a 
multi-gigabyte operation that requires a free API key from OpenTopography or 
Copernicus. For this script, we download the real weather data, but expect the 
user to place the downloaded DEM manually into data/raw/real_ladakh_dem.tif.
"""

import os
import json
import requests
import pandas as pd
import geopandas as gpd

import config

def get_aoi_centroid(shp_path):
    """Reads the user's shapefile and finds its exact center point."""
    print(f"Reading AOI from: {shp_path}")
    gdf = gpd.read_file(shp_path)
    
    # Ensure it's in lat/lon (EPSG:4326) for the Weather API
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
        
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0
    
    print(f"AOI Bounds: {bounds}")
    print(f"AOI Centroid: Lat {center_lat:.4f}, Lon {center_lon:.4f}")
    return center_lat, center_lon

def download_real_weather_data(lat, lon, start_date="2020-01-01", end_date="2024-07-01"):
    """
    Downloads REAL historical daily rainfall and temperature from the Open-Meteo 
    ERA5 reanalysis archive for the given coordinates.
    """
    print(f"\nConnecting to Open-Meteo ERA5 API for Lat {lat:.4f}, Lon {lon:.4f}...")
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily=precipitation_sum,temperature_2m_mean&"
        f"timezone=auto"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather data: {response.status_code} - {response.text}")
        
    data = response.json()
    
    # Convert JSON to pandas DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "rainfall_mm": data["daily"]["precipitation_sum"],
        "temp_c": data["daily"]["temperature_2m_mean"]
    })
    
    df.set_index("date", inplace=True)
    
    # Fill any missing days with 0 rain and average temp
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)
    df["temp_c"] = df["temp_c"].fillna(df["temp_c"].mean())
    
    csv_path = os.path.join(config.RAW_DIR, "real_ladakh_weather.csv")
    df.to_csv(csv_path)
    print(f"Successfully downloaded {len(df)} days of real weather data to: {csv_path}")
    
    # Show a preview of the data
    print("\nWeather Data Preview:")
    print(df.tail())
    return df

def generate_real_data_pipeline():
    """Main execution function to fetch real data."""
    aoi_path = os.path.join(config.BASE_DIR, "Leh_Ladakh_AOI", "leh_ladakh.shp")
    
    if not os.path.exists(aoi_path):
        print(f"ERROR: Could not find shapefile at {aoi_path}")
        return
        
    lat, lon = get_aoi_centroid(aoi_path)
    
    weather_df = download_real_weather_data(lat, lon)
    
    print("\n=======================================================")
    print("REAL DATA FETCH COMPLETE.")
    print("=======================================================")
    print("NEXT STEPS FOR FULL REAL-TIME PIPELINE:")
    print("1. Please download the 30m DEM for the AOI bounds printed above.")
    print("   (Use Copernicus GLO-30 or Bhuvan Cartosat-1).")
    print("2. Save the DEM to: data/raw/real_ladakh_dem.tif")
    print("3. Add historical landslide GPS coordinates to: data/raw/real_inventory.shp")
    print("=======================================================\n")

if __name__ == "__main__":
    generate_real_data_pipeline()
