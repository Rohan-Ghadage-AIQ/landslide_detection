"""
generate_synthetic_data.py
----------------------------
Generates a small synthetic DEM, NDVI raster, landslide inventory, and
daily rainfall record purely so the FULL pipeline (main.py) can be run
end-to-end and verified to work mechanically, without needing real Pune
data on hand yet.

THIS IS DEMO DATA ONLY. Replace with real harmonized rasters (Section 4
of the R&D docs) and a real historical inventory before trusting any
output. The synthetic "hill" and synthetic "historical event" below are
constructed, not measured.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin

import config


def _synthetic_dem(size=150, cell_size=10):
    """A single ridge/hill with some noise -- enough terrain variety for
    slope/TWI/etc. to be non-trivial."""
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.35, size * 0.6
    ridge = 400 - 0.06 * ((x - cx) ** 2 + 0.5 * (y - cy) ** 2)
    ridge = np.clip(ridge, 50, None)
    rng = np.random.default_rng(config.RANDOM_STATE)
    noise = rng.normal(0, 3, size=(size, size))
    dem = ridge + noise
    return dem.astype("float32")


def _synthetic_ndvi(size=150):
    rng = np.random.default_rng(config.RANDOM_STATE + 1)
    return np.clip(rng.normal(0.5, 0.15, size=(size, size)), -1, 1).astype("float32")


def generate_all(size=150, origin_x=500000.0, origin_y=8200000.0):
    """
    origin_x/y are arbitrary UTM-like coordinates so the synthetic AOI
    behaves like a real projected raster (EPSG:32643-style units, metres).
    """
    cell_size = config.TARGET_RESOLUTION_M
    transform = from_origin(origin_x, origin_y, cell_size, cell_size)

    dem = _synthetic_dem(size, cell_size)
    ndvi = _synthetic_ndvi(size)

    meta = {
        "driver": "GTiff", "height": size, "width": size, "count": 1,
        "dtype": "float32", "crs": config.TARGET_CRS, "transform": transform,
    }
    with rasterio.open(config.DEM_PATH, "w", **meta) as dst:
        dst.write(dem, 1)
    with rasterio.open(config.NDVI_BASELINE_PATH, "w", **meta) as dst:
        dst.write(ndvi, 1)

    # --- synthetic inventory: place a handful of "landslide" points on
    # the steep flank of the synthetic ridge, where slope is high --------
    from utils_raster import compute_slope_aspect
    slope, _ = compute_slope_aspect(dem, cell_size)
    steep_rows, steep_cols = np.where(slope > np.percentile(slope, 90))
    rng = np.random.default_rng(config.RANDOM_STATE + 2)
    chosen = rng.choice(len(steep_rows), size=min(12, len(steep_rows)),
                         replace=False)

    points = []
    for i in chosen:
        r, c = steep_rows[i], steep_cols[i]
        x, y = transform * (c + 0.5, r + 0.5)
        points.append(Point(x, y))

    inventory_gdf = gpd.GeoDataFrame(
        {"event_id": range(len(points))}, geometry=points, crs=config.TARGET_CRS)
    inventory_gdf.to_file(config.INVENTORY_PATH)

    # --- synthetic daily rainfall record, with a monsoon-like rain spike
    # in the days before a chosen "historical event date" ----------------
    dates = pd.date_range("2024-06-01", "2024-09-30", freq="D")
    rng2 = np.random.default_rng(config.RANDOM_STATE + 3)
    baseline = rng2.gamma(shape=1.5, scale=6, size=len(dates))  # mm/day
    rainfall = pd.Series(baseline, index=dates, name="rainfall_mm")

    event_date = pd.Timestamp("2024-07-20")
    spike_days = pd.date_range(event_date - pd.Timedelta(days=6), event_date)
    rainfall.loc[spike_days] += rng2.uniform(35, 70, size=len(spike_days))

    rainfall.to_csv(config.RAINFALL_CSV_PATH, header=True)

    # pick one inventory point to serve as the "historical event" used
    # for backtest validation
    event_point = points[0]

    return {
        "dem": dem, "ndvi": ndvi, "transform": transform,
        "inventory_gdf": inventory_gdf, "rainfall": rainfall,
        "event_x": event_point.x, "event_y": event_point.y,
        "event_date": event_date,
        "aoi_bounds": (origin_x, origin_y - size * cell_size,
                        origin_x + size * cell_size, origin_y),
    }


if __name__ == "__main__":
    data = generate_all()
    print("Synthetic data written to:", config.RAW_DIR)
    print("Historical test event date:", data["event_date"].date())
    print("Historical test event location (x, y):",
          data["event_x"], data["event_y"])
