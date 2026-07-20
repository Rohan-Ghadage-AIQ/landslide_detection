"""
ladakh_data.py
----------------
Generates synthetic data for SIX landslide-prone sub-regions across
Ladakh Union Territory (Leh + Kargil districts), so the Ladakh pipeline
(main_ladakh.py) can run end-to-end without real data.

Key differences from the national demo (multi_site_data.py):
- Higher-elevation DEMs (3,000–5,000m+ base vs 400–600m)
- Low-NDVI arid terrain (0.05–0.25 vs 0.5 in monsoon India)
- Short, intense cloudburst rainfall spikes (1–2 day bursts)
- Seasonal snowmelt proxy feature (freeze-thaw cycle)

THIS IS DEMO DATA ONLY — see implementation_plan.md Phase 2 for real
data sources (Copernicus GLO-30, GSI inventory, IMD rainfall, ERA5).
"""

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from geo_utils import get_utm_epsg
import ladakh_config


def _synthetic_ladakh_dem(size, cell_size, seed, ruggedness=1.5,
                          elevation_base=3500):
    """
    Generates a synthetic DEM representative of Ladakh terrain:
    high-elevation base with sharp ridges and deep valleys.
    """
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.4, size * 0.55

    # Primary ridge — higher and sharper than the Pune/monsoon demo
    ridge = elevation_base + 800 * ruggedness - 0.1 * ruggedness * (
        (x - cx) ** 2 + 0.7 * (y - cy) ** 2)
    ridge = np.clip(ridge, elevation_base * 0.6, None)

    # Add secondary gully/valley feature common in Ladakh terrain
    rng = np.random.default_rng(seed)
    gully = -60 * ruggedness * np.sin(0.08 * x + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, 6 * ruggedness, size=(size, size))

    dem = (ridge + gully + noise).astype("float32")
    return dem


def _synthetic_ladakh_ndvi(size, seed):
    """
    Ladakh is arid/semi-arid: NDVI is very low (bare rock/gravel)
    except near river valleys where some scrub vegetation grows.
    """
    rng = np.random.default_rng(seed)
    # Base: low NDVI (0.05–0.15) with sparse patches of 0.2–0.35
    base = rng.normal(0.10, 0.05, size=(size, size))

    # Simulate a river-valley strip with slightly higher vegetation
    y, x = np.mgrid[0:size, 0:size]
    valley_mask = np.abs(x - size * 0.4) < size * 0.1
    base[valley_mask] += rng.uniform(0.10, 0.25, size=valley_mask.sum())

    return np.clip(base, -0.1, 0.5).astype("float32")


def _synthetic_ladakh_rainfall(seed, ruggedness, event_offset_days=30):
    """
    Ladakh rainfall pattern: mostly dry with sudden intense cloudbursts.
    Total annual rainfall in Leh is only ~100mm — but individual events
    can dump 30–80mm in hours.
    """
    # Cover June–September (the window when most events occur)
    dates = pd.date_range("2024-05-01", "2024-09-30", freq="D")
    rng = np.random.default_rng(seed)

    # Very low baseline: 0–3 mm/day with many dry days
    baseline = rng.exponential(scale=1.5, size=len(dates))
    # Many completely dry days (characteristic of arid Ladakh)
    dry_mask = rng.random(size=len(dates)) < 0.6
    baseline[dry_mask] = 0.0

    rainfall = pd.Series(baseline, index=dates, name="rainfall_mm")

    # The triggering cloudburst: 1–2 day intense spike
    event_date = dates[0] + pd.Timedelta(days=event_offset_days)
    spike_days = pd.date_range(
        event_date - pd.Timedelta(days=1), event_date)  # 2-day burst
    rainfall.loc[spike_days] += rng.uniform(
        35 * ruggedness, 80 * ruggedness, size=len(spike_days))

    return rainfall, event_date


def _snowmelt_proxy(dem, day_of_year):
    """
    Simple elevation + season-based snowmelt proxy.

    Logic: higher elevations have more snow, and melt is strongest
    near the peak melt DOY (ladakh_config.SNOWMELT_PEAK_DOY ≈ June 9).
    Returns a 0–1 score: higher = more active snowmelt.
    """
    # Elevation factor: normalize DEM to 0–1 range
    dmin, dmax = np.nanmin(dem), np.nanmax(dem)
    if dmax - dmin > 0:
        elev_factor = (dem - dmin) / (dmax - dmin)
    else:
        elev_factor = np.zeros_like(dem)

    # Seasonal factor: Gaussian centered on peak melt DOY
    peak = ladakh_config.SNOWMELT_PEAK_DOY
    sigma = (ladakh_config.SNOWMELT_WINDOW_END_DOY -
             ladakh_config.SNOWMELT_WINDOW_START_DOY) / 4.0
    seasonal_factor = np.exp(-0.5 * ((day_of_year - peak) / sigma) ** 2)

    proxy = (elev_factor * seasonal_factor).astype("float32")
    return proxy


def generate_ladakh_site(region_name, region_info, seed, size=80,
                         cell_size=30, event_offset_days=30):
    """
    Builds one synthetic Ladakh sub-region site: a local DEM tile +
    rainfall record + snowmelt proxy + a single labeled event location,
    all in the correct local UTM CRS.
    """
    lon, lat = region_info["lon"], region_info["lat"]
    local_crs = get_utm_epsg(lon, lat)

    dem = _synthetic_ladakh_dem(
        size, cell_size, seed,
        ruggedness=region_info["ruggedness"],
        elevation_base=region_info["elevation_base"])

    ndvi = _synthetic_ladakh_ndvi(size, seed + 1)

    # Arbitrary local projected origin (in a real run, comes from the
    # actual DEM tile's georeference after reprojection)
    origin_x, origin_y = 500000.0, 3800000.0
    transform = from_origin(origin_x, origin_y, cell_size, cell_size)

    # Place the "historical event" on the steepest slope
    from utils_raster import compute_slope_aspect
    slope, _ = compute_slope_aspect(dem, cell_size)
    r, c = np.unravel_index(np.argmax(slope), slope.shape)
    event_x, event_y = transform * (c + 0.5, r + 0.5)

    # Rainfall + event date
    rainfall, event_date = _synthetic_ladakh_rainfall(
        seed + 2, region_info["ruggedness"], event_offset_days)

    # Snowmelt proxy (computed for the event date's day-of-year)
    event_doy = event_date.day_of_year
    snowmelt = _snowmelt_proxy(dem, event_doy)

    return {
        "site_id": region_name,
        "local_crs": local_crs,
        "lon": lon,
        "lat": lat,
        "description": region_info["description"],
        "dem": dem,
        "ndvi": ndvi,
        "snowmelt_proxy": snowmelt,
        "transform": transform,
        "event_x": event_x,
        "event_y": event_y,
        "event_date": event_date,
        "rainfall": rainfall,
        "aoi_bounds": (origin_x, origin_y - size * cell_size,
                       origin_x + size * cell_size, origin_y),
    }


def generate_ladakh_demo_sites():
    """
    Returns a list of 6 synthetic Ladakh sites, one per sub-region
    defined in ladakh_config.LADAKH_REGION_ANCHORS.
    """
    sites = []
    for i, (name, info) in enumerate(
            ladakh_config.LADAKH_REGION_ANCHORS.items()):
        # Stagger event offsets so they don't all land on the same date
        event_offset = 25 + i * 8
        site = generate_ladakh_site(
            name, info, seed=200 + i * 10,
            event_offset_days=event_offset)
        sites.append(site)
    return sites


if __name__ == "__main__":
    print("Generating synthetic Ladakh demo sites...")
    sites = generate_ladakh_demo_sites()
    for s in sites:
        print(f"  {s['site_id']:25s}  CRS={s['local_crs']}  "
              f"DEM shape={s['dem'].shape}  "
              f"event={s['event_date'].date()}")
    print(f"\n{len(sites)} Ladakh sub-region sites generated.")
