"""
multi_site_data.py
--------------------
Generates synthetic data for SEVERAL different "regions" (e.g. standing
in for Pune/Western Ghats, Kerala, Uttarakhand, Sikkim...) so the
multi-site / nationwide pipeline can be run and verified end-to-end
without needing real national data on hand yet.

THIS IS DEMO DATA ONLY -- see README section "Switching to real
nationwide data" for what to replace each part with.
"""

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from geo_utils import get_utm_epsg

# Rough (lon, lat) anchors for a handful of REAL India regions known for
# landslide activity, used only to make the demo geographically plausible.
# I am NOT claiming these are exact historical event coordinates -- they
# are illustrative anchors for demo purposes only; replace with real
# inventory coordinates for anything used in decision-making.
REGION_ANCHORS = {
    "western_ghats_pune":   (73.55, 18.55),   # Mulshi/Velhe area, Maharashtra
    "kerala_wayanad":       (76.13, 11.68),   # Wayanad, Kerala
    "uttarakhand_himalaya": (78.95, 30.35),   # near Rudraprayag, Uttarakhand
    "sikkim_himalaya":      (88.51, 27.33),   # near Gangtok, Sikkim
}


def _synthetic_local_dem(size, cell_size, seed, ruggedness=1.0):
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.4, size * 0.55
    ridge = 600 * ruggedness - 0.08 * ruggedness * ((x - cx) ** 2 + 0.6 * (y - cy) ** 2)
    ridge = np.clip(ridge, 20, None)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 4, size=(size, size))
    return (ridge + noise).astype("float32")


def generate_region_site(region_name, lon, lat, seed, size=80, cell_size=10,
                          ruggedness=1.0, event_offset_days=45):
    """
    Builds one synthetic "site": a local DEM tile + rainfall record +
    a single labeled event location, all in that region's correct local
    UTM CRS (via geo_utils.get_utm_epsg).
    """
    local_crs = get_utm_epsg(lon, lat)
    dem = _synthetic_local_dem(size, cell_size, seed, ruggedness)

    rng = np.random.default_rng(seed + 1)
    ndvi = np.clip(rng.normal(0.5, 0.15, size=(size, size)), -1, 1).astype("float32")

    # Arbitrary local projected origin -- in a real run this would come
    # from actually reprojecting the downloaded DEM tile, not be invented.
    origin_x, origin_y = 500000.0, 3000000.0
    transform = from_origin(origin_x, origin_y, cell_size, cell_size)

    # Place the "historical event" on a steep flank of this site's ridge
    from utils_raster import compute_slope_aspect
    slope, _ = compute_slope_aspect(dem, cell_size)
    r, c = np.unravel_index(np.argmax(slope), slope.shape)
    event_x, event_y = transform * (c + 0.5, r + 0.5)

    dates = pd.date_range("2024-05-01", "2024-10-31", freq="D")
    rng2 = np.random.default_rng(seed + 2)
    baseline = rng2.gamma(shape=1.5, scale=6 * ruggedness, size=len(dates))
    rainfall = pd.Series(baseline, index=dates, name="rainfall_mm")
    event_date = dates[0] + pd.Timedelta(days=event_offset_days)
    spike_days = pd.date_range(event_date - pd.Timedelta(days=6), event_date)
    rainfall.loc[spike_days] += rng2.uniform(35, 80, size=len(spike_days))

    return {
        "site_id": region_name,
        "local_crs": local_crs,
        "dem": dem, "ndvi": ndvi, "transform": transform,
        "event_x": event_x, "event_y": event_y, "event_date": event_date,
        "rainfall": rainfall,
        "aoi_bounds": (origin_x, origin_y - size * cell_size,
                        origin_x + size * cell_size, origin_y),
    }


def generate_national_demo_sites():
    """Returns a list of synthetic sites, one per REGION_ANCHORS entry,
    with varying ruggedness so they aren't all identical."""
    ruggedness_by_region = {
        "western_ghats_pune": 1.0,
        "kerala_wayanad": 1.3,
        "uttarakhand_himalaya": 1.8,
        "sikkim_himalaya": 1.6,
    }
    sites = []
    for i, (name, (lon, lat)) in enumerate(REGION_ANCHORS.items()):
        sites.append(generate_region_site(
            name, lon, lat, seed=100 + i * 10,
            ruggedness=ruggedness_by_region[name]))
    return sites
