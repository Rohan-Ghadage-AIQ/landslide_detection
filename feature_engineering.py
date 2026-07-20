"""
feature_engineering.py
-----------------------
Builds the full static + dynamic feature raster stack from harmonized
input rasters, using the terrain-derivative functions in utils_raster.py.
"""

import numpy as np
import pandas as pd

from utils_raster import (
    compute_slope_aspect, compute_curvature, compute_terrain_ruggedness,
    flow_accumulation_d8, compute_twi, compute_spi, distance_to_features,
)


def build_static_feature_stack(dem, cell_size, ndvi=None,
                                drainage_threshold_percentile=95):
    """
    dem : 2D numpy array of elevation values (already harmonized to the
          common grid / resolution / CRS).
    cell_size : pixel size in metres (must match TARGET_RESOLUTION_M).
    ndvi : optional 2D array, same shape as dem.

    Returns a dict of {feature_name: 2D array}.
    """
    slope, aspect = compute_slope_aspect(dem, cell_size)
    curvature = compute_curvature(dem, cell_size)
    tri = compute_terrain_ruggedness(dem)
    flow_accum = flow_accumulation_d8(dem)
    twi = compute_twi(slope, flow_accum, cell_size)
    spi = compute_spi(slope, flow_accum, cell_size)

    # Derive a simple drainage-network mask from high flow-accumulation
    # cells, then compute distance to that network. This is a simplified
    # stand-in for a real hydrologically-derived stream layer -- swap in
    # an actual drainage vector layer if you have one.
    threshold = np.percentile(flow_accum, drainage_threshold_percentile)
    drainage_mask = flow_accum >= threshold
    dist_to_drainage = distance_to_features(drainage_mask, cell_size)

    features = {
        "elevation": dem.astype("float32"),
        "slope": slope.astype("float32"),
        "aspect": aspect.astype("float32"),
        "curvature": curvature.astype("float32"),
        "tri": tri.astype("float32"),
        "flow_accum": flow_accum.astype("float32"),
        "twi": twi.astype("float32"),
        "spi": spi.astype("float32"),
        "dist_to_drainage": dist_to_drainage.astype("float32"),
    }
    if ndvi is not None:
        features["ndvi"] = ndvi.astype("float32")
    return features


def compute_api_series(daily_rainfall: pd.Series, k=0.9, window=15):
    """
    Antecedent Precipitation Index, computed for every day in the series
    (not just a single date), using a rolling weighted sum.

    daily_rainfall : pandas Series indexed by date (ascending), values = mm.
    Returns a pandas Series of the same length (NaN for the first
    `window` days where there isn't enough history).
    """
    values = daily_rainfall.values.astype("float64")
    n = len(values)
    api = np.full(n, np.nan)
    weights = np.array([k ** (window - 1 - i) for i in range(window)])
    for t in range(window - 1, n):
        window_vals = values[t - window + 1: t + 1]
        api[t] = float(np.dot(window_vals, weights))
    return pd.Series(api, index=daily_rainfall.index, name="API")
