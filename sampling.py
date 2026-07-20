"""
sampling.py
-----------
Turns a raster feature stack + a point inventory (landslide locations)
into a flat training table, and generates pseudo-absence (negative class)
points.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def raster_stack_to_arrays(feature_stack: dict, transform, crs):
    """Utility: keeps the stack + georeferencing together for sampling."""
    return {"features": feature_stack, "transform": transform, "crs": crs}


def _rowcol_from_xy(x, y, transform):
    """Inverse-affine: map projected (x, y) -> (row, col) integer indices."""
    col, row = ~transform * (x, y)
    return int(row), int(col)


def extract_features_at_points(points_gdf, feature_stack: dict, transform,
                                 label_col=None):
    """
    points_gdf   : GeoDataFrame of point geometries, in the SAME crs as
                   the feature rasters.
    feature_stack: dict {feature_name: 2D numpy array}, all same shape.
    transform    : rasterio Affine transform for the feature rasters.
    label_col    : optional column name in points_gdf holding 0/1 labels.

    Returns a pandas DataFrame: one row per point that falls inside the
    raster extent, one column per feature (+ label, x, y if available).
    """
    any_arr = next(iter(feature_stack.values()))
    n_rows, n_cols = any_arr.shape

    records = []
    for _, row in points_gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        r, c = _rowcol_from_xy(x, y, transform)
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            continue  # point falls outside the raster grid -- skip
        rec = {name: arr[r, c] for name, arr in feature_stack.items()}
        rec["x"], rec["y"] = x, y
        if label_col is not None:
            rec["landslide"] = row[label_col]
        records.append(rec)

    df = pd.DataFrame(records)
    return df.dropna()  # drop points that landed on nodata pixels


def generate_pseudo_absences(inventory_gdf, aoi_bounds, n_samples,
                              min_dist_m=200, random_state=42, crs=None):
    """
    Randomly samples negative (non-landslide) points inside the AOI
    bounding box, excluding a buffer zone around known positive points.

    aoi_bounds : (minx, miny, maxx, maxy) in the same CRS as inventory_gdf.
    """
    rng = np.random.default_rng(random_state)
    exclusion_zone = inventory_gdf.geometry.buffer(min_dist_m).unary_union

    minx, miny, maxx, maxy = aoi_bounds
    points = []
    attempts = 0
    max_attempts = n_samples * 50
    while len(points) < n_samples and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        candidate = Point(x, y)
        if not exclusion_zone.contains(candidate):
            points.append(candidate)
        attempts += 1

    gdf = gpd.GeoDataFrame(geometry=points, crs=crs or inventory_gdf.crs)
    gdf["landslide"] = 0
    return gdf


def build_training_table(inventory_gdf, feature_stack, transform, aoi_bounds,
                          pseudo_absence_ratio=2, min_dist_m=200,
                          random_state=42):
    """
    End-to-end: positives from the inventory + generated pseudo-absences,
    both sampled against the feature stack, combined into one table.
    """
    inventory_gdf = inventory_gdf.copy()
    inventory_gdf["landslide"] = 1

    positives = extract_features_at_points(
        inventory_gdf, feature_stack, transform, label_col="landslide")

    n_negatives = int(len(positives) * pseudo_absence_ratio)
    pseudo_absences = generate_pseudo_absences(
        inventory_gdf, aoi_bounds, n_negatives, min_dist_m, random_state,
        crs=inventory_gdf.crs)
    negatives = extract_features_at_points(
        pseudo_absences, feature_stack, transform, label_col="landslide")

    table = pd.concat([positives, negatives], ignore_index=True)
    return table.sample(frac=1, random_state=random_state).reset_index(drop=True)
