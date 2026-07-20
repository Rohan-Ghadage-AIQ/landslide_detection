"""
main.py
--------
End-to-end pipeline runner:

  1. Load data (DEM, NDVI, inventory, rainfall)
  2. Build the static feature stack (terrain derivatives)
  3. Build the training table (positives + pseudo-absences)
  4. Train + evaluate with spatial cross-validation
  5. Train the final deployed model
  6. Produce a district-wide susceptibility raster
  7. Backtest the early-warning fusion logic against a historical event

Run as-is with `python main.py` and it will generate synthetic demo data
automatically (see generate_synthetic_data.py) so you can verify the
whole pipeline runs mechanically end-to-end.

TO USE WITH REAL DATA: set USE_SYNTHETIC_DATA = False below, populate
config.py's RAW_DIR paths with your real harmonized rasters + inventory
shapefile + rainfall CSV, and rerun. The rest of the code does not need
to change.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import config
from feature_engineering import build_static_feature_stack
from sampling import build_training_table
from train_model import (
    run_spatial_cv, train_final_model, save_model, optimize_threshold,
)
from validate_historical import backtest_event, summarize_backtest
from utils_raster import write_raster

USE_SYNTHETIC_DATA = True


def load_real_data():
    """Loads real harmonized rasters/vectors from config.RAW_DIR.
    Fill this in once you have real Section-2 data harmonized to the
    common grid (see config.TARGET_CRS / TARGET_RESOLUTION_M)."""
    with rasterio.open(config.DEM_PATH) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
    ndvi = None
    if os.path.exists(config.NDVI_BASELINE_PATH):
        with rasterio.open(config.NDVI_BASELINE_PATH) as src:
            ndvi = src.read(1).astype("float32")
    inventory_gdf = gpd.read_file(config.INVENTORY_PATH)
    rainfall = pd.read_csv(config.RAINFALL_CSV_PATH, index_col=0,
                            parse_dates=True)["rainfall_mm"]
    with rasterio.open(config.DEM_PATH) as src:
        bounds = src.bounds
        aoi_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    return dem, ndvi, transform, inventory_gdf, rainfall, aoi_bounds, None, None


def main():
    print("=" * 70)
    print("LANDSLIDE EARLY-WARNING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    if USE_SYNTHETIC_DATA:
        print("\n[1/7] Generating synthetic demo data (replace with real "
              "data for production use)...")
        from generate_synthetic_data import generate_all
        data = generate_all()
        dem, ndvi = data["dem"], data["ndvi"]
        transform = data["transform"]
        inventory_gdf = data["inventory_gdf"]
        rainfall = data["rainfall"]
        aoi_bounds = data["aoi_bounds"]
        event_x, event_y, event_date = (
            data["event_x"], data["event_y"], data["event_date"])
    else:
        print("\n[1/7] Loading real data from", config.RAW_DIR)
        (dem, ndvi, transform, inventory_gdf, rainfall, aoi_bounds,
         event_x, event_y) = load_real_data()
        event_date = None  # supply your own known historical event here

    print(f"    DEM shape: {dem.shape}, inventory points: {len(inventory_gdf)}, "
          f"rainfall days: {len(rainfall)}")

    # ---------------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------------
    print("\n[2/7] Building static feature stack (slope, TWI, SPI, ...)...")
    feature_stack = build_static_feature_stack(
        dem, config.TARGET_RESOLUTION_M, ndvi=ndvi)
    available_features = [c for c in config.FEATURE_COLUMNS
                           if c in feature_stack]
    print(f"    Features available: {available_features}")

    # ---------------------------------------------------------------
    # 3. Training table
    # ---------------------------------------------------------------
    print("\n[3/7] Building training table (positives + pseudo-absences)...")
    training_table = build_training_table(
        inventory_gdf, feature_stack, transform, aoi_bounds,
        pseudo_absence_ratio=config.PSEUDO_ABSENCE_RATIO,
        min_dist_m=config.PSEUDO_ABSENCE_MIN_DIST_M,
        random_state=config.RANDOM_STATE)
    training_table.to_csv(config.TRAINING_TABLE_PATH, index=False)
    print(f"    Training table: {len(training_table)} rows "
          f"({training_table['landslide'].sum()} positive)")

    # ---------------------------------------------------------------
    # 4. Spatial cross-validation
    # ---------------------------------------------------------------
    print("\n[4/7] Running spatial cross-validation...")
    cv_metrics, oof_probs = run_spatial_cv(
        training_table, available_features, model_type=config.MODEL_TYPE,
        n_splits=config.N_CV_SPLITS, block_size_m=config.SPATIAL_BLOCK_SIZE_M,
        random_state=config.RANDOM_STATE)
    print(cv_metrics.to_string(index=False))
    valid_mask = ~np.isnan(oof_probs)
    if valid_mask.sum() > 0 and training_table["landslide"][valid_mask].nunique() > 1:
        threshold, thresh_info = optimize_threshold(
            training_table["landslide"][valid_mask], oof_probs[valid_mask],
            min_precision=config.MIN_PRECISION_FOR_THRESHOLD)
        print(f"    Optimized decision threshold: {threshold:.3f}  "
              f"(precision={thresh_info['precision']}, recall={thresh_info['recall']})")
    else:
        print("    Not enough out-of-fold predictions to optimize a "
              "threshold on this run (expected on tiny demo data).")

    # ---------------------------------------------------------------
    # 5. Train final deployed model
    # ---------------------------------------------------------------
    print("\n[5/7] Training final model on all available labeled data...")
    final_model = train_final_model(
        training_table, available_features, model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE)
    save_model(final_model, config.MODEL_PATH)
    print(f"    Model saved to {config.MODEL_PATH}")

    # ---------------------------------------------------------------
    # 6. District-wide susceptibility raster
    # ---------------------------------------------------------------
    print("\n[6/7] Generating susceptibility raster...")
    stacked = np.stack([feature_stack[c] for c in available_features], axis=-1)
    flat = stacked.reshape(-1, stacked.shape[-1])
    valid = ~np.isnan(flat).any(axis=1)
    probs = np.full(flat.shape[0], np.nan, dtype="float32")
    probs[valid] = final_model.predict_proba(flat[valid])[:, 1]
    susceptibility_raster = probs.reshape(dem.shape)

    meta = {
        "driver": "GTiff", "height": dem.shape[0], "width": dem.shape[1],
        "count": 1, "dtype": "float32", "crs": config.TARGET_CRS,
        "transform": transform,
    }
    write_raster(config.SUSCEPTIBILITY_RASTER_PATH, susceptibility_raster, meta)
    print(f"    Susceptibility raster saved to {config.SUSCEPTIBILITY_RASTER_PATH}")

    # ---------------------------------------------------------------
    # 7. Backtest against a historical event
    # ---------------------------------------------------------------
    print("\n[7/7] Backtesting early-warning logic against a historical event...")
    if event_date is not None:
        trace = backtest_event(
            event_x, event_y, event_date, rainfall, final_model,
            feature_stack, transform, available_features, lookback_days=10)
        print(trace.to_string(index=False))
        summary = summarize_backtest(trace)
        print("\n   ", summary["message"])
    else:
        print("    No historical event supplied (event_date is None) -- "
              "set one in load_real_data() to backtest against a real "
              "past landslide.")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
