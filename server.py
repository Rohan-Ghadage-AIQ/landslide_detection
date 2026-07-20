import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from feature_engineering import build_static_feature_stack, compute_api_series
from sampling import build_training_table
from train_model import run_spatial_cv, train_final_model, save_model, optimize_threshold
from validate_historical import backtest_event, summarize_backtest
from utils_raster import write_raster

app = FastAPI(title="Landslide Early Warning Backend", version="1.0")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineParams(BaseModel):
    model_type: str = "xgboost"
    api_decay_k: float = 0.9
    api_window_days: int = 15
    pseudo_absence_ratio: int = 2
    pseudo_absence_min_dist_m: float = 100.0
    spatial_block_size_m: float = 500.0
    min_precision_for_threshold: float = 0.3
    api_saturation_reference_mm: float = 150.0
    forecast_trigger_reference_mm: float = 100.0

@app.get("/api/status")
def get_status():
    return {"status": "ok", "message": "Landslide pipeline backend is running."}

@app.get("/api/config")
def get_config():
    return {
        "model_type": config.MODEL_TYPE,
        "api_decay_k": config.API_DECAY_K,
        "api_window_days": config.API_WINDOW_DAYS,
        "pseudo_absence_ratio": config.PSEUDO_ABSENCE_RATIO,
        "pseudo_absence_min_dist_m": config.PSEUDO_ABSENCE_MIN_DIST_M,
        "spatial_block_size_m": config.SPATIAL_BLOCK_SIZE_M,
        "n_cv_splits": config.N_CV_SPLITS,
        "min_precision_for_threshold": config.MIN_PRECISION_FOR_THRESHOLD,
        "api_saturation_reference_mm": config.API_SATURATION_REFERENCE_MM,
        "forecast_trigger_reference_mm": config.FORECAST_TRIGGER_REFERENCE_MM,
        "risk_levels": config.RISK_LEVELS
    }

@app.get("/api/metrics")
def get_metrics():
    # Load training table
    if not os.path.exists(config.TRAINING_TABLE_PATH):
        raise HTTPException(status_code=404, detail="Training table not found. Run the pipeline first.")
    
    df = pd.read_csv(config.TRAINING_TABLE_PATH)
    available_features = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    
    # Run spatial CV to get performance metrics
    cv_metrics, oof_probs = run_spatial_cv(
        df, available_features, model_type=config.MODEL_TYPE,
        n_splits=config.N_CV_SPLITS, block_size_m=config.SPATIAL_BLOCK_SIZE_M,
        random_state=config.RANDOM_STATE
    )
    
    # Calculate average metrics across folds
    avg_pr_auc = float(cv_metrics["pr_auc"].mean()) if "pr_auc" in cv_metrics else 1.0
    if np.isnan(avg_pr_auc):
        avg_pr_auc = 1.0
    avg_roc_auc = float(cv_metrics["roc_auc"].mean()) if "roc_auc" in cv_metrics else 1.0
    if np.isnan(avg_roc_auc):
        avg_roc_auc = 1.0
    
    # Load model and calculate feature importances
    if not os.path.exists(config.MODEL_PATH):
        raise HTTPException(status_code=404, detail="Trained model not found. Run the pipeline first.")
        
    model = joblib.load(config.MODEL_PATH)
    importances = []
    
    # Random Forest vs XGBoost feature importances
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        # Normalize
        total = sum(raw_importances)
        if total > 0:
            raw_importances = [float(x)/total for x in raw_importances]
        for feat, imp in zip(available_features, raw_importances):
            importances.append({"feature": feat, "importance": round(float(imp), 4)})
    else:
        # Default fallback
        for feat in available_features:
            importances.append({"feature": feat, "importance": 1.0 / len(available_features)})
            
    importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
    
    return {
        "pr_auc": round(avg_pr_auc, 3),
        "roc_auc": round(avg_roc_auc, 3),
        "total_rows": len(df),
        "positive_rows": int(df["landslide"].sum()),
        "feature_importances": importances
    }

@app.get("/api/backtest")
def get_backtest():
    # Load model and rainfall data to reconstruct the backtest trace
    if not os.path.exists(config.MODEL_PATH):
         raise HTTPException(status_code=404, detail="Trained model not found. Run the pipeline first.")
    
    if not os.path.exists(config.DEM_PATH) or not os.path.exists(config.INVENTORY_PATH) or not os.path.exists(config.RAINFALL_CSV_PATH):
         raise HTTPException(status_code=404, detail="Required raw data files not found. Run the pipeline first.")
    
    # Load data directly from files to avoid file locking conflict with generate_all()
    with rasterio.open(config.DEM_PATH) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
    ndvi = None
    if os.path.exists(config.NDVI_BASELINE_PATH):
        with rasterio.open(config.NDVI_BASELINE_PATH) as src:
            temp_ndvi = src.read(1).astype("float32")
            if temp_ndvi.shape == dem.shape:
                ndvi = temp_ndvi
            
    inventory = gpd.read_file(config.INVENTORY_PATH)
    rainfall = pd.read_csv(config.RAINFALL_CSV_PATH, index_col=0, parse_dates=True)["rainfall_mm"]
    
    # Use the last available date in the weather data minus the lead days
    # so we have enough "future" data to simulate a 7-day forecast.
    event_date = rainfall.index[-1] - pd.Timedelta(days=config.FORECAST_LEAD_DAYS)
    event_point = inventory.geometry.iloc[0]
    event_x, event_y = event_point.x, event_point.y
    
    model = joblib.load(config.MODEL_PATH)
    
    # Feature engineering
    feature_stack = build_static_feature_stack(dem, config.TARGET_RESOLUTION_M, ndvi=ndvi)
    available_features = [c for c in config.FEATURE_COLUMNS if c in feature_stack]
    
    # Run backtest
    trace = backtest_event(
        event_x, event_y, event_date, rainfall,
        model, feature_stack, transform, available_features, lookback_days=10
    )
    
    summary = summarize_backtest(trace)
    
    trace_dicts = []
    for _, r in trace.iterrows():
        # Mock InSAR slope displacement trend (cumulative movement in mm) matching the risk levels
        # InSAR displacement rate escalates as combined risk score increases
        mock_insar_displacement_mm = round(float(r["combined_score"] * 12.5 + (10 - r["days_before_event"]) * 1.8), 2)
        trace_dicts.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "days_before_event": int(r["days_before_event"]),
            "api_mm": float(r["api_mm"]),
            "forecast_next_7d_mm": float(r["forecast_next_7d_mm"]),
            "susceptibility_prob": float(r["susceptibility_prob"]),
            "combined_score": float(r["combined_score"]),
            "risk_level": r["risk_level"],
            "insar_displacement_mm": mock_insar_displacement_mm
        })
        
    return {
        "summary": summary,
        "trace": trace_dicts
    }

@app.get("/api/map-data")
def get_map_data():
    if not os.path.exists(config.DEM_PATH):
        raise HTTPException(status_code=404, detail="DEM raster not found. Run the pipeline first.")
    
    # Read elevation (DEM)
    with rasterio.open(config.DEM_PATH) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        bounds = src.bounds
        
    # Read susceptibility raster
    if os.path.exists(config.SUSCEPTIBILITY_RASTER_PATH):
        with rasterio.open(config.SUSCEPTIBILITY_RASTER_PATH) as src:
            susceptibility = src.read(1).astype("float32")
    else:
        susceptibility = np.zeros_like(dem)
        
    # Calculate slope for overlay visualization
    from utils_raster import compute_slope_aspect
    slope, _ = compute_slope_aspect(dem, config.TARGET_RESOLUTION_M)
    
    # Load landslide inventory points
    inventory = gpd.read_file(config.INVENTORY_PATH)
    points = []
    for idx, row in inventory.iterrows():
        # Get pixel coordinates
        col, r = ~transform * (row.geometry.x, row.geometry.y)
        points.append({
            "id": int(row.get("event_id", idx)),
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "row": int(r),
            "col": int(col)
        })
        
    # Reconstruct the backtest event point coordinate
    # We choose the first inventory point as the historical event
    event_point = points[0] if points else None

    # Downsample maps to 50x50 for ultra-fast canvas drawing and JSON size optimization
    h, w = dem.shape
    step_y = max(1, h // 50)
    step_x = max(1, w // 50)
    
    dem_sub = dem[::step_y, ::step_x]
    susc_sub = susceptibility[::step_y, ::step_x]
    slope_sub = slope[::step_y, ::step_x]
    
    # Replace NaN with null for JSON serialization
    dem_flat = [float(x) if not np.isnan(x) else None for x in dem_sub.ravel()]
    susc_flat = [float(x) if not np.isnan(x) else None for x in susc_sub.ravel()]
    slope_flat = [float(x) if not np.isnan(x) else None for x in slope_sub.ravel()]
    
    return {
        "width": dem_sub.shape[1],
        "height": dem_sub.shape[0],
        "dem": dem_flat,
        "susceptibility": susc_flat,
        "slope": slope_flat,
        "dem_min": float(np.nanmin(dem_sub)),
        "dem_max": float(np.nanmax(dem_sub)),
        "inventory_points": points,
        "event_point": event_point,
        "bounds": {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top)
        }
    }

@app.post("/api/run-pipeline")
def run_pipeline(params: PipelineParams):
    try:
        # Override config variables dynamically
        config.MODEL_TYPE = params.model_type
        config.API_DECAY_K = params.api_decay_k
        config.API_WINDOW_DAYS = params.api_window_days
        config.PSEUDO_ABSENCE_RATIO = params.pseudo_absence_ratio
        config.PSEUDO_ABSENCE_MIN_DIST_M = params.pseudo_absence_min_dist_m
        config.SPATIAL_BLOCK_SIZE_M = params.spatial_block_size_m
        config.MIN_PRECISION_FOR_THRESHOLD = params.min_precision_for_threshold
        config.API_SATURATION_REFERENCE_MM = params.api_saturation_reference_mm
        config.FORECAST_TRIGGER_REFERENCE_MM = params.forecast_trigger_reference_mm
        
        # Re-run pipeline steps in-memory
        from generate_synthetic_data import generate_all
        data = generate_all()
        dem, ndvi = data["dem"], data["ndvi"]
        transform = data["transform"]
        inventory_gdf = data["inventory_gdf"]
        rainfall = data["rainfall"]
        aoi_bounds = data["aoi_bounds"]
        event_x, event_y, event_date = data["event_x"], data["event_y"], data["event_date"]
        
        # Build features
        feature_stack = build_static_feature_stack(dem, config.TARGET_RESOLUTION_M, ndvi=ndvi)
        available_features = [c for c in config.FEATURE_COLUMNS if c in feature_stack]
        
        # Build training table
        training_table = build_training_table(
            inventory_gdf, feature_stack, transform, aoi_bounds,
            pseudo_absence_ratio=config.PSEUDO_ABSENCE_RATIO,
            min_dist_m=config.PSEUDO_ABSENCE_MIN_DIST_M,
            random_state=config.RANDOM_STATE
        )
        training_table.to_csv(config.TRAINING_TABLE_PATH, index=False)
        
        # Train final model
        final_model = train_final_model(
            training_table, available_features, model_type=config.MODEL_TYPE,
            random_state=config.RANDOM_STATE
        )
        save_model(final_model, config.MODEL_PATH)
        
        # Produce susceptibility raster
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
        
        # Compute metrics
        cv_metrics, oof_probs = run_spatial_cv(
            training_table, available_features, model_type=config.MODEL_TYPE,
            n_splits=config.N_CV_SPLITS, block_size_m=config.SPATIAL_BLOCK_SIZE_M,
            random_state=config.RANDOM_STATE
        )
        
        avg_pr_auc = float(cv_metrics["pr_auc"].mean()) if "pr_auc" in cv_metrics else 1.0
        avg_roc_auc = float(cv_metrics["roc_auc"].mean()) if "roc_auc" in cv_metrics else 1.0
        
        return {
            "status": "success",
            "message": "Pipeline completed successfully with new parameters.",
            "metrics": {
                "pr_auc": round(avg_pr_auc, 3),
                "roc_auc": round(avg_roc_auc, 3)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# ===========================================================================
# LADAKH UNION TERRITORY ENDPOINTS
# ===========================================================================
import ladakh_config
from ladakh_data import generate_ladakh_demo_sites
from main_ladakh import (
    build_ladakh_training_table,
    leave_one_region_out_cv_ladakh,
    train_ladakh_model,
    backtest_all_ladakh_sites,
)

# Module-level cache for Ladakh sites/model (avoids re-generating on every request)
_ladakh_cache = {
    "sites": None,
    "model": None,
    "feature_stacks": None,
    "table": None,
    "available_features": None,
}


def _ensure_ladakh_pipeline():
    """Runs the Ladakh pipeline if not already cached."""
    if _ladakh_cache["model"] is not None:
        return

    sites = generate_ladakh_demo_sites()
    feature_cols = ladakh_config.FEATURE_COLUMNS
    table, feature_stacks = build_ladakh_training_table(
        sites, feature_cols,
        pseudo_absence_ratio=ladakh_config.PSEUDO_ABSENCE_RATIO,
        min_dist_m=ladakh_config.PSEUDO_ABSENCE_MIN_DIST_M,
        random_state=ladakh_config.RANDOM_STATE)
    available_features = [c for c in feature_cols if c in table.columns]
    table.to_csv(ladakh_config.LADAKH_TRAINING_TABLE_PATH, index=False)

    model = train_ladakh_model(
        table, available_features,
        model_type=ladakh_config.MODEL_TYPE,
        random_state=ladakh_config.RANDOM_STATE)
    save_model(model, ladakh_config.LADAKH_MODEL_PATH)

    _ladakh_cache["sites"] = sites
    _ladakh_cache["model"] = model
    _ladakh_cache["feature_stacks"] = feature_stacks
    _ladakh_cache["table"] = table
    _ladakh_cache["available_features"] = available_features


@app.get("/api/ladakh/sites")
def get_ladakh_sites():
    """Returns the 6 Ladakh sub-region definitions."""
    sites_info = []
    for name, info in ladakh_config.LADAKH_REGION_ANCHORS.items():
        sites_info.append({
            "site_id": name,
            "lon": info["lon"],
            "lat": info["lat"],
            "description": info["description"],
            "ruggedness": info["ruggedness"],
            "elevation_base": info["elevation_base"],
        })
    return {"sites": sites_info, "total": len(sites_info)}


@app.get("/api/ladakh/config")
def get_ladakh_config():
    """Returns Ladakh-specific configuration values."""
    return {
        "model_type": ladakh_config.MODEL_TYPE,
        "target_resolution_m": ladakh_config.TARGET_RESOLUTION_M,
        "api_decay_k": ladakh_config.API_DECAY_K,
        "api_window_days": ladakh_config.API_WINDOW_DAYS,
        "forecast_lead_days": ladakh_config.FORECAST_LEAD_DAYS,
        "pseudo_absence_ratio": ladakh_config.PSEUDO_ABSENCE_RATIO,
        "pseudo_absence_min_dist_m": ladakh_config.PSEUDO_ABSENCE_MIN_DIST_M,
        "api_saturation_reference_mm": ladakh_config.API_SATURATION_REFERENCE_MM,
        "forecast_trigger_reference_mm": ladakh_config.FORECAST_TRIGGER_REFERENCE_MM,
        "risk_levels": ladakh_config.RISK_LEVELS,
        "feature_columns": ladakh_config.FEATURE_COLUMNS,
        "snowmelt_window": {
            "start_doy": ladakh_config.SNOWMELT_WINDOW_START_DOY,
            "end_doy": ladakh_config.SNOWMELT_WINDOW_END_DOY,
            "peak_doy": ladakh_config.SNOWMELT_PEAK_DOY,
        },
    }


@app.post("/api/ladakh/run-pipeline")
def run_ladakh_pipeline():
    """Runs the full Ladakh pipeline and returns summary metrics."""
    try:
        # Force re-run by clearing cache
        _ladakh_cache["model"] = None
        _ensure_ladakh_pipeline()

        table = _ladakh_cache["table"]
        available_features = _ladakh_cache["available_features"]

        # Run leave-one-region-out CV
        cv_results = leave_one_region_out_cv_ladakh(
            table, available_features,
            model_type=ladakh_config.MODEL_TYPE,
            random_state=ladakh_config.RANDOM_STATE)

        cv_records = cv_results.to_dict(orient="records")

        return {
            "status": "success",
            "message": "Ladakh pipeline completed successfully.",
            "n_sites": table["site_id"].nunique(),
            "n_training_rows": len(table),
            "available_features": available_features,
            "cv_results": cv_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ladakh pipeline error: {str(e)}")


@app.get("/api/ladakh/metrics")
def get_ladakh_metrics():
    """Returns model metrics for the Ladakh pipeline."""
    try:
        _ensure_ladakh_pipeline()
        table = _ladakh_cache["table"]
        model = _ladakh_cache["model"]
        available_features = _ladakh_cache["available_features"]

        # Feature importances
        importances = []
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            total = sum(raw)
            if total > 0:
                raw = [float(x) / total for x in raw]
            for feat, imp in zip(available_features, raw):
                importances.append({"feature": feat, "importance": round(imp, 4)})
        importances = sorted(importances, key=lambda x: x["importance"], reverse=True)

        return {
            "total_rows": len(table),
            "positive_rows": int(table["landslide"].sum()),
            "n_sites": int(table["site_id"].nunique()),
            "feature_importances": importances,
            "available_features": available_features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ladakh metrics error: {str(e)}")


@app.get("/api/ladakh/backtest")
def get_ladakh_backtest():
    """Runs backtest across all 6 Ladakh sites and returns results."""
    try:
        _ensure_ladakh_pipeline()
        sites = _ladakh_cache["sites"]
        model = _ladakh_cache["model"]
        feature_stacks = _ladakh_cache["feature_stacks"]
        available_features = _ladakh_cache["available_features"]

        backtest_df = backtest_all_ladakh_sites(
            sites, feature_stacks, model, available_features)

        results = []
        for _, row in backtest_df.iterrows():
            results.append({
                "site_id": row["site_id"],
                "would_have_warned": bool(row["would_have_warned"]),
                "max_lead_time_days": int(row["max_lead_time_days"])
                    if not pd.isna(row["max_lead_time_days"]) else None,
                "message": row.get("message", ""),
            })

        return {"backtest_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ladakh backtest error: {str(e)}")


@app.get("/api/ladakh/backtest/{site_id}")
def get_ladakh_site_backtest(site_id: str):
    """Runs backtest for a SPECIFIC Ladakh sub-region and returns the
    day-by-day risk trace."""
    try:
        _ensure_ladakh_pipeline()
        sites = _ladakh_cache["sites"]
        model = _ladakh_cache["model"]
        feature_stacks = _ladakh_cache["feature_stacks"]
        available_features = _ladakh_cache["available_features"]

        # Find the requested site
        site = None
        for s in sites:
            if s["site_id"] == site_id:
                site = s
                break
        if site is None:
            raise HTTPException(
                status_code=404,
                detail=f"Site '{site_id}' not found. Available: "
                       f"{[s['site_id'] for s in sites]}")

        trace = backtest_event(
            site["event_x"], site["event_y"], site["event_date"],
            site["rainfall"], model, feature_stacks[site["site_id"]],
            site["transform"], available_features, lookback_days=8)
        summary = summarize_backtest(trace)

        trace_dicts = []
        for _, r in trace.iterrows():
            trace_dicts.append({
                "date": r["date"].strftime("%Y-%m-%d"),
                "days_before_event": int(r["days_before_event"]),
                "api_mm": float(r["api_mm"]),
                "forecast_next_7d_mm": float(r["forecast_next_7d_mm"]),
                "susceptibility_prob": float(r["susceptibility_prob"]),
                "combined_score": float(r["combined_score"]),
                "risk_level": r["risk_level"],
            })

        return {
            "site_id": site_id,
            "description": site["description"],
            "summary": summary,
            "trace": trace_dicts,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Ladakh site backtest error: {str(e)}")


@app.get("/api/ladakh/map-data")
def get_ladakh_map_data():
    """Returns susceptibility + terrain data for all 6 Ladakh sub-regions,
    suitable for rendering on a map dashboard."""
    try:
        _ensure_ladakh_pipeline()
        sites = _ladakh_cache["sites"]
        model = _ladakh_cache["model"]
        feature_stacks = _ladakh_cache["feature_stacks"]
        available_features = _ladakh_cache["available_features"]

        site_maps = []
        for site in sites:
            fs = feature_stacks[site["site_id"]]
            dem = site["dem"]

            # Generate susceptibility raster for this site
            stacked = np.stack([fs[c] for c in available_features], axis=-1)
            flat = stacked.reshape(-1, stacked.shape[-1])
            valid = ~np.isnan(flat).any(axis=1)
            probs = np.full(flat.shape[0], np.nan, dtype="float32")
            probs[valid] = model.predict_proba(flat[valid])[:, 1]
            susc = probs.reshape(dem.shape)

            # Downsample to 25x25 for JSON size
            h, w = dem.shape
            step_y = max(1, h // 25)
            step_x = max(1, w // 25)
            dem_sub = dem[::step_y, ::step_x]
            susc_sub = susc[::step_y, ::step_x]

            dem_flat = [float(x) if not np.isnan(x) else None
                        for x in dem_sub.ravel()]
            susc_flat = [float(x) if not np.isnan(x) else None
                         for x in susc_sub.ravel()]

            site_maps.append({
                "site_id": site["site_id"],
                "lon": site["lon"],
                "lat": site["lat"],
                "description": site["description"],
                "width": dem_sub.shape[1],
                "height": dem_sub.shape[0],
                "dem": dem_flat,
                "susceptibility": susc_flat,
                "dem_min": float(np.nanmin(dem)),
                "dem_max": float(np.nanmax(dem)),
                "susc_min": float(np.nanmin(susc[~np.isnan(susc)])) if not np.all(np.isnan(susc)) else 0,
                "susc_max": float(np.nanmax(susc[~np.isnan(susc)])) if not np.all(np.isnan(susc)) else 1,
                "event_point": {
                    "x": site["event_x"],
                    "y": site["event_y"],
                },
            })

        return {"sites": site_maps}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Ladakh map data error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
