"""
main_ladakh.py
----------------
Entry point for the LADAKH UNION TERRITORY landslide early-warning pipeline.

Trains a model across 6 landslide-prone sub-regions in Leh and Kargil
districts and validates it with leave-one-region-out cross-validation
plus per-site historical-event backtesting.

This is the Ladakh-specific counterpart of main_national.py: same
multi-site architecture, but tailored to Ladakh's high-altitude arid
terrain, cloudburst triggers, and freeze-thaw / snowmelt dynamics.

Run as-is for a synthetic 6-region Ladakh demo:
    python main_ladakh.py

For real data: swap ladakh_data.generate_ladakh_demo_sites() with a
loader that reads real Copernicus GLO-30 DEM tiles + GSI/COOLR inventory
points + IMD rainfall — see implementation_plan.md Phase 2.
"""

import numpy as np
import pandas as pd

import config
import ladakh_config
from ladakh_data import generate_ladakh_demo_sites, _snowmelt_proxy
from feature_engineering import build_static_feature_stack
from sampling import extract_features_at_points, generate_pseudo_absences
from train_model import (
    train_xgboost, train_random_forest, apply_smote, save_model,
)
from validate_historical import backtest_event, summarize_backtest

import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import GroupKFold


def build_ladakh_site_training_rows(site, feature_cols,
                                     pseudo_absence_ratio=2,
                                     min_dist_m=200, random_state=42):
    """
    Builds labeled training rows for ONE Ladakh site, including the
    snowmelt_proxy as an extra feature beyond the standard terrain stack.
    """
    # Standard terrain features from DEM
    feature_stack = build_static_feature_stack(
        site["dem"], ladakh_config.TARGET_RESOLUTION_M, ndvi=site["ndvi"])

    # Add Ladakh-specific snowmelt proxy to the feature stack
    if "snowmelt_proxy" in site and "snowmelt_proxy" in feature_cols:
        feature_stack["snowmelt_proxy"] = site["snowmelt_proxy"]

    # Create inventory GeoDataFrame with the single known event point
    inventory_gdf = gpd.GeoDataFrame(
        {"landslide": [1]},
        geometry=[Point(site["event_x"], site["event_y"])],
        crs=site["local_crs"])

    # Extract features at the positive (landslide) point
    positives = extract_features_at_points(
        inventory_gdf, feature_stack, site["transform"],
        label_col="landslide")

    # Generate pseudo-absence (non-landslide) points
    pseudo_absences = generate_pseudo_absences(
        inventory_gdf, site["aoi_bounds"],
        n_samples=pseudo_absence_ratio,
        min_dist_m=min_dist_m,
        random_state=random_state,
        crs=site["local_crs"])
    negatives = extract_features_at_points(
        pseudo_absences, feature_stack, site["transform"],
        label_col="landslide")

    rows = pd.concat([positives, negatives], ignore_index=True)
    rows["site_id"] = site["site_id"]
    return rows, feature_stack


def build_ladakh_training_table(sites, feature_cols, **kwargs):
    """Builds the combined training table across all Ladakh sites."""
    all_rows = []
    feature_stacks = {}
    for site in sites:
        rows, feature_stack = build_ladakh_site_training_rows(
            site, feature_cols, **kwargs)
        all_rows.append(rows)
        feature_stacks[site["site_id"]] = feature_stack
    table = pd.concat(all_rows, ignore_index=True)
    return table, feature_stacks


def leave_one_region_out_cv_ladakh(df, feature_cols, label_col="landslide",
                                    model_type="xgboost", use_smote=True,
                                    random_state=42):
    """
    Leave-one-region-out CV for Ladakh sub-regions: train on 5 regions,
    test on the 6th. Measures geographic generalization across Ladakh's
    diverse terrain types (river valleys vs high passes vs gorges).
    """
    groups = df["site_id"].values
    unique_sites = df["site_id"].unique()
    n_splits = len(unique_sites)
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct sites for "
                         "leave-one-region-out validation.")

    gkf = GroupKFold(n_splits=n_splits)
    X = df[feature_cols].values
    y = df[label_col].values

    results = []
    train_fn = train_xgboost if model_type == "xgboost" else train_random_forest

    for train_idx, test_idx in gkf.split(X, y, groups):
        held_out_region = df.iloc[test_idx]["site_id"].iloc[0]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train, random_state)
        if len(np.unique(y_train)) < 2:
            continue

        model = train_fn(X_train, y_train, random_state)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "held_out_region": held_out_region,
            "n_test_points": len(test_idx),
            "mean_predicted_prob": float(np.mean(probs)),
            "note": "held-out region NOT in training data for this fold",
        })

    return pd.DataFrame(results)


def train_ladakh_model(df, feature_cols, label_col="landslide",
                        model_type="xgboost", use_smote=True,
                        random_state=42):
    """Final Ladakh model trained on all 6 regions."""
    X = df[feature_cols].values
    y = df[label_col].values
    if use_smote:
        X, y = apply_smote(X, y, random_state)
    train_fn = train_xgboost if model_type == "xgboost" else train_random_forest
    return train_fn(X, y, random_state)


def backtest_all_ladakh_sites(sites, feature_stacks, model, feature_cols,
                               lookback_days=8):
    """
    Runs the historical-event backtest at every Ladakh site.
    Uses a shorter lookback (8 days) than the monsoon pipeline (10 days)
    because Ladakh cloudburst events develop faster.
    """
    summaries = []
    for site in sites:
        trace = backtest_event(
            site["event_x"], site["event_y"], site["event_date"],
            site["rainfall"], model, feature_stacks[site["site_id"]],
            site["transform"], feature_cols, lookback_days=lookback_days)
        summary = summarize_backtest(trace)
        summary["site_id"] = site["site_id"]
        summaries.append(summary)
    return pd.DataFrame(summaries)


def main():
    print("=" * 70)
    print("LADAKH UNION TERRITORY — LANDSLIDE EARLY-WARNING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 1. Generate / load Ladakh sites
    # -------------------------------------------------------------------
    print("\n[1/5] Generating synthetic Ladakh sub-region demo sites "
          "(replace with real data for production)...")
    sites = generate_ladakh_demo_sites()
    for s in sites:
        print(f"    - {s['site_id']:25s}  CRS={s['local_crs']}  "
              f"event={s['event_date'].date()}  "
              f"desc: {s['description']}")

    # -------------------------------------------------------------------
    # 2. Build training table across all 6 sub-regions
    # -------------------------------------------------------------------
    print("\n[2/5] Building training rows for each Ladakh sub-region "
          "(1 positive + pseudo-absences per site)...")
    feature_cols = ladakh_config.FEATURE_COLUMNS
    table, feature_stacks = build_ladakh_training_table(
        sites, feature_cols,
        pseudo_absence_ratio=ladakh_config.PSEUDO_ABSENCE_RATIO,
        min_dist_m=ladakh_config.PSEUDO_ABSENCE_MIN_DIST_M,
        random_state=ladakh_config.RANDOM_STATE)

    available_features = [c for c in feature_cols if c in table.columns]
    print(f"    Combined training table: {len(table)} rows across "
          f"{table['site_id'].nunique()} sub-regions")
    print(f"    Features available: {available_features}")

    # Save training table
    table.to_csv(ladakh_config.LADAKH_TRAINING_TABLE_PATH, index=False)
    print(f"    Training table saved to {ladakh_config.LADAKH_TRAINING_TABLE_PATH}")

    # -------------------------------------------------------------------
    # 3. Leave-one-region-out cross-validation
    # -------------------------------------------------------------------
    print("\n[3/5] Leave-one-region-out cross-validation "
          "(train on 5 sub-regions, test on held-out)...")
    cv_results = leave_one_region_out_cv_ladakh(
        table, available_features,
        model_type=ladakh_config.MODEL_TYPE,
        random_state=ladakh_config.RANDOM_STATE)
    print(cv_results.to_string(index=False))
    print("\n    NOTE: with only ~1 labeled event per synthetic region, "
          "these numbers illustrate the MECHANISM only — real "
          "generalization testing needs many events per region.")

    # -------------------------------------------------------------------
    # 4. Train the final Ladakh model on all sub-regions
    # -------------------------------------------------------------------
    print("\n[4/5] Training the final Ladakh model on all sub-regions...")
    ladakh_model = train_ladakh_model(
        table, available_features,
        model_type=ladakh_config.MODEL_TYPE,
        random_state=ladakh_config.RANDOM_STATE)
    save_model(ladakh_model, ladakh_config.LADAKH_MODEL_PATH)
    print(f"    Model saved to {ladakh_config.LADAKH_MODEL_PATH}")

    # -------------------------------------------------------------------
    # 5. Backtest against each sub-region's historical event
    # -------------------------------------------------------------------
    print("\n[5/5] Backtesting the trained model against each "
          "sub-region's historical event...")
    backtest_summary = backtest_all_ladakh_sites(
        sites, feature_stacks, ladakh_model, available_features)
    print(backtest_summary[["site_id", "would_have_warned",
                             "max_lead_time_days"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("LADAKH PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
