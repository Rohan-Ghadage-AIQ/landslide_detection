"""
multi_site_pipeline.py
------------------------
Trains a model across MULTIPLE regions (not just one district), and
validates it the honest way for a "works anywhere" claim: by holding
whole regions out of training and testing on them -- i.e. does the
model generalize to a region it never saw during training, not just to
unseen points within a region it already learned from.

Each site's local tile becomes its own spatial-CV group automatically,
since sites are geographically far apart by construction (see
multi_site_data.py) -- this reuses GroupKFold directly rather than the
grid-block logic in spatial_cv.py, which was designed for many points
inside ONE contiguous district.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from feature_engineering import build_static_feature_stack
from sampling import extract_features_at_points, generate_pseudo_absences
from train_model import (
    train_xgboost, train_random_forest, apply_smote,
    optimize_threshold,
)
from validate_historical import backtest_event, summarize_backtest
import geopandas as gpd
from shapely.geometry import Point

import config


def build_site_training_rows(site, feature_cols, pseudo_absence_ratio=2,
                              min_dist_m=200, random_state=42):
    """
    Builds labeled training rows for ONE site: the known event point
    (positive) + pseudo-absences sampled from elsewhere in that site's
    local tile (negative). Returns a DataFrame tagged with site_id.
    """
    feature_stack = build_static_feature_stack(
        site["dem"], config.TARGET_RESOLUTION_M, ndvi=site["ndvi"])

    inventory_gdf = gpd.GeoDataFrame(
        {"landslide": [1]}, geometry=[Point(site["event_x"], site["event_y"])],
        crs=site["local_crs"])

    positives = extract_features_at_points(
        inventory_gdf, feature_stack, site["transform"], label_col="landslide")

    pseudo_absences = generate_pseudo_absences(
        inventory_gdf, site["aoi_bounds"], n_samples=pseudo_absence_ratio,
        min_dist_m=min_dist_m, random_state=random_state, crs=site["local_crs"])
    negatives = extract_features_at_points(
        pseudo_absences, feature_stack, site["transform"], label_col="landslide")

    rows = pd.concat([positives, negatives], ignore_index=True)
    rows["site_id"] = site["site_id"]
    return rows, feature_stack


def build_national_training_table(sites, feature_cols, **kwargs):
    """Runs build_site_training_rows across all sites and concatenates."""
    all_rows = []
    feature_stacks = {}
    for site in sites:
        rows, feature_stack = build_site_training_rows(site, feature_cols, **kwargs)
        all_rows.append(rows)
        feature_stacks[site["site_id"]] = feature_stack
    table = pd.concat(all_rows, ignore_index=True)
    return table, feature_stacks


def leave_one_region_out_cv(df, feature_cols, label_col="landslide",
                             model_type="xgboost", use_smote=True,
                             random_state=42):
    """
    The honest generalization test: for each region, train on ALL OTHER
    regions and test on the held-out one. Reports per-region metrics so
    you can see whether the model transfers to genuinely new terrain,
    not just new points within terrain it already learned from.
    """
    groups = df["site_id"].values
    unique_sites = df["site_id"].unique()
    n_splits = len(unique_sites)
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct sites/regions for "
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
            "note": "held-out region was NOT in training data for this fold",
        })

    return pd.DataFrame(results)


def train_national_model(df, feature_cols, label_col="landslide",
                          model_type="xgboost", use_smote=True,
                          random_state=42):
    """Final model trained on every region's data -- the one you deploy."""
    X = df[feature_cols].values
    y = df[label_col].values
    if use_smote:
        X, y = apply_smote(X, y, random_state)
    train_fn = train_xgboost if model_type == "xgboost" else train_random_forest
    return train_fn(X, y, random_state)


def backtest_all_sites(sites, feature_stacks, model, feature_cols,
                        lookback_days=10):
    """Runs the historical-event backtest independently at every site,
    using each site's own local feature stack, transform, and rainfall
    record -- exactly the same backtest logic used for the single-district
    Pune pipeline, just looped across regions."""
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
