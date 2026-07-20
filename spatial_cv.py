"""
spatial_cv.py
-------------
Spatial (block-based) cross-validation, to avoid the optimistic bias that
plain random k-fold CV produces on spatially autocorrelated data.
"""

import numpy as np
from sklearn.model_selection import GroupKFold


def assign_spatial_blocks(df, block_size_m=5000, x_col="x", y_col="y"):
    """
    Assigns every row a block id based on a coarse spatial grid. Used as
    the CV 'group' label so no fold splits within a block.
    """
    df = df.copy()
    df["block_x"] = (df[x_col] // block_size_m).astype(int)
    df["block_y"] = (df[y_col] // block_size_m).astype(int)
    df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)
    return df


def spatial_kfold_splits(df, feature_cols, label_col="landslide", n_splits=5,
                          block_size_m=5000):
    """Returns a list of (train_idx, test_idx) index arrays."""
    df = assign_spatial_blocks(df, block_size_m)
    n_blocks = df["block_id"].nunique()
    effective_splits = min(n_splits, n_blocks) if n_blocks > 1 else 1
    if effective_splits < 2:
        # Not enough distinct spatial blocks for CV (e.g. tiny demo data) --
        # fall back to a single train/test split so the pipeline still runs.
        idx = np.arange(len(df))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        split_point = int(len(idx) * 0.8)
        return [(idx[:split_point], idx[split_point:])]

    gkf = GroupKFold(n_splits=effective_splits)
    X = df[feature_cols].values
    y = df[label_col].values
    groups = df["block_id"].values
    return list(gkf.split(X, y, groups))
