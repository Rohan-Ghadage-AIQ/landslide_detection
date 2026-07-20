"""
train_model.py
---------------
Trains Random Forest / XGBoost classifiers with class-imbalance handling,
evaluates with spatial cross-validation, and optimizes the decision
threshold for recall-first, disaster-management-relevant performance.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    recall_score, precision_score,
)
from xgboost import XGBClassifier

from spatial_cv import spatial_kfold_splits

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    counts = np.bincount(y_train.astype(int))
    neg, pos = counts[0], counts[1] if len(counts) > 1 else 1
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(neg / max(pos, 1), 1.0),
        eval_metric="aucpr", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def apply_smote(X_train, y_train, random_state=42):
    if not HAS_SMOTE:
        return X_train, y_train
    n_pos = int((y_train == 1).sum())
    if n_pos < 2:
        return X_train, y_train  # SMOTE needs at least a few positives
    k = min(5, n_pos - 1)
    sm = SMOTE(random_state=random_state, k_neighbors=max(k, 1))
    return sm.fit_resample(X_train, y_train)


def optimize_threshold(y_true, probs, min_precision=0.3):
    """
    Picks the probability threshold that maximizes recall subject to a
    minimum acceptable precision (a business decision, not a purely
    statistical one -- see the R&D overview doc, Section 8).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    valid = precision[:-1] >= min_precision
    if not valid.any():
        return 0.5, {"precision": None, "recall": None}
    candidate_recalls = recall[:-1][valid]
    best_local_idx = int(np.argmax(candidate_recalls))
    best_threshold = thresholds[valid][best_local_idx]
    return float(best_threshold), {
        "precision": float(precision[:-1][valid][best_local_idx]),
        "recall": float(candidate_recalls[best_local_idx]),
    }


def run_spatial_cv(df, feature_cols, label_col="landslide",
                    model_type="xgboost", n_splits=5, block_size_m=5000,
                    use_smote=True, random_state=42):
    """
    Runs spatial k-fold CV, training a fresh model per fold, and returns
    per-fold metrics plus out-of-fold predictions for overall evaluation.
    """
    splits = spatial_kfold_splits(df, feature_cols, label_col, n_splits,
                                   block_size_m)
    fold_metrics = []
    oof_probs = np.full(len(df), np.nan)

    train_fn = train_xgboost if model_type == "xgboost" else train_random_forest

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train = df.iloc[train_idx][feature_cols].values
        y_train = df.iloc[train_idx][label_col].values
        X_test = df.iloc[test_idx][feature_cols].values
        y_test = df.iloc[test_idx][label_col].values

        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train, random_state)

        if len(np.unique(y_train)) < 2:
            continue  # degenerate fold (demo data) -- skip

        model = train_fn(X_train, y_train, random_state)
        probs = model.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = probs

        if len(np.unique(y_test)) < 2:
            fold_metrics.append({"fold": fold, "pr_auc": np.nan,
                                  "roc_auc": np.nan, "n_test": len(test_idx)})
            continue

        fold_metrics.append({
            "fold": fold,
            "pr_auc": average_precision_score(y_test, probs),
            "roc_auc": roc_auc_score(y_test, probs),
            "n_test": len(test_idx),
        })

    return pd.DataFrame(fold_metrics), oof_probs


def train_final_model(df, feature_cols, label_col="landslide",
                       model_type="xgboost", use_smote=True,
                       random_state=42):
    """Trains on ALL available labeled data -- the model that gets deployed."""
    X = df[feature_cols].values
    y = df[label_col].values
    if use_smote:
        X, y = apply_smote(X, y, random_state)
    train_fn = train_xgboost if model_type == "xgboost" else train_random_forest
    return train_fn(X, y, random_state)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
