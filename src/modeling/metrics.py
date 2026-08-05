"""Evaluation metrics for ordinal fatigue models."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score

ORDINAL_METRIC_COLS = ["mae", "rmse", "r2", "qwk"]

# Cutpoints equivalent to np.rint for classes 0–5 (boundaries at k + 0.5).
DEFAULT_REGRESSION_CUTPOINTS = np.array([0.5, 1.5, 2.5, 3.5, 4.5])


def clip_ordinal_predictions(preds, low=0, high=5):
    return np.clip(np.rint(preds), low, high).astype(int)


def discretize_with_cutpoints(preds, cutpoints):
    """Map continuous scores to integers 0–5 using five ascending cutpoints."""
    cutpoints = np.asarray(cutpoints, dtype=float)
    if cutpoints.shape != (5,):
        raise ValueError(f"cutpoints must have length 5, got {cutpoints.shape}")
    return np.searchsorted(cutpoints, np.asarray(preds, dtype=float), side="right").astype(
        int
    )


def mae_from_continuous(y_true, y_cont, cutpoints=None):
    """MAE after discretizing continuous predictions (None => rint + clip)."""
    y_true = np.asarray(y_true)
    if cutpoints is None:
        y_pred = clip_ordinal_predictions(y_cont)
    else:
        y_pred = discretize_with_cutpoints(y_cont, cutpoints)
    return float(mean_absolute_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = clip_ordinal_predictions(y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def summarize_fold_metrics(fold_df, metric_cols):
    summary = {
        "mean": fold_df[metric_cols].mean(),
        "std": fold_df[metric_cols].std(),
    }
    import pandas as pd

    return pd.DataFrame(summary).T
