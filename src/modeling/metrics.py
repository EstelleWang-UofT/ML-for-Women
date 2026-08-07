"""Evaluation metrics for ordinal fatigue models."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score

ORDINAL_METRIC_COLS = ["mae", "rmse", "r2", "qwk"]
REGRESSION_METRIC_COLS = ["mae", "rmse", "r2"]


def clip_ordinal_predictions(preds, low=0, high=5):
    return np.clip(np.rint(preds), low, high).astype(int)


def compute_regression_metrics(y_true, y_pred):
    """MAE / RMSE / R² on continuous predictions vs integer fatigue labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


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
