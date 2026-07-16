"""Evaluation metrics for ordinal and multiclass tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def clip_ordinal_predictions(preds, low=0, high=5):
    return np.clip(np.rint(preds), low, high).astype(int)


def compute_metrics(y_true, y_pred, task="ordinal"):
    y_true = np.asarray(y_true)
    if task == "ordinal":
        y_pred = clip_ordinal_predictions(y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": rmse,
            "r2": r2_score(y_true, y_pred),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        }
    if task == "multiclass":
        y_pred = clip_ordinal_predictions(y_pred)
        return {
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }
    raise ValueError(f"Unknown task: {task}")


def summarize_fold_metrics(fold_df, metric_cols):
    summary = {
        "mean": fold_df[metric_cols].mean(),
        "std": fold_df[metric_cols].std(),
    }
    import pandas as pd

    return pd.DataFrame(summary).T
