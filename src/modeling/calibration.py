"""Post-hoc cutpoint calibration and cutpoint-integrated retuning for regression models.

Regression models predict continuous scores; ordinal MAE requires mapping to 0–5.
This module supports:

1. Post-hoc cutpoint search on OOF continuous predictions (hyperparams fixed).
2. Wrapping a fitted regressor so ``predict()`` uses fixed cutpoints during CV/tuning.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from modeling.metrics import (
    DEFAULT_REGRESSION_CUTPOINTS,
    compute_metrics,
    discretize_with_cutpoints,
    mae_from_continuous,
)
from modeling.registry import ORDINAL_MODELS, make_model_factory, resolve_training_data

RINT_CLIP_MODELS = {
    "linear_regression",
    "elasticnet_regression",
    "svr_regression",
    "ordinal_rf",
    "catboost_regressor",
}

DISCRETIZATION_DOCS = {
    "linear_regression": "np.rint + clip [0, 5] via clip_ordinal_predictions",
    "elasticnet_regression": "np.rint + clip [0, 5] via clip_ordinal_predictions",
    "svr_regression": "np.rint + clip [0, 5] via clip_ordinal_predictions",
    "ordinal_rf": "np.rint + clip [0, 5] via clip_ordinal_predictions",
    "catboost_regressor": "np.rint + clip [0, 5] via clip_ordinal_predictions",
}


def cutpoints_differ_from_default(cutpoints):
    """True when cutpoints are not equivalent to np.rint boundaries."""
    return not np.allclose(cutpoints, DEFAULT_REGRESSION_CUTPOINTS)


class CutpointRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrap a regressor so predict() uses fixed cutpoint discretization."""

    def __init__(self, base_estimator, cutpoints):
        self.base_estimator = base_estimator
        self.cutpoints = np.asarray(cutpoints, dtype=float)

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict_continuous(self, X):
        if hasattr(self.base_estimator, "predict_continuous"):
            return self.base_estimator.predict_continuous(X)
        return self.base_estimator.predict(X)

    def predict(self, X):
        return discretize_with_cutpoints(self.predict_continuous(X), self.cutpoints)


def make_model_factory_with_cutpoints(name, params, cutpoints, registry):
    """Return a factory that builds an inner model wrapped with fixed cutpoints.

    Used by ``tune_and_benchmark_model(..., cutpoints=...)`` so Optuna CV and
    test evaluation score ``discretize_with_cutpoints(continuous, cutpoints)``
    instead of rint.
    """
    base_factory = make_model_factory(name, params, registry)
    cutpoints_arr = np.asarray(cutpoints, dtype=float)

    def factory():
        return CutpointRegressorWrapper(base_factory(), cutpoints_arr)

    return factory


def collect_oof_continuous_predictions(
    model_factory,
    X,
    y,
    groups,
    n_splits=5,
    test_ids=None,
):
    """Stack out-of-fold continuous predictions on train/val (participant GroupKFold)."""
    gkf = GroupKFold(n_splits=n_splits)
    y_series = pd.Series(y).reset_index(drop=True)
    groups_series = pd.Series(groups).reset_index(drop=True)
    test_id_set = set(test_ids or [])

    y_parts = []
    cont_parts = []

    for train_idx, val_idx in gkf.split(X, y_series, groups_series):
        train_group_ids = set(groups_series.iloc[train_idx])
        val_group_ids = set(groups_series.iloc[val_idx])
        assert train_group_ids.isdisjoint(val_group_ids)
        if test_id_set:
            assert train_group_ids.isdisjoint(test_id_set)
            assert val_group_ids.isdisjoint(test_id_set)

        model = model_factory()
        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]

        model.fit(X_train, y_series.iloc[train_idx])
        y_parts.append(y_series.iloc[val_idx].to_numpy())
        cont_parts.append(model.predict_continuous(X_val))

    return np.concatenate(y_parts), np.concatenate(cont_parts)


def _is_valid_cutpoints(cutpoints, min_gap):
    cutpoints = np.asarray(cutpoints, dtype=float)
    if cutpoints.shape != (5,):
        return False
    if not np.all(np.diff(cutpoints) >= min_gap):
        return False
    return True


def tune_cutpoints_grid(
    y_true,
    y_cont,
    cutpoint_offsets=None,
    min_gap=0.05,
    n_passes=3,
):
    """Coordinate grid search for five monotonic cutpoints minimizing MAE."""
    if cutpoint_offsets is None:
        cutpoint_offsets = np.round(np.arange(-0.4, 0.41, 0.1), 1)

    best_cutpoints = DEFAULT_REGRESSION_CUTPOINTS.copy()
    best_mae = mae_from_continuous(y_true, y_cont, cutpoints=None)
    log_rows = []

    baseline_mae = best_mae
    log_rows.append(
        {
            "pass": 0,
            "index": -1,
            "offset": 0.0,
            "cutpoints": best_cutpoints.tolist(),
            "mae": baseline_mae,
            "delta_vs_best": 0.0,
        }
    )

    for pass_idx in range(1, n_passes + 1):
        improved = False
        for idx in range(5):
            for offset in cutpoint_offsets:
                candidate = best_cutpoints.copy()
                candidate[idx] = best_cutpoints[idx] + offset
                if not _is_valid_cutpoints(candidate, min_gap):
                    continue
                mae = mae_from_continuous(y_true, y_cont, cutpoints=candidate)
                delta = mae - best_mae
                log_rows.append(
                    {
                        "pass": pass_idx,
                        "index": idx,
                        "offset": float(offset),
                        "cutpoints": candidate.tolist(),
                        "mae": mae,
                        "delta_vs_best": delta,
                    }
                )
                if mae < best_mae - 1e-12:
                    best_mae = mae
                    best_cutpoints = candidate.copy()
                    improved = True
        if not improved:
            break

    search_log = pd.DataFrame(log_rows)
    return {
        "best_cutpoints": best_cutpoints,
        "best_mae": best_mae,
        "baseline_mae": baseline_mae,
        "search_log": search_log,
    }


def benchmark_cutpoint_calibration(
    name,
    bundle,
    params,
    registry=ORDINAL_MODELS,
    feature_set="base",
    n_splits=5,
):
    """Tune cutpoints on OOF continuous preds; return tuned OOF and test metrics."""
    if name not in RINT_CLIP_MODELS:
        raise ValueError(
            f"Cutpoint calibration applies to rint+clip models only; got {name!r}"
        )

    data_kwargs = resolve_training_data(name, bundle, feature_set=feature_set)
    factory = make_model_factory(name, params, registry)
    X_train_val = data_kwargs["X_train_val"]
    X_test = data_kwargs["X_test"]
    y_train_val = data_kwargs["y_train_val"]
    y_test = data_kwargs["y_test"]
    groups = data_kwargs["groups"]

    oof_y, oof_cont = collect_oof_continuous_predictions(
        factory,
        X_train_val,
        y_train_val,
        groups,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
    )
    tune_result = tune_cutpoints_grid(oof_y, oof_cont)
    best_cutpoints = tune_result["best_cutpoints"]

    oof_pred_tuned = discretize_with_cutpoints(oof_cont, best_cutpoints)
    oof_tuned_metrics = compute_metrics(oof_y, oof_pred_tuned)

    model = factory()
    model.fit(X_train_val, y_train_val)
    test_cont = model.predict_continuous(X_test)
    test_pred_tuned = discretize_with_cutpoints(test_cont, best_cutpoints)
    test_tuned_metrics = compute_metrics(y_test, test_pred_tuned)

    return {
        "best_cutpoints": best_cutpoints,
        "oof_tuned_metrics": oof_tuned_metrics,
        "test_tuned_metrics": test_tuned_metrics,
        "tune_result": tune_result,
    }
