"""Simple baseline predictors for benchmark comparison."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from modeling.config import N_CV_FOLDS
from modeling.data import attach_prior_frame
from modeling.metrics import ORDINAL_METRIC_COLS, clip_ordinal_predictions


class GlobalMeanBaseline(BaseEstimator):
    """Predict the rounded training-set mean target for every row."""

    def fit(self, X, y):
        del X
        self.value_ = int(np.rint(np.mean(y)))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=int)


class GlobalModeBaseline(BaseEstimator):
    """Predict the training-set mode target for every row."""

    def fit(self, X, y):
        del X
        values, counts = np.unique(y, return_counts=True)
        self.value_ = int(values[np.argmax(counts)])
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=int)


class PriorValueBaseline(BaseEstimator):
    """Predict a precomputed prior target column; NaN values use a training fallback."""

    prior_col = "__prior__"

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} expects a DataFrame with {self.prior_col}.")
        prior = pd.to_numeric(X[self.prior_col], errors="coerce")
        if prior.notna().any():
            self.fallback_ = int(np.rint(prior.dropna().mean()))
        else:
            self.fallback_ = int(np.rint(np.mean(y)))
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} expects a DataFrame with {self.prior_col}.")
        prior = pd.to_numeric(X[self.prior_col], errors="coerce")
        preds = prior.fillna(self.fallback_).round().astype(int)
        return clip_ordinal_predictions(preds.to_numpy())


class Lag1FatigueBaseline(PriorValueBaseline):
    """Predict previous-day fatigue (persistence baseline)."""


class ExpandingMeanBaseline(PriorValueBaseline):
    """Predict the participant's expanding mean fatigue from prior days."""


def build_global_mean_baseline(**kwargs):
    del kwargs
    return GlobalMeanBaseline()


def build_global_mode_baseline(**kwargs):
    del kwargs
    return GlobalModeBaseline()


def build_lag1_baseline(**kwargs):
    del kwargs
    return Lag1FatigueBaseline()


def build_expanding_mean_baseline(**kwargs):
    del kwargs
    return ExpandingMeanBaseline()


ORDINAL_BASELINES = {
    "global_mean": build_global_mean_baseline,
    "global_mode": build_global_mode_baseline,
    "lag1_fatigue": build_lag1_baseline,
    "expanding_mean": build_expanding_mean_baseline,
}


def summarize_baseline_metrics(results):
    rows = []
    for result in results:
        row = {"model": result["name"]}
        cv_mean = result["cv_summary"].loc["mean", ORDINAL_METRIC_COLS]
        row.update({f"cv_{k}": cv_mean[k] for k in ORDINAL_METRIC_COLS})
        row.update({f"test_{k}": result["test_metrics"][k] for k in ORDINAL_METRIC_COLS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _run_persistence_baseline(name, builder, bundle, n_splits):
    from modeling.cv import run_model_benchmark

    if name == "lag1_fatigue":
        X_train_val = attach_prior_frame(bundle.X_train_val, bundle.y_lag1_train_val)
        X_test = attach_prior_frame(bundle.X_test, bundle.y_lag1_test)
    elif name == "expanding_mean":
        X_train_val = attach_prior_frame(bundle.X_train_val, bundle.y_expanding_train_val)
        X_test = attach_prior_frame(bundle.X_test, bundle.y_expanding_test)
    else:
        raise ValueError(f"Unknown persistence baseline: {name}")

    return run_model_benchmark(
        name=name,
        model_factory=builder,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
        X_train_val=X_train_val,
        X_test=X_test,
        y_train_val=bundle.y_ord_train_val,
        y_test=bundle.y_ord_test,
        groups=bundle.groups_train_val,
    )


def run_baseline_benchmark(name, builder, bundle, n_splits=N_CV_FOLDS):
    """Evaluate one baseline with the same CV / test protocol as tuned models."""
    from modeling.cv import run_model_benchmark

    factory = builder

    if name in {"lag1_fatigue", "expanding_mean"}:
        return _run_persistence_baseline(name, builder, bundle, n_splits)

    return run_model_benchmark(
        name=name,
        model_factory=factory,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
        X_train_val=bundle.X_train_val,
        X_test=bundle.X_test,
        y_train_val=bundle.y_ord_train_val,
        y_test=bundle.y_ord_test,
        groups=bundle.groups_train_val,
    )


def run_all_baseline_benchmarks(bundle, n_splits=N_CV_FOLDS):
    results = []
    for name, builder in ORDINAL_BASELINES.items():
        result = run_baseline_benchmark(name, builder, bundle, n_splits=n_splits)
        result["best_params"] = {}
        results.append(result)
    return results
