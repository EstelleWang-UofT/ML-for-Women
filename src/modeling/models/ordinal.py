"""Ordinal model builders."""

import warnings

import numpy as np
import pandas as pd
import mord
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm

from modeling.config import (
    CATEGORICAL_FEATURES,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    PHASE_TO_IDX,
    RANDOM_STATE,
)
from modeling.metrics import clip_ordinal_predictions

PARTICIPANT_CONSTANT_COLS = {
    "age_of_first_menarche",
    "age",
    "menstrual_health_literacy_num",
    "sexually_active_num",
}
DAY_VARYING_CATEGORICAL = ["is_weekend", "phase", "exerciselevel_num"]
FIT_METHODS = ("lbfgs", "bfgs")
NUM_ORDINAL_CLASSES = 6
NUM_ORDINAL_THRESHOLDS = NUM_ORDINAL_CLASSES - 1


def _has_history_features(X):
    return all(col in X.columns for col in HISTORY_FEATURES)


def _base_preprocessor(include_history=False):
    numeric_features = list(NUMERIC_FEATURES)
    if include_history:
        numeric_features = numeric_features + list(HISTORY_FEATURES)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


class RidgeOrdinalEstimator(BaseEstimator):
    """Ridge on scaled/OHE daily features with clipped ordinal predictions."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prep_ = None
        self.model_ = None

    def fit(self, X, y):
        self.prep_ = _base_preprocessor(include_history=_has_history_features(X))
        Xt = self.prep_.fit_transform(X)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self.prep_.transform(X)
        return clip_ordinal_predictions(self.model_.predict(Xt))


class OrderedLogisticEstimator(BaseEstimator):
    """mord all-threshold logistic on scaled/OHE daily features."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.prep_ = None
        self.model_ = None

    def fit(self, X, y):
        self.prep_ = _base_preprocessor(include_history=_has_history_features(X))
        Xt = self.prep_.fit_transform(X)
        self.model_ = mord.LogisticAT(alpha=self.alpha)
        self.model_.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self.prep_.transform(X)
        return self.model_.predict(Xt)


def build_linear_regression(alpha=1.0, **kwargs):
    del kwargs
    return RidgeOrdinalEstimator(alpha=alpha)


def build_ordered_logistic(alpha=1.0, **kwargs):
    del kwargs
    return OrderedLogisticEstimator(alpha=alpha)


def build_ordinal_rf(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    **kwargs,
):
    del kwargs
    return OrdinalRegressorWrapper(
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )


def build_catboost_regressor(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    **kwargs,
):
    del kwargs
    return OrdinalRegressorWrapper(
        _make_catboost_regressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
        )
    )


def build_ordinal_forest(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    **kwargs,
):
    del kwargs
    return CumulativeOrdinalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )


def build_catboost_ordinal(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    **kwargs,
):
    del kwargs
    return CatBoostOrdinalWrapper(
        _make_catboost_classifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
        ),
        loss_mode="multiclass",
    )


def _make_catboost_regressor(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=RANDOM_STATE,
        verbose=0,
        train_dir=None,
    )


def _make_catboost_classifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
):
    return CatBoostClassifier(
        loss_function="MultiClass",
        classes_count=NUM_ORDINAL_CLASSES,
        auto_class_weights="Balanced",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=RANDOM_STATE,
        verbose=0,
        train_dir=None,
    )


def build_population_ordered_logistic(maxiter=400, **kwargs):
    del kwargs
    return PopulationOrderedLogisticWrapper(maxiter=maxiter)


class OrdinalRegressorWrapper(BaseEstimator):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        preds = self.regressor.predict(X)
        return clip_ordinal_predictions(preds)


class CumulativeOrdinalForest(BaseEstimator):
    """All-threshold random forest ensemble for ordered fatigue classes."""

    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.threshold_models_ = None

    def _make_threshold_classifier(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

    @staticmethod
    def _positive_class_prob(model, X):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(X)) if 0 in classes else np.ones(len(X))
        return proba[:, classes.index(1)]

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.threshold_models_ = []
        for threshold in range(NUM_ORDINAL_THRESHOLDS):
            y_gt = (y > threshold).astype(int)
            model = self._make_threshold_classifier()
            model.fit(X, y_gt)
            self.threshold_models_.append(model)
        return self

    def _prob_greater_than(self, X):
        n_samples = len(X)
        prob_gt = np.zeros((n_samples, NUM_ORDINAL_CLASSES), dtype=float)
        for threshold, model in enumerate(self.threshold_models_):
            prob_gt[:, threshold] = self._positive_class_prob(model, X)
        prob_gt[:, NUM_ORDINAL_THRESHOLDS] = 0.0
        for threshold in range(1, NUM_ORDINAL_CLASSES):
            prob_gt[:, threshold] = np.minimum(prob_gt[:, threshold], prob_gt[:, threshold - 1])
        return prob_gt

    def predict(self, X):
        prob_gt = self._prob_greater_than(X)
        prob_le = np.ones((prob_gt.shape[0], NUM_ORDINAL_CLASSES + 1), dtype=float)
        prob_le[:, 1:] = prob_gt
        class_probs = np.maximum(prob_le[:, :-1] - prob_le[:, 1:], 0.0)
        expected = class_probs @ np.arange(NUM_ORDINAL_CLASSES)
        return clip_ordinal_predictions(expected)


class CatBoostOrdinalWrapper(BaseEstimator):
    """CatBoost ordinal model with RMSE regression or MultiClass + expected value."""

    def __init__(self, model, loss_mode="rmse"):
        self.model = model
        self.loss_mode = loss_mode

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.loss_mode == "multiclass":
            probs = self.model.predict_proba(X)
            classes = np.asarray(self.model.classes_, dtype=float)
            expected = probs @ classes
            return clip_ordinal_predictions(expected)
        preds = self.model.predict(X)
        return clip_ordinal_predictions(preds)


class PopulationOrderedLogisticModel(BaseEstimator, ClassifierMixin):
    """Population-level ordered logistic on day-varying features (statsmodels OrderedModel)."""

    def __init__(self, maxiter=400):
        self.maxiter = maxiter
        self.result_ = None
        self.feature_columns_ = None
        self.fill_values_ = None
        self.continuous_columns_ = None
        self.feature_means_ = None
        self.feature_stds_ = None
        self.classes_ = np.arange(6)
        self.converged_ = False
        self.fit_method_ = None

    def _build_exog_raw(self, X, groups=None):
        del groups
        day_varying_num = [
            c for c in NUMERIC_FEATURES if c not in PARTICIPANT_CONSTANT_COLS and c in X.columns
        ]
        exog = X[day_varying_num].apply(pd.to_numeric, errors="coerce").astype(float)
        for col in DAY_VARYING_CATEGORICAL:
            if col not in X.columns:
                continue
            if col == "phase":
                exog[col] = X["phase"].astype(str).map(PHASE_TO_IDX).fillna(0).astype(float)
            else:
                exog[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        history_cols = [col for col in HISTORY_FEATURES if col in X.columns]
        if history_cols:
            exog = pd.concat(
                [exog, X[history_cols].apply(pd.to_numeric, errors="coerce").astype(float)],
                axis=1,
            )
        return exog.reset_index(drop=True)

    def _continuous_columns(self, columns):
        return list(columns)

    def _prepare_exog(self, exog_raw, fit=False):
        exog = exog_raw.copy()
        if fit:
            self.feature_columns_ = exog.columns.tolist()
            self.fill_values_ = exog.median(numeric_only=True)
            self.continuous_columns_ = self._continuous_columns(self.feature_columns_)
            self.feature_means_ = exog[self.continuous_columns_].mean()
            self.feature_stds_ = exog[self.continuous_columns_].std().replace(0, 1.0)
        else:
            for col in self.feature_columns_:
                if col not in exog.columns:
                    exog[col] = np.nan
            exog = exog[self.feature_columns_]

        exog = exog.fillna(self.fill_values_)
        exog[self.continuous_columns_] = (
            exog[self.continuous_columns_] - self.feature_means_
        ) / self.feature_stds_
        return exog

    def _fit_ordered_model(self, endog, exog):
        last_result = None
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for method in FIT_METHODS:
                model = OrderedModel(endog, exog, distr="logit")
                try:
                    result = model.fit(method=method, maxiter=self.maxiter, disp=False)
                except Exception:
                    continue
                last_result = result
                converged = bool(result.mle_retvals.get("converged", False))
                if converged:
                    self.converged_ = True
                    self.fit_method_ = method
                    return result
        if last_result is None:
            raise RuntimeError("PopulationOrderedLogisticModel failed for all fit methods.")
        self.converged_ = bool(last_result.mle_retvals.get("converged", False))
        self.fit_method_ = "fallback"
        return last_result

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("PopulationOrderedLogisticModel requires participant groups.")
        exog_raw = self._build_exog_raw(X, groups=groups)
        exog = self._prepare_exog(exog_raw, fit=True)
        endog = pd.Series(y).astype(int).reset_index(drop=True)
        if endog.nunique() < 2:
            raise ValueError("Need at least two ordinal classes in training fold.")
        self.result_ = self._fit_ordered_model(endog, exog)
        return self

    def predict(self, X, groups=None):
        if groups is None:
            raise ValueError("PopulationOrderedLogisticModel requires participant groups on predict.")
        if self.feature_columns_ is None or self.result_ is None:
            raise ValueError("PopulationOrderedLogisticModel has not been fitted yet.")

        exog_raw = self._build_exog_raw(X, groups=groups)
        exog = self._prepare_exog(exog_raw, fit=False)
        probs = self.result_.predict(exog)
        if isinstance(probs, pd.DataFrame):
            return probs.to_numpy().argmax(axis=1).astype(int)
        probs = np.asarray(probs)
        if probs.ndim == 1:
            return probs.astype(int)
        return probs.argmax(axis=1).astype(int)


class PopulationOrderedLogisticWrapper(BaseEstimator):
    """Sklearn wrapper that carries groups alongside feature columns."""

    def __init__(self, maxiter=400):
        self.maxiter = maxiter
        self.model_ = PopulationOrderedLogisticModel(maxiter=maxiter)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        groups = X["__group__"].values
        features = X.drop(columns=["__group__"])
        self.model_.fit(features, y, groups=groups)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        groups = X["__group__"].values
        features = X.drop(columns=["__group__"])
        return self.model_.predict(features, groups=groups)


def attach_groups(X: pd.DataFrame, groups) -> pd.DataFrame:
    out = X.copy()
    out["__group__"] = groups.values if hasattr(groups, "values") else groups
    return out


GEE_META_COLS = ["__group__", "__day_in_study__"]
GEE_SUMMARY_HEADER = (
    "GEE clusters: one participant-interval (id × study_interval); "
    "Autoregressive working correlation; rows sorted by day_in_study within cluster.\n\n"
)


def attach_gee_features(X: pd.DataFrame, groups, study_interval, day_in_study) -> pd.DataFrame:
    """Attach wave cluster, study_interval covariate, and day_in_study sort key for GEE."""
    out = attach_groups(X, groups)
    out["study_interval"] = (
        study_interval.values if hasattr(study_interval, "values") else study_interval
    )
    out["__day_in_study__"] = (
        day_in_study.values if hasattr(day_in_study, "values") else day_in_study
    )
    return out


def _sort_gee_frame(X, y=None):
    X = X.copy()
    order = np.lexsort((X["__day_in_study__"].to_numpy(), X["__group__"].astype(str).to_numpy()))
    X_sorted = X.iloc[order].reset_index(drop=True)
    if y is None:
        return X_sorted
    y_sorted = pd.Series(y).iloc[order].reset_index(drop=True)
    return X_sorted, y_sorted


def _split_gee_features(X):
    groups = X["__group__"].values
    features = X.drop(columns=GEE_META_COLS)
    return features, groups


def _encode_study_interval(series):
    values = pd.to_numeric(series, errors="coerce")
    mapping = {value: float(idx) for idx, value in enumerate(sorted(values.dropna().unique()))}
    return values.map(mapping).astype(float)


def _build_gee_exog_raw(X, *, include_study_interval=True):
    """Build numeric design columns for GEE from base features plus optional study_interval."""
    exog_parts = []
    numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
    if numeric_cols:
        exog_parts.append(X[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(float))

    cat_frame = pd.DataFrame(index=X.index)
    for col in CATEGORICAL_FEATURES:
        if col not in X.columns:
            continue
        if col == "phase":
            cat_frame[col] = X["phase"].astype(str).map(PHASE_TO_IDX).fillna(0).astype(float)
        else:
            cat_frame[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    if not cat_frame.empty:
        exog_parts.append(cat_frame)

    if include_study_interval and "study_interval" in X.columns:
        exog_parts.append(_encode_study_interval(X["study_interval"]).rename("study_interval"))

    if not exog_parts:
        raise ValueError("No GEE design columns found in input features.")
    return pd.concat(exog_parts, axis=1).reset_index(drop=True)


class _GEEExogMixin:
    def _prepare_gee_exog(self, exog_raw, *, fit=False, add_intercept=True):
        exog = exog_raw.copy()
        if fit:
            self.feature_columns_ = exog.columns.tolist()
            self.fill_values_ = exog.median(numeric_only=True)
        else:
            for col in self.feature_columns_:
                if col not in exog.columns:
                    exog[col] = np.nan
            exog = exog[self.feature_columns_]

        exog = exog.fillna(self.fill_values_)
        if add_intercept:
            exog = sm.add_constant(exog, has_constant="add")
        return exog


def _fit_binomial_gee_threshold(y_bin, exog, groups, maxiter=60):
    """Fit one cumulative-threshold Binomial GEE, preferring Autoregressive."""
    try:
        result = GEE(
            y_bin,
            exog,
            groups=groups,
            family=Binomial(),
            cov_struct=Autoregressive(grid=True),
        ).fit(maxiter=maxiter)
        structure = "Autoregressive"
    except ValueError:
        result = GEE(
            y_bin,
            exog,
            groups=groups,
            family=Binomial(),
            cov_struct=Exchangeable(),
        ).fit(maxiter=maxiter)
        structure = "Exchangeable (Autoregressive fit failed)"
    return result, structure


class FatigueGEEGaussianModel(_GEEExogMixin):
    """Gaussian GEE for ordinal regression with autoregressive wave-level correlation."""

    def __init__(self, maxiter=60):
        self.maxiter = maxiter
        self.result_ = None
        self.summary_ = None
        self.feature_columns_ = None
        self.fill_values_ = None

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("FatigueGEEGaussianModel requires participant-interval groups.")
        exog = self._prepare_gee_exog(_build_gee_exog_raw(X), fit=True, add_intercept=True)
        endog = pd.Series(y).astype(float).reset_index(drop=True)
        groups = pd.Series(groups).reset_index(drop=True)
        model = GEE(
            endog,
            exog,
            groups=groups,
            family=Gaussian(),
            cov_struct=Autoregressive(grid=True),
        )
        self.result_ = model.fit(maxiter=self.maxiter)
        self.summary_ = GEE_SUMMARY_HEADER + self.result_.summary().as_text()
        return self

    def predict(self, X, groups=None):
        del groups
        if self.result_ is None:
            raise ValueError("FatigueGEEGaussianModel has not been fitted yet.")
        exog = self._prepare_gee_exog(_build_gee_exog_raw(X), fit=False, add_intercept=True)
        return clip_ordinal_predictions(self.result_.predict(exog))


class FatigueGEEOrdinalModel(_GEEExogMixin):
    """Cumulative-threshold Binomial GEE with autoregressive wave-level correlation."""

    def __init__(self, maxiter=60):
        self.maxiter = maxiter
        self.threshold_results_ = None
        self.summary_ = None
        self.feature_columns_ = None
        self.fill_values_ = None

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("FatigueGEEOrdinalModel requires participant-interval groups.")
        exog = self._prepare_gee_exog(
            _build_gee_exog_raw(X, include_study_interval=False), fit=True, add_intercept=True
        )
        endog = pd.Series(y).astype(int).reset_index(drop=True)
        groups = pd.Series(groups).reset_index(drop=True)
        self.threshold_results_ = []
        summary_parts = [GEE_SUMMARY_HEADER.rstrip(), "Cumulative-threshold Binomial GEE models:"]
        for threshold in range(NUM_ORDINAL_THRESHOLDS):
            y_bin = (endog > threshold).astype(int)
            result, structure = _fit_binomial_gee_threshold(
                y_bin, exog, groups, maxiter=self.maxiter
            )
            self.threshold_results_.append(result)
            summary_parts.append(f"\n--- Threshold y > {threshold} ({structure}) ---")
            summary_parts.append(result.summary().as_text())
        self.summary_ = "\n".join(summary_parts) + "\n"
        return self

    def predict(self, X, groups=None):
        del groups
        if not self.threshold_results_:
            raise ValueError("FatigueGEEOrdinalModel has not been fitted yet.")
        exog = self._prepare_gee_exog(
            _build_gee_exog_raw(X, include_study_interval=False), fit=False, add_intercept=True
        )
        n_samples = len(exog)
        prob_gt = np.zeros((n_samples, NUM_ORDINAL_CLASSES), dtype=float)
        for threshold, result in enumerate(self.threshold_results_):
            prob_gt[:, threshold] = result.predict(exog)
        prob_gt[:, NUM_ORDINAL_THRESHOLDS] = 0.0
        for threshold in range(1, NUM_ORDINAL_CLASSES):
            prob_gt[:, threshold] = np.minimum(prob_gt[:, threshold], prob_gt[:, threshold - 1])
        prob_le = np.ones((n_samples, NUM_ORDINAL_CLASSES + 1), dtype=float)
        prob_le[:, 1:] = prob_gt
        class_probs = np.maximum(prob_le[:, :-1] - prob_le[:, 1:], 0.0)
        expected = class_probs @ np.arange(NUM_ORDINAL_CLASSES)
        return clip_ordinal_predictions(expected)


class FatigueGEEWrapper(BaseEstimator):
    """Sklearn wrapper for wave-clustered GEE models."""

    def __init__(self, model_cls, maxiter=60):
        self.model_cls = model_cls
        self.maxiter = maxiter
        self.model_ = None
        self.summary_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X, y = _sort_gee_frame(X, y)
        features, groups = _split_gee_features(X)
        self.model_ = self.model_cls(maxiter=self.maxiter)
        self.model_.fit(features, y, groups=groups)
        self.summary_ = self.model_.summary_
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = _sort_gee_frame(X)
        features, groups = _split_gee_features(X)
        return self.model_.predict(features, groups=groups)

    def summary(self):
        if self.summary_ is None:
            raise ValueError("FatigueGEEWrapper has not been fitted yet.")
        return self.summary_


def build_gee_gaussian(maxiter=60, **kwargs):
    del kwargs
    return FatigueGEEWrapper(FatigueGEEGaussianModel, maxiter=maxiter)


def build_gee_ordinal(maxiter=60, **kwargs):
    del kwargs
    return FatigueGEEWrapper(FatigueGEEOrdinalModel, maxiter=maxiter)
