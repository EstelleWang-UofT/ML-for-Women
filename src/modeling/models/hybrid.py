"""Hybrid ordinal models: persistence baseline + ML residual."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator

from modeling.config import PRIOR_COL, RANDOM_STATE
from modeling.metrics import clip_ordinal_predictions


class ResidualOrdinalWrapper(BaseEstimator):
    """Predict ordinal target as prior + residual adjustment from a regressor."""

    def __init__(self, regressor, prior_col=PRIOR_COL):
        self.regressor = regressor
        self.prior_col = prior_col
        self.fallback_ = None

    def _prior_series(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"ResidualOrdinalWrapper expects a DataFrame with {self.prior_col}.")
        return pd.to_numeric(X[self.prior_col], errors="coerce")

    def _feature_matrix(self, X):
        return X.drop(columns=[self.prior_col])

    def fit(self, X, y):
        prior = self._prior_series(X)
        if prior.notna().any():
            self.fallback_ = float(prior.dropna().median())
        else:
            self.fallback_ = float(np.median(y))
        prior_filled = prior.fillna(self.fallback_)
        residual = np.asarray(y, dtype=float) - prior_filled.to_numpy()
        self.regressor.fit(self._feature_matrix(X), residual)
        return self

    def predict(self, X):
        prior = self._prior_series(X).fillna(self.fallback_)
        residual_hat = self.regressor.predict(self._feature_matrix(X))
        return clip_ordinal_predictions(prior.to_numpy() + residual_hat)


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


def build_catboost_residual_expanding(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    **kwargs,
):
    del kwargs
    return ResidualOrdinalWrapper(
        _make_catboost_regressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
        )
    )
