"""Binary classification model builders."""

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from modeling.config import RANDOM_STATE


def _make_catboost_classifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
):
    return CatBoostClassifier(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=RANDOM_STATE,
        verbose=0,
        train_dir=None,
    )


def build_catboost_history_classifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    **kwargs,
):
    del kwargs
    return _make_catboost_classifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
    )


def build_lightgbm_classifier(
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    **kwargs,
):
    del kwargs
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        verbose=-1,
    )


def build_rf_classifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    **kwargs,
):
    del kwargs
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
    )
