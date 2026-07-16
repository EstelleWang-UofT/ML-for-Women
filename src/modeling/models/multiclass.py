"""Multiclass model builders for fatigue_num (0-5)."""

from sklearn.ensemble import RandomForestClassifier

from modeling.config import RANDOM_STATE

NUM_CLASSES = 6


def _make_catboost_multiclass(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="MultiClass",
        classes_count=NUM_CLASSES,
        auto_class_weights="Balanced",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=RANDOM_STATE,
        verbose=0,
        train_dir=None,
    )


def build_catboost_history_multiclass(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    **kwargs,
):
    del kwargs
    return _make_catboost_multiclass(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
    )


def build_lightgbm_multiclass(
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    **kwargs,
):
    del kwargs
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        verbose=-1,
    )


def build_rf_multiclass(
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
