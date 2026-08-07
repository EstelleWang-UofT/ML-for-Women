"""Optuna hyperparameter tuning with GroupKFold on train/val only."""

import warnings

import optuna

from modeling.config import N_CV_FOLDS, OPTUNA_TRIALS, ORDINAL_METRIC, RANDOM_STATE
from modeling.cv import run_group_cv
from modeling.metrics import compute_metrics, compute_regression_metrics
from modeling.models.ordinal import attach_groups
from modeling.registry import CONTINUOUS_REGRESSION_MODELS, GROUPED_ORDINAL_MODELS, make_model_factory


def _run_group_cv(factory, X, y, groups, n_splits, test_ids, metrics_fn):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Pipeline instance is not fitted yet.*",
        )
        return run_group_cv(
            factory,
            X,
            y,
            groups,
            n_splits=n_splits,
            test_ids=test_ids,
            metrics_fn=metrics_fn,
        )


def tune_model(
    name,
    registry,
    search_space,
    y_train_val,
    groups,
    X_train_val=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    test_ids=None,
    bundle=None,
):
    """Run Optuna with GroupKFold CV on train/val."""
    del bundle
    use_grouped = name in GROUPED_ORDINAL_MODELS
    metrics_fn = (
        compute_regression_metrics
        if name in CONTINUOUS_REGRESSION_MODELS
        else compute_metrics
    )

    if X_train_val is None:
        raise ValueError("X_train_val is required.")

    if use_grouped:
        X_train_val = attach_groups(X_train_val, groups)

    def objective(trial):
        params = search_space(trial)
        factory = make_model_factory(name, params, registry)
        fold_df, _ = _run_group_cv(
            factory,
            X_train_val,
            y_train_val,
            groups,
            n_splits=n_splits,
            test_ids=test_ids,
            metrics_fn=metrics_fn,
        )
        return fold_df[ORDINAL_METRIC].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value
