"""Optuna hyperparameter tuning with GroupKFold on train/val only."""

import warnings

import optuna

from modeling.config import (
    EWMA_ALPHA,
    N_CV_FOLDS,
    OPTUNA_TRIALS,
    ORDINAL_METRIC,
    ROLLING_WINDOW,
)
from modeling.cv import run_group_cv
from modeling.data import build_history_tree_matrices, build_hybrid_tree_matrices
from modeling.models.ordinal import attach_groups
from modeling.registry import (
    MIXED_EFFECTS_MODELS,
    is_history_tree_model,
    is_residual_tree_model,
    make_model_factory,
)


def _run_group_cv(factory, X, y, groups, n_splits, test_ids):
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
        )


def _build_trial_matrices(name, bundle, params):
    ewma_alpha = params.get("ewma_alpha", EWMA_ALPHA)
    rolling_window = params.get("rolling_window", ROLLING_WINDOW)
    if is_history_tree_model(name):
        return build_history_tree_matrices(
            bundle.df,
            bundle.train_val_mask,
            bundle.test_mask,
            ewma_alpha=ewma_alpha,
            rolling_window=rolling_window,
        )[0]
    if is_residual_tree_model(name):
        return build_hybrid_tree_matrices(
            bundle.df,
            bundle.train_val_mask,
            bundle.test_mask,
            ewma_alpha=ewma_alpha,
            rolling_window=rolling_window,
        )[0]
    raise ValueError(f"No trial matrix builder for model: {name}")


def tune_model(
    name,
    registry,
    search_space,
    y_train_val,
    groups,
    task="ordinal",
    X_train_val=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    test_ids=None,
    bundle=None,
):
    del task
    use_mixed = name in MIXED_EFFECTS_MODELS
    use_history = is_history_tree_model(name)
    use_residual = is_residual_tree_model(name)

    if use_history or use_residual:
        if bundle is None:
            raise ValueError("bundle is required for history/residual models.")
    elif X_train_val is None:
        raise ValueError("X_train_val is required for tabular models.")

    if use_mixed:
        X_train_val = attach_groups(X_train_val, groups)

    def objective(trial):
        params = search_space(trial)
        factory = make_model_factory(name, params, registry)
        if use_history or use_residual:
            X_tv = _build_trial_matrices(name, bundle, params)
            fold_df, _ = _run_group_cv(
                factory,
                X_tv,
                y_train_val,
                groups,
                n_splits=n_splits,
                test_ids=test_ids,
            )
        else:
            fold_df, _ = _run_group_cv(
                factory,
                X_train_val,
                y_train_val,
                groups,
                n_splits=n_splits,
                test_ids=test_ids,
            )
        return fold_df[ORDINAL_METRIC].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name=name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value
