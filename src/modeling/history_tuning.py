"""Tune history construction params and run history-feature ablation studies."""

from __future__ import annotations

import warnings

import optuna
import pandas as pd

from modeling.config import (
    EWMA_ALPHA,
    EWMA_ALPHA_RANGE,
    HISTORY_ABLATION_MODEL,
    HISTORY_FEATURES,
    HISTORY_PROXY_PARAMS,
    HISTORY_TUNING_TRIALS,
    N_CV_FOLDS,
    ORDINAL_METRIC,
    RANDOM_STATE,
    ROLLING_WINDOW,
    ROLLING_WINDOW_CHOICES,
    ROLLING_WINDOW_COLUMNS,
    ROLLING_WINDOW_PARAM_NAMES,
    ROLLING_WINDOWS,
)
from modeling.cv import evaluate_on_test, run_group_cv
from modeling.data import (
    build_tree_matrix,
    impute_history_features,
    make_feature_matrix,
    make_feature_matrix_with_history,
    prepare_splits,
    resolve_rolling_windows,
)
from modeling.registry import (
    GROUPED_ORDINAL_MODELS,
    ORDINAL_MODELS,
    PIPELINE_ORDINAL_MODELS,
    TREE_ORDINAL_MODELS,
    make_model_factory,
)

DEFAULT_PROXY_PARAMS = {
    "catboost_ordinal": dict(HISTORY_PROXY_PARAMS),
    "catboost_regressor": {},
    "linear_regression": {"alpha": 1.0},
}


def _default_model_params(model_name):
    if model_name == HISTORY_ABLATION_MODEL:
        return dict(HISTORY_PROXY_PARAMS)
    return None


def rolling_windows_from_construction_params(params):
    """Map Optuna construction params to per-column rolling windows."""
    return {
        col: int(params[ROLLING_WINDOW_PARAM_NAMES[col]]) for col in ROLLING_WINDOW_COLUMNS
    }


def history_construction_search_space(trial):
    """Optuna search space for ewma_alpha and per-column rolling windows."""
    low, high = EWMA_ALPHA_RANGE
    params = {"ewma_alpha": trial.suggest_float("ewma_alpha", low, high, log=True)}
    for col in ROLLING_WINDOW_COLUMNS:
        param_name = ROLLING_WINDOW_PARAM_NAMES[col]
        params[param_name] = trial.suggest_categorical(
            param_name, list(ROLLING_WINDOW_CHOICES)
        )
    return params


def _resolve_model_matrix(name, X_raw, X_tree):
    if name in TREE_ORDINAL_MODELS:
        return X_tree
    if name in PIPELINE_ORDINAL_MODELS | GROUPED_ORDINAL_MODELS:
        return X_raw
    return X_raw


def build_history_matrices(
    df,
    train_val_mask,
    test_mask,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    history_cols=None,
):
    """Build imputed raw and tree matrices for a history configuration."""
    windows = resolve_rolling_windows(rolling_windows, rolling_window)
    if history_cols is not None and len(history_cols) == 0:
        X_all = make_feature_matrix(df)
        X_train_val = X_all.loc[train_val_mask].reset_index(drop=True)
        X_test = X_all.loc[test_mask].reset_index(drop=True)
        return (
            X_train_val,
            X_test,
            build_tree_matrix(X_train_val),
            build_tree_matrix(X_test),
        )

    X_all_history = make_feature_matrix_with_history(
        df,
        ewma_alpha=ewma_alpha,
        rolling_windows=windows,
    )
    X_train_val = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test = X_all_history.loc[test_mask].reset_index(drop=True)
    impute_cols = history_cols if history_cols is not None else HISTORY_FEATURES
    X_train_val, X_test = impute_history_features(
        X_train_val,
        X_test,
        history_cols=impute_cols,
    )
    if history_cols is not None:
        drop = [
            col
            for col in HISTORY_FEATURES
            if col in X_train_val.columns and col not in history_cols
        ]
        if drop:
            X_train_val = X_train_val.drop(columns=drop)
            X_test = X_test.drop(columns=drop)
    return (
        X_train_val,
        X_test,
        build_tree_matrix(X_train_val),
        build_tree_matrix(X_test),
    )


def cv_mae_with_history(
    df,
    train_val_mask,
    test_mask,
    y_train_val,
    groups,
    model_name,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    history_cols=None,
    model_params=None,
    n_splits=N_CV_FOLDS,
    test_ids=None,
):
    """GroupKFold CV MAE for a fixed proxy model and history configuration."""
    if model_name not in ORDINAL_MODELS:
        raise KeyError(f"Unknown model: {model_name!r}")
    params = dict(DEFAULT_PROXY_PARAMS.get(model_name, {}))
    if model_params is None:
        model_params = _default_model_params(model_name)
    if model_params:
        params.update(model_params)

    X_raw, _, X_tree, _ = build_history_matrices(
        df,
        train_val_mask,
        test_mask,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
        rolling_window=rolling_window,
        history_cols=history_cols,
    )
    X = _resolve_model_matrix(model_name, X_raw, X_tree)
    factory = make_model_factory(model_name, params, ORDINAL_MODELS)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*Pipeline instance is not fitted yet.*",
        )
        fold_df, _ = run_group_cv(
            factory,
            X,
            y_train_val,
            groups,
            n_splits=n_splits,
            test_ids=test_ids,
        )
    return float(fold_df[ORDINAL_METRIC].mean())


def test_mae_with_history(
    df,
    train_val_mask,
    test_mask,
    y_train_val,
    y_test,
    model_name,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    history_cols=None,
    model_params=None,
):
    """Held-out test MAE for a fixed proxy model and history configuration."""
    if model_name not in ORDINAL_MODELS:
        raise KeyError(f"Unknown model: {model_name!r}")
    params = dict(DEFAULT_PROXY_PARAMS.get(model_name, {}))
    if model_params is None:
        model_params = _default_model_params(model_name)
    if model_params:
        params.update(model_params)

    X_train_raw, X_test_raw, X_train_tree, X_test_tree = build_history_matrices(
        df,
        train_val_mask,
        test_mask,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
        rolling_window=rolling_window,
        history_cols=history_cols,
    )
    X_train = _resolve_model_matrix(model_name, X_train_raw, X_train_tree)
    X_test = _resolve_model_matrix(model_name, X_test_raw, X_test_tree)
    factory = make_model_factory(model_name, params, ORDINAL_MODELS)
    metrics, _ = evaluate_on_test(factory, X_train, y_train_val, X_test, y_test)
    return float(metrics[ORDINAL_METRIC])


def tune_history_construction(
    df,
    bundle,
    model_name=HISTORY_ABLATION_MODEL,
    model_params=None,
    n_trials=HISTORY_TUNING_TRIALS,
    n_splits=N_CV_FOLDS,
    seed=RANDOM_STATE,
):
    """Optuna study over ewma_alpha and per-column rolling windows (all history features)."""
    del seed
    if model_params is None:
        model_params = _default_model_params(model_name)
    train_val_mask = bundle.train_val_mask
    test_mask = bundle.test_mask
    y_train_val = bundle.y_ord_train_val
    groups = bundle.groups_train_val
    test_ids = bundle.test_ids

    def objective(trial):
        params = history_construction_search_space(trial)
        return cv_mae_with_history(
            df,
            train_val_mask,
            test_mask,
            y_train_val,
            groups,
            model_name=model_name,
            ewma_alpha=params["ewma_alpha"],
            rolling_windows=rolling_windows_from_construction_params(params),
            history_cols=list(HISTORY_FEATURES),
            model_params=model_params,
            n_splits=n_splits,
            test_ids=test_ids,
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name="history_construction",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_cv_mae": study.best_value,
        "study": study,
    }


def run_leave_one_out_ablation(
    df,
    bundle,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    model_name=HISTORY_ABLATION_MODEL,
    model_params=None,
    n_splits=N_CV_FOLDS,
):
    """For each history feature, trains without that column and reports CV MAE increase.

    Higher ``cv_mae_increase_vs_all`` means removing that feature hurts prediction more.
    """
    if model_params is None:
        model_params = _default_model_params(model_name)
    train_val_mask = bundle.train_val_mask
    test_mask = bundle.test_mask
    y_train_val = bundle.y_ord_train_val
    groups = bundle.groups_train_val
    test_ids = bundle.test_ids

    all_features_cv_mae = cv_mae_with_history(
        df,
        train_val_mask,
        test_mask,
        y_train_val,
        groups,
        model_name=model_name,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
        rolling_window=rolling_window,
        history_cols=list(HISTORY_FEATURES),
        model_params=model_params,
        n_splits=n_splits,
        test_ids=test_ids,
    )

    rows = []
    for feature in HISTORY_FEATURES:
        subset = [col for col in HISTORY_FEATURES if col != feature]
        cv_mae_without_feature = cv_mae_with_history(
            df,
            train_val_mask,
            test_mask,
            y_train_val,
            groups,
            model_name=model_name,
            ewma_alpha=ewma_alpha,
            rolling_windows=rolling_windows,
            rolling_window=rolling_window,
            history_cols=subset,
            model_params=model_params,
            n_splits=n_splits,
            test_ids=test_ids,
        )
        rows.append(
            {
                "history_feature": feature,
                "n_history_features_remaining": len(subset),
                "cv_mae_without_feature": cv_mae_without_feature,
                "cv_mae_increase_vs_all": cv_mae_without_feature - all_features_cv_mae,
            }
        )

    feature_importance = pd.DataFrame(rows)
    feature_importance = feature_importance.sort_values(
        "cv_mae_increase_vs_all", ascending=False
    ).reset_index(drop=True)
    return all_features_cv_mae, feature_importance


def run_forward_selection(
    df,
    bundle,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    model_name=HISTORY_ABLATION_MODEL,
    model_params=None,
    n_splits=N_CV_FOLDS,
):
    """Greedy forward selection starting from base features only."""
    if model_params is None:
        model_params = _default_model_params(model_name)
    train_val_mask = bundle.train_val_mask
    test_mask = bundle.test_mask
    y_train_val = bundle.y_ord_train_val
    groups = bundle.groups_train_val
    test_ids = bundle.test_ids

    selected = []
    remaining = list(HISTORY_FEATURES)
    path_rows = []

    base_mae = cv_mae_with_history(
        df,
        train_val_mask,
        test_mask,
        y_train_val,
        groups,
        model_name=model_name,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
        rolling_window=rolling_window,
        history_cols=[],
        model_params=model_params,
        n_splits=n_splits,
        test_ids=test_ids,
    )
    best_mae = base_mae
    path_rows.append(
        {
            "step": 0,
            "added_feature": "(base only)",
            "selected_features": [],
            "cv_mae": base_mae,
            "delta_vs_prev": 0.0,
        }
    )

    step = 0
    while remaining:
        step += 1
        candidates = []
        for feature in remaining:
            trial_cols = selected + [feature]
            cv_mae = cv_mae_with_history(
                df,
                train_val_mask,
                test_mask,
                y_train_val,
                groups,
                model_name=model_name,
                ewma_alpha=ewma_alpha,
                rolling_windows=rolling_windows,
                rolling_window=rolling_window,
                history_cols=trial_cols,
                model_params=model_params,
                n_splits=n_splits,
                test_ids=test_ids,
            )
            candidates.append((feature, cv_mae))

        best_feature, trial_mae = min(candidates, key=lambda item: item[1])
        delta = trial_mae - best_mae
        if trial_mae >= best_mae:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_mae = trial_mae
        path_rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "selected_features": list(selected),
                "cv_mae": trial_mae,
                "delta_vs_prev": delta,
            }
        )

    path = pd.DataFrame(path_rows)
    return selected, best_mae, path


def summarize_history_recommendation(
    construction_result,
    forward_selected,
    forward_cv_mae,
    default_cv_mae=None,
    feature_importance=None,
):
    """Combine construction tuning and ablation into one recommendation dict."""
    best_params = construction_result["best_params"]
    best_rolling_windows = rolling_windows_from_construction_params(best_params)
    recommendation = {
        "ewma_alpha": best_params["ewma_alpha"],
        "rolling_windows": best_rolling_windows,
        "history_features": list(forward_selected),
        "construction_cv_mae": construction_result["best_cv_mae"],
        "forward_selection_cv_mae": forward_cv_mae,
        "default_ewma_alpha": EWMA_ALPHA,
        "default_rolling_windows": dict(ROLLING_WINDOWS),
        "default_history_features": list(HISTORY_FEATURES),
    }
    for col in ROLLING_WINDOW_COLUMNS:
        param_name = ROLLING_WINDOW_PARAM_NAMES[col]
        recommendation[param_name] = best_rolling_windows[col]
    if default_cv_mae is not None:
        recommendation["default_construction_cv_mae"] = default_cv_mae
    if feature_importance is not None and not feature_importance.empty:
        recommendation["top_history_features_by_importance"] = (
            feature_importance.head(3)["history_feature"].tolist()
        )
    return recommendation


def prepare_tuned_bundle(
    df,
    ewma_alpha,
    rolling_windows=None,
    rolling_window=None,
    seed=RANDOM_STATE,
):
    """Rebuild SplitBundle with tuned history construction params."""
    return prepare_splits(
        df,
        seed=seed,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
        rolling_window=rolling_window,
    )
