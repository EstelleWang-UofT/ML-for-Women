"""Single-model tune + benchmark pipeline for notebook cells."""

from modeling.config import EWMA_ALPHA, N_CV_FOLDS, OPTUNA_TRIALS, ROLLING_WINDOW
from modeling.cv import run_model_benchmark
from modeling.data import build_history_tree_matrices, build_hybrid_tree_matrices
from modeling.registry import (
    get_search_space,
    is_history_tree_model,
    is_residual_tree_model,
    make_model_factory,
    resolve_training_data,
)
from modeling.tuning import tune_model


def _history_params_from_params(params):
    return {
        "ewma_alpha": params.get("ewma_alpha", EWMA_ALPHA),
        "rolling_window": params.get("rolling_window", ROLLING_WINDOW),
    }


def tune_and_benchmark_model(
    name,
    bundle,
    registry,
    task="ordinal",
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    params=None,
):
    """Tune (optional) + GroupKFold benchmark + test eval for one model."""
    data_kwargs = resolve_training_data(name, bundle, task=task)
    cv_score = None
    use_history = is_history_tree_model(name, task=task)
    use_residual = is_residual_tree_model(name, task=task)

    if params is None:
        tune_kwargs = {
            k: v
            for k, v in data_kwargs.items()
            if k in {"X_train_val", "y_train_val", "groups", "bundle"}
        }
        params, cv_score = tune_model(
            name=name,
            registry=registry,
            search_space=get_search_space(name, task=task),
            task=task,
            n_trials=n_trials,
            n_splits=n_splits,
            test_ids=bundle.test_ids,
            **tune_kwargs,
        )

    history_params = None
    if use_history:
        history_params = _history_params_from_params(params)
        X_train_val, X_test = build_history_tree_matrices(
            bundle.df,
            bundle.train_val_mask,
            bundle.test_mask,
            **history_params,
        )
    elif use_residual:
        history_params = _history_params_from_params(params)
        X_train_val, X_test = build_hybrid_tree_matrices(
            bundle.df,
            bundle.train_val_mask,
            bundle.test_mask,
            **history_params,
        )
    else:
        X_train_val = data_kwargs["X_train_val"]
        X_test = data_kwargs["X_test"]

    factory = make_model_factory(name, params, registry)
    result = run_model_benchmark(
        name=name,
        model_factory=factory,
        task=task,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
        y_train_val=data_kwargs["y_train_val"],
        y_test=data_kwargs["y_test"],
        groups=data_kwargs["groups"],
        X_train_val=X_train_val,
        X_test=X_test,
    )
    result["best_params"] = params
    result["best_cv_score"] = cv_score
    if history_params is not None:
        result["history_params"] = history_params
    return result, params
