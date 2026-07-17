"""Single-model tune + benchmark pipeline for notebook cells."""

from modeling.config import N_CV_FOLDS, OPTUNA_TRIALS
from modeling.cv import run_model_benchmark
from modeling.registry import get_search_space, make_model_factory, resolve_training_data
from modeling.tuning import tune_model


def tune_and_benchmark_model(
    name,
    bundle,
    registry,
    task="ordinal",
    feature_set="tabular",
    display_name=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    params=None,
):
    """Tune (optional) + GroupKFold benchmark + test eval for one model."""
    data_kwargs = resolve_training_data(name, bundle, feature_set=feature_set)
    cv_score = None

    if params is None:
        tune_kwargs = {
            k: v
            for k, v in data_kwargs.items()
            if k in {"X_train_val", "y_train_val", "groups"}
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

    X_train_val = data_kwargs["X_train_val"]
    X_test = data_kwargs["X_test"]
    factory = make_model_factory(name, params, registry)
    result = run_model_benchmark(
        name=display_name or name,
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
    result["name"] = display_name or name
    result["best_params"] = params
    result["best_cv_score"] = cv_score
    result["feature_set"] = feature_set
    return result, params
