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
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    params=None,
):
    """Tune (optional) + GroupKFold benchmark + test eval for one model."""
    data_kwargs = resolve_training_data(name, bundle, task=task)
    cv_score = None

    if params is None:
        tune_kwargs = {
            k: v
            for k, v in data_kwargs.items()
            if k in {"X_train_val", "y_train_val", "groups", "seq_train_val"}
        }
        params, cv_score = tune_model(
            name=name,
            registry=registry,
            search_space=get_search_space(name),
            task=task,
            n_trials=n_trials,
            n_splits=n_splits,
            test_ids=bundle.test_ids,
            **tune_kwargs,
        )

    factory = make_model_factory(name, params, registry)
    benchmark_kwargs = {
        "name": name,
        "model_factory": factory,
        "task": task,
        "n_splits": n_splits,
        "test_ids": bundle.test_ids,
        "y_train_val": data_kwargs["y_train_val"],
        "y_test": data_kwargs["y_test"],
        "groups": data_kwargs["groups"],
        "use_sequences": data_kwargs.get("use_sequences", False),
    }
    if benchmark_kwargs["use_sequences"]:
        benchmark_kwargs["seq_train_val"] = data_kwargs["seq_train_val"]
        benchmark_kwargs["seq_test"] = data_kwargs["seq_test"]
    else:
        benchmark_kwargs["X_train_val"] = data_kwargs["X_train_val"]
        benchmark_kwargs["X_test"] = data_kwargs["X_test"]

    result = run_model_benchmark(**benchmark_kwargs)
    result["best_params"] = params
    result["best_cv_score"] = cv_score
    return result, params
