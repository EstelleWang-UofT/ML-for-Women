"""Model registry and Optuna search spaces."""

from modeling.data import history_feature_matrices
from modeling.models.ordinal import (
    attach_groups,
    build_catboost_ordinal,
    build_catboost_regressor,
    build_linear_regression,
    build_population_ordered_logistic,
    build_ordered_logistic,
    build_ordinal_forest,
    build_ordinal_rf,
)

ORDINAL_MODELS = {
    "linear_regression": build_linear_regression,
    "ordinal_rf": build_ordinal_rf,
    "catboost_regressor": build_catboost_regressor,
    "ordered_logistic": build_ordered_logistic,
    "ordinal_forest": build_ordinal_forest,
    "population_ordered_logistic": build_population_ordered_logistic,
    "catboost_ordinal": build_catboost_ordinal,
}

POPULATION_ORDINAL_MODELS = {"population_ordered_logistic"}
TREE_ORDINAL_MODELS = {
    "ordinal_rf",
    "catboost_regressor",
    "ordinal_forest",
    "catboost_ordinal",
}
PIPELINE_ORDINAL_MODELS = {"linear_regression", "ordered_logistic"}


def _catboost_search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


def resolve_training_data(name, bundle, feature_set="tabular"):
    """Return kwargs for tune_model / run_model_benchmark for a given model."""
    y_train_val = bundle.y_ord_train_val
    y_test = bundle.y_ord_test
    groups = bundle.groups_train_val

    if feature_set == "history":
        X_train_val_raw, X_test_raw, X_train_val_tree, X_test_tree = history_feature_matrices(
            bundle
        )
    else:
        X_train_val_raw = bundle.X_train_val
        X_test_raw = bundle.X_test
        X_train_val_tree = bundle.X_train_val_tree
        X_test_tree = bundle.X_test_tree

    if name in POPULATION_ORDINAL_MODELS:
        return {
            "X_train_val": attach_groups(X_train_val_raw, groups),
            "X_test": attach_groups(X_test_raw, bundle.groups_test),
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": groups,
        }

    if name in TREE_ORDINAL_MODELS:
        return {
            "X_train_val": X_train_val_tree,
            "X_test": X_test_tree,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": groups,
        }

    if name in PIPELINE_ORDINAL_MODELS:
        return {
            "X_train_val": X_train_val_raw,
            "X_test": X_test_raw,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": groups,
        }

    return {
        "X_train_val": X_train_val_raw,
        "X_test": X_test_raw,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "groups": groups,
    }


def get_search_space(name, task="ordinal"):
    del task
    spaces = {
        "linear_regression": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        },
        "ordered_logistic": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        },
        "ordinal_rf": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        },
        "ordinal_forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        },
        "catboost_regressor": _catboost_search_space,
        "catboost_ordinal": _catboost_search_space,
        "population_ordered_logistic": lambda trial: {
            "maxiter": trial.suggest_int("maxiter", 200, 800),
        },
    }
    if name not in spaces:
        raise KeyError(f"No search space registered for model: {name}")
    return spaces[name]


def make_model_factory(name, params, registry):
    builder = registry[name]

    def factory():
        return builder(**params)

    return factory
