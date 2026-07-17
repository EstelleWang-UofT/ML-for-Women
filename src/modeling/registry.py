"""Model registry and Optuna search spaces."""

from modeling.config import EWMA_ALPHA_RANGE, ROLLING_WINDOW_CHOICES
from modeling.models.hybrid import build_catboost_residual_expanding
from modeling.models.ordinal import (
    attach_groups,
    build_catboost_history,
    build_catboost_ordinal,
    build_catboost_regressor,
    build_linear_regression,
    build_mixed_effects,
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
    "mixed_effects": build_mixed_effects,
    "catboost_ordinal": build_catboost_ordinal,
}

HISTORY_ORDINAL_MODELS = {
    "catboost_history": build_catboost_history,
    "catboost_residual_expanding": build_catboost_residual_expanding,
}

RESIDUAL_ORDINAL_MODELS = {
    "catboost_residual_expanding": build_catboost_residual_expanding,
}

MIXED_EFFECTS_MODELS = {"mixed_effects"}
TREE_ORDINAL_MODELS = {
    "ordinal_rf",
    "catboost_regressor",
    "ordinal_forest",
    "catboost_ordinal",
}
PIPELINE_ORDINAL_MODELS = {"linear_regression", "ordered_logistic"}
HISTORY_TREE_ORDINAL_MODELS = {"catboost_history"}
RESIDUAL_TREE_ORDINAL_MODELS = {"catboost_residual_expanding"}

TUNING_ONLY_PARAM_KEYS = frozenset({"ewma_alpha", "rolling_window"})


def _catboost_search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


def _catboost_history_search_space(trial):
    return {
        **_catboost_search_space(trial),
        "ewma_alpha": trial.suggest_float(
            "ewma_alpha", EWMA_ALPHA_RANGE[0], EWMA_ALPHA_RANGE[1]
        ),
        "rolling_window": trial.suggest_categorical("rolling_window", ROLLING_WINDOW_CHOICES),
        "loss_mode": trial.suggest_categorical("loss_mode", ["rmse", "multiclass"]),
    }


def _catboost_residual_search_space(trial):
    return {
        **_catboost_search_space(trial),
        "ewma_alpha": trial.suggest_float(
            "ewma_alpha", EWMA_ALPHA_RANGE[0], EWMA_ALPHA_RANGE[1]
        ),
        "rolling_window": trial.suggest_categorical("rolling_window", ROLLING_WINDOW_CHOICES),
    }


def is_history_tree_model(name, task="ordinal"):
    del task
    return name in HISTORY_TREE_ORDINAL_MODELS


def is_residual_tree_model(name, task="ordinal"):
    del task
    return name in RESIDUAL_TREE_ORDINAL_MODELS


def resolve_training_data(name, bundle, task="ordinal"):
    """Return kwargs for tune_model / run_model_benchmark for a given model."""
    del task
    y_train_val = bundle.y_ord_train_val
    y_test = bundle.y_ord_test

    if name in MIXED_EFFECTS_MODELS:
        return {
            "X_train_val": attach_groups(bundle.X_train_val, bundle.groups_train_val),
            "X_test": attach_groups(bundle.X_test, bundle.groups_test),
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
        }

    if name in RESIDUAL_TREE_ORDINAL_MODELS:
        return {
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if name in HISTORY_TREE_ORDINAL_MODELS:
        return {
            "X_train_val": bundle.X_train_val_history_tree,
            "X_test": bundle.X_test_history_tree,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if name in TREE_ORDINAL_MODELS:
        return {
            "X_train_val": bundle.X_train_val_tree,
            "X_test": bundle.X_test_tree,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
        }

    if name in PIPELINE_ORDINAL_MODELS:
        return {
            "X_train_val": bundle.X_train_val,
            "X_test": bundle.X_test,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
        }

    return {
        "X_train_val": bundle.X_train_val,
        "X_test": bundle.X_test,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "groups": bundle.groups_train_val,
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
        "catboost_history": _catboost_history_search_space,
        "catboost_residual_expanding": _catboost_residual_search_space,
        "mixed_effects": lambda trial: {
            "maxiter": trial.suggest_int("maxiter", 200, 800),
        },
    }
    if name not in spaces:
        raise KeyError(f"No search space registered for model: {name}")
    return spaces[name]


def make_model_factory(name, params, registry):
    builder = registry[name]
    model_params = {k: v for k, v in params.items() if k not in TUNING_ONLY_PARAM_KEYS}

    def factory():
        return builder(**model_params)

    return factory
