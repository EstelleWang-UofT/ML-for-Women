"""Model registry and Optuna search spaces."""

from modeling.config import EWMA_ALPHA_RANGE, ROLLING_WINDOW_CHOICES
from modeling.models.classification import (
    build_catboost_history_classifier,
    build_lightgbm_classifier,
    build_rf_classifier,
)
from modeling.models.hybrid import (
    build_catboost_residual_expanding,
    build_catboost_residual_expanding_clf,
)
from modeling.models.ordinal import (
    attach_groups,
    build_catboost_history,
    build_catboost_ordinal,
    build_mixed_effects,
    build_ordered_logistic,
    build_ordinal_rf,
)
from modeling.models.sequence import build_lstm

ORDINAL_MODELS = {
    "ordered_logistic": build_ordered_logistic,
    "ordinal_rf": build_ordinal_rf,
    "catboost_ordinal": build_catboost_ordinal,
    "mixed_effects": build_mixed_effects,
    "lstm": build_lstm,
}

HISTORY_ORDINAL_MODELS = {
    "catboost_history": build_catboost_history,
    "catboost_residual_expanding": build_catboost_residual_expanding,
}

RESIDUAL_ORDINAL_MODELS = {
    "catboost_residual_expanding": build_catboost_residual_expanding,
}

CLASSIFICATION_MODELS = {
    "lightgbm": build_lightgbm_classifier,
    "random_forest": build_rf_classifier,
}

HISTORY_CLASSIFICATION_MODELS = {
    "catboost_history_clf": build_catboost_history_classifier,
}

RESIDUAL_CLASSIFICATION_MODELS = {
    "catboost_residual_expanding_clf": build_catboost_residual_expanding_clf,
}

SEQUENCE_MODELS = {"lstm"}
MIXED_EFFECTS_MODELS = {"mixed_effects"}
TREE_ORDINAL_MODELS = {"ordinal_rf", "catboost_ordinal"}
HISTORY_TREE_ORDINAL_MODELS = {"catboost_history"}
RESIDUAL_TREE_ORDINAL_MODELS = {"catboost_residual_expanding"}
HISTORY_TREE_CLASSIFICATION_MODELS = {"catboost_history_clf"}
RESIDUAL_TREE_CLASSIFICATION_MODELS = {"catboost_residual_expanding_clf"}

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


def _catboost_history_clf_search_space(trial):
    return {
        **_catboost_search_space(trial),
        "ewma_alpha": trial.suggest_float(
            "ewma_alpha", EWMA_ALPHA_RANGE[0], EWMA_ALPHA_RANGE[1]
        ),
        "rolling_window": trial.suggest_categorical("rolling_window", ROLLING_WINDOW_CHOICES),
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
    if task == "classification":
        return name in HISTORY_TREE_CLASSIFICATION_MODELS
    return name in HISTORY_TREE_ORDINAL_MODELS


def is_residual_tree_model(name, task="ordinal"):
    if task == "classification":
        return name in RESIDUAL_TREE_CLASSIFICATION_MODELS
    return name in RESIDUAL_TREE_ORDINAL_MODELS


def resolve_training_data(name, bundle, task="ordinal"):
    """Return kwargs for tune_model / run_model_benchmark for a given model."""
    if name in SEQUENCE_MODELS:
        y_train_val = bundle.y_ord_train_val if task == "ordinal" else bundle.y_clf_train_val
        y_test = bundle.y_ord_test if task == "ordinal" else bundle.y_clf_test
        return {
            "use_sequences": True,
            "seq_train_val": bundle.seq_train_val,
            "seq_test": bundle.seq_test,
            "y_train_val": y_train_val,
            "y_test": y_test,
            "groups": bundle.groups_train_val,
        }

    if name in HISTORY_TREE_CLASSIFICATION_MODELS:
        return {
            "X_train_val": bundle.X_train_val_history_tree,
            "X_test": bundle.X_test_history_tree,
            "y_train_val": bundle.y_clf_train_val,
            "y_test": bundle.y_clf_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if name in RESIDUAL_TREE_CLASSIFICATION_MODELS:
        return {
            "y_train_val": bundle.y_clf_train_val,
            "y_test": bundle.y_clf_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if task == "classification":
        return {
            "X_train_val": bundle.X_train_val_tree,
            "X_test": bundle.X_test_tree,
            "y_train_val": bundle.y_clf_train_val,
            "y_test": bundle.y_clf_test,
            "groups": bundle.groups_train_val,
        }

    if name in MIXED_EFFECTS_MODELS:
        return {
            "X_train_val": attach_groups(bundle.X_train_val, bundle.groups_train_val),
            "X_test": attach_groups(bundle.X_test, bundle.groups_test),
            "y_train_val": bundle.y_ord_train_val,
            "y_test": bundle.y_ord_test,
            "groups": bundle.groups_train_val,
        }

    if name in RESIDUAL_TREE_ORDINAL_MODELS:
        return {
            "y_train_val": bundle.y_ord_train_val,
            "y_test": bundle.y_ord_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if name in HISTORY_TREE_ORDINAL_MODELS:
        return {
            "X_train_val": bundle.X_train_val_history_tree,
            "X_test": bundle.X_test_history_tree,
            "y_train_val": bundle.y_ord_train_val,
            "y_test": bundle.y_ord_test,
            "groups": bundle.groups_train_val,
            "bundle": bundle,
        }

    if name in TREE_ORDINAL_MODELS:
        return {
            "X_train_val": bundle.X_train_val_tree,
            "X_test": bundle.X_test_tree,
            "y_train_val": bundle.y_ord_train_val,
            "y_test": bundle.y_ord_test,
            "groups": bundle.groups_train_val,
        }

    return {
        "X_train_val": bundle.X_train_val,
        "X_test": bundle.X_test,
        "y_train_val": bundle.y_ord_train_val,
        "y_test": bundle.y_ord_test,
        "groups": bundle.groups_train_val,
    }


def get_search_space(name):
    spaces = {
        "ordered_logistic": lambda trial: {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        },
        "ordinal_rf": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        },
        "catboost_ordinal": _catboost_search_space,
        "catboost_history": _catboost_history_search_space,
        "catboost_history_clf": _catboost_history_clf_search_space,
        "catboost_residual_expanding": _catboost_residual_search_space,
        "catboost_residual_expanding_clf": _catboost_residual_search_space,
        "mixed_effects": lambda trial: {
            "maxiter": trial.suggest_int("maxiter", 200, 800),
        },
        "lstm": lambda trial: {
            "rnn_type": trial.suggest_categorical("rnn_type", ["lstm", "gru"]),
            "hidden_size": trial.suggest_int("hidden_size", 32, 128),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 20, 60),
            "patience": trial.suggest_int("patience", 5, 12),
        },
        "lightgbm": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        },
        "random_forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
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
