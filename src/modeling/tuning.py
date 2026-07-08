"""Optuna hyperparameter tuning with GroupKFold on train/val only."""

import optuna

from modeling.config import (
    CLASSIFICATION_METRIC,
    N_CV_FOLDS,
    OPTUNA_TRIALS,
    ORDINAL_METRIC,
)
from modeling.cv import run_group_cv, run_group_cv_sequences
from modeling.models.ordinal import attach_groups
from modeling.registry import MIXED_EFFECTS_MODELS, SEQUENCE_MODELS, make_model_factory


def _primary_metric(task):
    return ORDINAL_METRIC if task == "ordinal" else CLASSIFICATION_METRIC


def _direction(task):
    return "minimize" if task == "ordinal" else "maximize"


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
    seq_train_val=None,
):
    metric = _primary_metric(task)
    use_sequences = name in SEQUENCE_MODELS
    use_mixed = name in MIXED_EFFECTS_MODELS

    if use_sequences and seq_train_val is None:
        raise ValueError("seq_train_val is required for sequence models.")
    if not use_sequences and X_train_val is None:
        raise ValueError("X_train_val is required for tabular models.")

    if use_mixed:
        X_train_val = attach_groups(X_train_val, groups)

    def objective(trial):
        params = search_space(trial)
        factory = make_model_factory(name, params, registry)
        if use_sequences:
            fold_df, _ = run_group_cv_sequences(
                factory,
                seq_train_val,
                n_splits=n_splits,
                task=task,
                test_ids=test_ids,
            )
        else:
            fold_df, _ = run_group_cv(
                factory,
                X_train_val,
                y_train_val,
                groups,
                n_splits=n_splits,
                task=task,
                test_ids=test_ids,
            )
        return fold_df[metric].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=_direction(task), study_name=name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_all_models(
    registry,
    model_names,
    y_train_val,
    groups,
    task="ordinal",
    X_train_val=None,
    n_trials=OPTUNA_TRIALS,
    test_ids=None,
    seq_train_val=None,
):
    from modeling.registry import get_search_space

    best_params = {}
    best_cv_scores = {}
    for name in model_names:
        params, score = tune_model(
            name=name,
            registry=registry,
            search_space=get_search_space(name),
            y_train_val=y_train_val,
            groups=groups,
            task=task,
            X_train_val=X_train_val,
            n_trials=n_trials,
            test_ids=test_ids,
            seq_train_val=seq_train_val,
        )
        best_params[name] = params
        best_cv_scores[name] = score
    return best_params, best_cv_scores
