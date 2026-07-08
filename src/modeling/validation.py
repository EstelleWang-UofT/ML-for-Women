"""Repeated participant-split stability studies."""

import pandas as pd

from modeling.baselines import (
    build_expanding_mean_baseline,
    run_baseline_benchmark,
)
from modeling.config import (
    N_CV_FOLDS,
    OPTUNA_TRIALS,
    STABILITY_MODELS,
    STABILITY_SEEDS,
)
from modeling.data import prepare_splits
from modeling.registry import HISTORY_ORDINAL_MODELS, RESIDUAL_ORDINAL_MODELS
from modeling.runner import tune_and_benchmark_model


def run_stability_study(
    df,
    seeds=None,
    models=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    retune=True,
):
    """Run repeated stratified participant splits with per-seed evaluation.

    For ``catboost_history`` and ``catboost_residual_expanding``, Optuna is
    re-run on each seed when ``retune=True``. ``expanding_mean`` is evaluated
    without tuning on each split.
    """
    del retune  # full retune is always used for tuned models in this study
    seeds = list(seeds or STABILITY_SEEDS)
    models = list(models or STABILITY_MODELS)
    rows = []

    for seed in seeds:
        bundle = prepare_splits(df, seed=seed)
        n_test_participants = len(bundle.test_ids)

        for model_name in models:
            if model_name == "expanding_mean":
                result = run_baseline_benchmark(
                    "expanding_mean",
                    build_expanding_mean_baseline,
                    bundle,
                    task="ordinal",
                    n_splits=n_splits,
                )
                best_params = {}
                cv_mae_mean = result["cv_summary"].loc["mean", "mae"]
                cv_mae_std = result["cv_summary"].loc["std", "mae"]
            elif model_name == "catboost_history":
                result, best_params = tune_and_benchmark_model(
                    name="catboost_history",
                    bundle=bundle,
                    registry=HISTORY_ORDINAL_MODELS,
                    task="ordinal",
                    n_trials=n_trials,
                    n_splits=n_splits,
                )
                cv_mae_mean = result["cv_summary"].loc["mean", "mae"]
                cv_mae_std = result["cv_summary"].loc["std", "mae"]
            elif model_name == "catboost_residual_expanding":
                result, best_params = tune_and_benchmark_model(
                    name="catboost_residual_expanding",
                    bundle=bundle,
                    registry=RESIDUAL_ORDINAL_MODELS,
                    task="ordinal",
                    n_trials=n_trials,
                    n_splits=n_splits,
                )
                cv_mae_mean = result["cv_summary"].loc["mean", "mae"]
                cv_mae_std = result["cv_summary"].loc["std", "mae"]
            else:
                raise ValueError(f"Unsupported stability model: {model_name}")

            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "test_mae": result["test_metrics"]["mae"],
                    "test_rmse": result["test_metrics"]["rmse"],
                    "test_qwk": result["test_metrics"]["qwk"],
                    "cv_mae_mean": cv_mae_mean,
                    "cv_mae_std": cv_mae_std,
                    "best_params": best_params,
                    "n_test_participants": n_test_participants,
                }
            )

    return pd.DataFrame(rows)


def summarize_stability(stability_df):
    """Aggregate per-model test MAE across seeds."""
    if stability_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for model_name, group in stability_df.groupby("model"):
        row = {
            "model": model_name,
            "n_seeds": len(group),
            "test_mae_mean": group["test_mae"].mean(),
            "test_mae_std": group["test_mae"].std(),
            "test_mae_ci95_half": 1.96 * group["test_mae"].std() / (len(group) ** 0.5),
            "cv_mae_mean": group["cv_mae_mean"].mean(),
            "cv_mae_std": group["cv_mae_std"].mean(),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).set_index("model")
