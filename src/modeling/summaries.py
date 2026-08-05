"""Aggregate benchmark results into summary tables for notebooks."""

import pandas as pd

from modeling.baselines import ORDINAL_BASELINES
from modeling.metrics import ORDINAL_METRIC_COLS
from modeling.registry import ORDINAL_MODELS

CATEGORY_ORDER = ("baseline", "base", "history")


def collect_summaries(results, include_best_params=True):
    """Build CV and held-out test summary tables from benchmark result dicts.

    Each result is the dict returned by ``run_model_benchmark`` or
    ``tune_and_benchmark_model`` (keys: name, cv_summary, test_metrics,
    best_params).

    CV columns are means over GroupKFold folds on train/val participants.
    Test columns come from a single refit on all train/val rows, evaluated
    on held-out test participants.
    """
    cv_rows, test_rows = [], []
    for result in results:
        cv_mean = result["cv_summary"].loc["mean", ORDINAL_METRIC_COLS]
        cv_std = result["cv_summary"].loc["std", ORDINAL_METRIC_COLS]
        cv_row = {"model": result["name"]}
        test_row = {"model": result["name"]}
        if include_best_params:
            cv_row["best_params"] = str(result.get("best_params", {}))
            test_row["best_params"] = str(result.get("best_params", {}))
        for col in ORDINAL_METRIC_COLS:
            cv_row[f"cv_{col}"] = cv_mean[col]
            test_row[f"test_{col}"] = result["test_metrics"][col]
        cv_row["cv_mae_std"] = cv_std["mae"]
        cv_rows.append(cv_row)
        test_rows.append(test_row)
    return (
        pd.DataFrame(cv_rows).set_index("model"),
        pd.DataFrame(test_rows).set_index("model"),
    )


def model_category(name: str) -> str:
    """Classify a benchmark result name into baseline, base, or history."""
    if name in ORDINAL_BASELINES:
        return "baseline"
    if name.endswith(("_history", "_history7", "_history3")):
        return "history"
    if name in ORDINAL_MODELS:
        return "base"
    raise ValueError(f"Unknown model category for {name!r}")


def collect_categorized_summaries(results, include_best_params=True):
    """Build per-category CV and test tables sorted by cv_mae and test_mae."""
    cv_summary, test_summary = collect_summaries(
        results, include_best_params=include_best_params
    )
    categorized = {}
    for category in CATEGORY_ORDER:
        cv_cat = cv_summary[cv_summary.index.map(lambda n: model_category(n) == category)]
        test_cat = test_summary[test_summary.index.map(lambda n: model_category(n) == category)]
        categorized[category] = (
            cv_cat.sort_values("cv_mae"),
            test_cat.sort_values("test_mae"),
        )
    return categorized


def build_history_ablation_summary(test_summary, model_names):
    """Compare base vs history test MAE for each base model name.

    Expects ``test_summary`` indexed by model name with a ``test_mae`` column
    and companion rows named ``{base}_history`` for each base in
    ``model_names``.
    """
    rows = []
    for base in model_names:
        hist = f"{base}_history"
        if base not in test_summary.index or hist not in test_summary.index:
            continue
        base_mae = test_summary.loc[base, "test_mae"]
        history_mae = test_summary.loc[hist, "test_mae"]
        rows.append(
            {
                "model": base,
                "test_mae_base": base_mae,
                "test_mae_history": history_mae,
                "delta_mae": history_mae - base_mae,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model").sort_values("delta_mae")


def build_history_feature_count_comparison(test_summary, model_names):
    """Compare 3-col vs 7-col history test MAE for each base model name."""
    rows = []
    for base in model_names:
        hist7 = f"{base}_history7"
        hist3 = f"{base}_history3"
        if hist7 not in test_summary.index or hist3 not in test_summary.index:
            continue
        mae7 = test_summary.loc[hist7, "test_mae"]
        mae3 = test_summary.loc[hist3, "test_mae"]
        rows.append(
            {
                "model": base,
                "test_mae_history7": mae7,
                "test_mae_history3": mae3,
                "delta_mae_3_minus_7": mae3 - mae7,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model").sort_values("delta_mae_3_minus_7")
