"""Aggregate benchmark results into summary tables for notebooks."""

import pandas as pd

from modeling.metrics import ORDINAL_METRIC_COLS


def collect_summaries(results):
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
        cv_row = {
            "model": result["name"],
            "best_params": str(result.get("best_params", {})),
        }
        test_row = {
            "model": result["name"],
            "best_params": str(result.get("best_params", {})),
        }
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


def build_history_ablation_summary(test_summary, model_names):
    """Compare tabular vs history test MAE for each base model name.

    Expects ``test_summary`` indexed by model name with a ``test_mae`` column
    and companion rows named ``{base}_history`` for each base in
    ``model_names``.
    """
    rows = []
    for base in model_names:
        hist = f"{base}_history"
        if base not in test_summary.index or hist not in test_summary.index:
            continue
        tabular_mae = test_summary.loc[base, "test_mae"]
        history_mae = test_summary.loc[hist, "test_mae"]
        rows.append(
            {
                "model": base,
                "test_mae_tabular": tabular_mae,
                "test_mae_history": history_mae,
                "delta_mae": history_mae - tabular_mae,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model").sort_values("delta_mae")
