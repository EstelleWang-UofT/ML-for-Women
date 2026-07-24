"""Fatigue modeling package."""

from modeling.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    GEE_OPTUNA_TRIALS,
    HISTORY_ABLATION_MODEL,
    HISTORY_FEATURES,
    HISTORY_PROXY_PARAMS,
    HISTORY_TUNING_TRIALS,
    N_CV_FOLDS,
    NUMERIC_FEATURES,
    OPTUNA_TRIALS,
    RANDOM_STATE,
    TEST_SIZE,
)
from modeling.cv import evaluate_on_test, run_group_cv, run_model_benchmark
from modeling.data import load_fatigue_data, prepare_splits, split_summary_table
from modeling.metrics import compute_metrics
from modeling.baselines import run_all_baseline_benchmarks, summarize_baseline_metrics
from modeling.registry import ORDINAL_MODELS, get_search_space
from modeling.runner import tune_and_benchmark_model
from modeling.summaries import (
    CATEGORY_ORDER,
    build_history_ablation_summary,
    collect_categorized_summaries,
    collect_summaries,
)
from modeling.history_tuning import (
    prepare_tuned_bundle,
    run_forward_selection,
    run_leave_one_out_ablation,
    summarize_history_recommendation,
    test_mae_with_history,
    tune_history_construction,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "DATA_PATH",
    "FEATURE_COLUMNS",
    "GEE_OPTUNA_TRIALS",
    "HISTORY_ABLATION_MODEL",
    "HISTORY_FEATURES",
    "HISTORY_PROXY_PARAMS",
    "HISTORY_TUNING_TRIALS",
    "N_CV_FOLDS",
    "NUMERIC_FEATURES",
    "OPTUNA_TRIALS",
    "ORDINAL_MODELS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "CATEGORY_ORDER",
    "build_history_ablation_summary",
    "collect_categorized_summaries",
    "collect_summaries",
    "compute_metrics",
    "evaluate_on_test",
    "get_search_space",
    "load_fatigue_data",
    "prepare_splits",
    "prepare_tuned_bundle",
    "run_all_baseline_benchmarks",
    "run_forward_selection",
    "run_group_cv",
    "run_leave_one_out_ablation",
    "run_model_benchmark",
    "split_summary_table",
    "summarize_baseline_metrics",
    "summarize_history_recommendation",
    "test_mae_with_history",
    "tune_and_benchmark_model",
    "tune_history_construction",
    "tune_model",
]
