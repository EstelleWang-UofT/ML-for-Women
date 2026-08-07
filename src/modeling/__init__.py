"""Fatigue modeling package."""

from modeling.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EWMA_ALPHA,
    EWMA_ALPHA_GRID,
    FEATURE_COLUMNS,
    GEE_OPTUNA_TRIALS,
    HISTORY_ABLATION_MODEL,
    HISTORY_CANDIDATE_FEATURES,
    HISTORY_FEATURES,
    HISTORY_PROXY_PARAMS,
    N_CV_FOLDS,
    NUMERIC_FEATURES,
    OPTUNA_TRIALS,
    RANDOM_STATE,
    TEST_SIZE,
)
from modeling.cv import evaluate_on_test, run_group_cv, run_model_benchmark
from modeling.data import (
    build_split_bundle,
    load_fatigue_data,
    participant_strata,
    preprocess_after_split,
    prepare_splits,
    split_participant_ids,
    split_summary_table,
)
from modeling.metrics import REGRESSION_METRIC_COLS, compute_metrics, compute_regression_metrics
from modeling.baselines import run_all_baseline_benchmarks, summarize_baseline_metrics
from modeling.registry import CONTINUOUS_REGRESSION_MODELS, ORDINAL_MODELS, get_search_space
from modeling.runner import tune_and_benchmark_model
from modeling.tuning import tune_model
from modeling.significance import (
    compare_history_feature_count_significance,
    paired_participant_significance_test,
)
from modeling.summaries import (
    CATEGORY_ORDER,
    build_history_ablation_summary,
    build_history_feature_count_comparison,
    collect_categorized_summaries,
    collect_summaries,
)
from modeling.history_tuning import (
    exhaustive_tune_history_construction,
    plot_history_construction_grid_mae,
    prepare_tuned_bundle,
    run_forward_selection,
    run_leave_one_out_ablation,
    summarize_history_recommendation,
    test_mae_with_history,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "DATA_PATH",
    "FEATURE_COLUMNS",
    "GEE_OPTUNA_TRIALS",
    "HISTORY_ABLATION_MODEL",
    "HISTORY_CANDIDATE_FEATURES",
    "EWMA_ALPHA_GRID",
    "HISTORY_FEATURES",
    "HISTORY_PROXY_PARAMS",
    "N_CV_FOLDS",
    "NUMERIC_FEATURES",
    "OPTUNA_TRIALS",
    "ORDINAL_MODELS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "CONTINUOUS_REGRESSION_MODELS",
    "REGRESSION_METRIC_COLS",
    "build_history_ablation_summary",
    "build_history_feature_count_comparison",
    "build_split_bundle",
    "collect_categorized_summaries",
    "collect_summaries",
    "compare_history_feature_count_significance",
    "compute_metrics",
    "compute_regression_metrics",
    "evaluate_on_test",
    "exhaustive_tune_history_construction",
    "get_search_space",
    "load_fatigue_data",
    "participant_strata",
    "plot_history_construction_grid_mae",
    "paired_participant_significance_test",
    "preprocess_after_split",
    "prepare_splits",
    "prepare_tuned_bundle",
    "run_all_baseline_benchmarks",
    "run_forward_selection",
    "run_group_cv",
    "run_leave_one_out_ablation",
    "run_model_benchmark",
    "split_participant_ids",
    "split_summary_table",
    "summarize_baseline_metrics",
    "summarize_history_recommendation",
    "test_mae_with_history",
    "tune_and_benchmark_model",
    "tune_model",
]
