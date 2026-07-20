"""Fatigue modeling package."""

from modeling.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
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
from modeling.summaries import build_history_ablation_summary, collect_summaries
from modeling.tuning import tune_model

__all__ = [
    "CATEGORICAL_FEATURES",
    "DATA_PATH",
    "FEATURE_COLUMNS",
    "N_CV_FOLDS",
    "NUMERIC_FEATURES",
    "OPTUNA_TRIALS",
    "ORDINAL_MODELS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "build_history_ablation_summary",
    "collect_summaries",
    "compute_metrics",
    "evaluate_on_test",
    "get_search_space",
    "load_fatigue_data",
    "prepare_splits",
    "run_all_baseline_benchmarks",
    "run_group_cv",
    "run_model_benchmark",
    "split_summary_table",
    "summarize_baseline_metrics",
    "tune_and_benchmark_model",
    "tune_model",
]
