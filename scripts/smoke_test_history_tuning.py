"""Smoke test for history feature ablation (feature_set='history')."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.config import DATA_PATH, HISTORY_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from modeling.data import (
    history_feature_matrices,
    load_fatigue_data,
    prepare_splits,
)
from modeling.registry import ORDINAL_MODELS
from modeling.runner import tune_and_benchmark_model

df = load_fatigue_data(DATA_PATH)
bundle = prepare_splits(df)
X_tv_raw, X_te_raw, X_tv_tree, X_te_tree = history_feature_matrices(bundle)

expected_raw_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(HISTORY_FEATURES)
assert X_tv_raw.shape[1] == expected_raw_cols
assert X_te_raw.shape[1] == expected_raw_cols
assert X_tv_tree.shape[0] == len(bundle.y_ord_train_val)
print("history raw cols", X_tv_raw.shape[1], "tree cols", X_tv_tree.shape[1])

result, params = tune_and_benchmark_model(
    "catboost_regressor",
    bundle,
    ORDINAL_MODELS,
    feature_set="history",
    display_name="catboost_regressor_history",
    n_trials=2,
    n_splits=3,
)
print("catboost_regressor_history test_mae", result["test_metrics"]["mae"])
print("params keys", sorted(params.keys()))
assert result["name"] == "catboost_regressor_history"
assert result["feature_set"] == "history"

result_lr, _ = tune_and_benchmark_model(
    "linear_regression",
    bundle,
    ORDINAL_MODELS,
    feature_set="history",
    display_name="linear_regression_history",
    n_trials=2,
    n_splits=3,
)
print("linear_regression_history test_mae", result_lr["test_metrics"]["mae"])
print("OK")
