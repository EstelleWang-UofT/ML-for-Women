"""Smoke test for history construction tuning and feature ablation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.config import (
    DATA_PATH,
    HISTORY_ABLATION_MODEL,
    HISTORY_FEATURES,
    HISTORY_PROXY_PARAMS,
)
from modeling.data import load_fatigue_data, prepare_splits
from modeling.history_tuning import (
    cv_mae_with_history,
    run_forward_selection,
    run_leave_one_out_ablation,
    test_mae_with_history,
    tune_history_construction,
)

df = load_fatigue_data(DATA_PATH)
bundle = prepare_splits(df)

default_mae = cv_mae_with_history(
    df,
    bundle.train_val_mask,
    bundle.test_mask,
    bundle.y_ord_train_val,
    bundle.groups_train_val,
    model_name=HISTORY_ABLATION_MODEL,
    ewma_alpha=0.3,
    rolling_window=3,
    history_cols=list(HISTORY_FEATURES),
    n_splits=3,
    test_ids=bundle.test_ids,
)
print("proxy model", HISTORY_ABLATION_MODEL)
print("proxy params", HISTORY_PROXY_PARAMS)
print("default cv_mae", round(default_mae, 4))

result = tune_history_construction(
    df,
    bundle,
    model_name=HISTORY_ABLATION_MODEL,
    n_trials=3,
    n_splits=3,
)
print("best params", result["best_params"])
print("best cv_mae", round(result["best_cv_mae"], 4))

alpha = result["best_params"]["ewma_alpha"]
window = int(result["best_params"]["rolling_window"])
_, feature_importance = run_leave_one_out_ablation(
    df,
    bundle,
    ewma_alpha=alpha,
    rolling_window=window,
    model_name=HISTORY_ABLATION_MODEL,
    n_splits=3,
)
print("feature importance rows", len(feature_importance))
assert len(feature_importance) == len(HISTORY_FEATURES)
assert list(feature_importance.columns) == [
    "history_feature",
    "n_history_features_remaining",
    "cv_mae_without_feature",
    "cv_mae_increase_vs_all",
]

selected, forward_mae, path = run_forward_selection(
    df,
    bundle,
    ewma_alpha=alpha,
    rolling_window=window,
    model_name=HISTORY_ABLATION_MODEL,
    n_splits=3,
)
print("forward selected", selected)
print("forward cv_mae", round(forward_mae, 4))
print("forward steps", len(path))

test_mae = test_mae_with_history(
    df,
    bundle.train_val_mask,
    bundle.test_mask,
    bundle.y_ord_train_val,
    bundle.y_ord_test,
    model_name=HISTORY_ABLATION_MODEL,
    ewma_alpha=alpha,
    rolling_window=window,
    history_cols=selected,
)
print("test mae", round(test_mae, 4))
print("OK")
