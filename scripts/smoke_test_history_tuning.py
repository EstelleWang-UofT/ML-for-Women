"""Smoke test for history construction tuning and feature ablation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.config import (
    DATA_PATH,
    HISTORY_ABLATION_MODEL,
    HISTORY_CANDIDATE_FEATURES,
    HISTORY_PROXY_PARAMS,
    ROLLING_WINDOWS,
)
from modeling.data import load_fatigue_data, prepare_splits
from modeling.history_tuning import (
    cv_mae_with_history,
    exhaustive_tune_history_construction,
    rolling_windows_from_construction_params,
    run_forward_selection,
    run_leave_one_out_ablation,
    test_mae_with_history,
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
    rolling_windows=ROLLING_WINDOWS,
    history_cols=list(HISTORY_CANDIDATE_FEATURES),
    n_splits=3,
    test_ids=bundle.test_ids,
)
print("proxy model", HISTORY_ABLATION_MODEL)
print("proxy params", HISTORY_PROXY_PARAMS)
print("default cv_mae", round(default_mae, 4))

result = exhaustive_tune_history_construction(
    df,
    bundle,
    model_name=HISTORY_ABLATION_MODEL,
    n_splits=3,
    alpha_values=[0.2, 0.4],
    rolling_choices=[2, 3],
    show_progress=False,
)
print("n_evaluated", result["n_evaluated"])
print("best params", result["best_params"])
print("best cv_mae", round(result["best_cv_mae"], 4))
assert result["n_evaluated"] == 16
assert len(result["grid_results"]) == 16
assert "eval_index" in result["grid_results"].columns
assert "ewma_alpha" in result["best_params"]

alpha = result["best_params"]["ewma_alpha"]
windows = rolling_windows_from_construction_params(result["best_params"])
print("best rolling_windows", windows)

_, feature_importance = run_leave_one_out_ablation(
    df,
    bundle,
    ewma_alpha=alpha,
    rolling_windows=windows,
    model_name=HISTORY_ABLATION_MODEL,
    n_splits=3,
)
print("feature importance rows", len(feature_importance))
assert len(feature_importance) == len(HISTORY_CANDIDATE_FEATURES)

selected, forward_mae, path = run_forward_selection(
    df,
    bundle,
    ewma_alpha=alpha,
    rolling_windows=windows,
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
    rolling_windows=windows,
    history_cols=selected,
)
print("test mae", round(test_mae, 4))
print("OK")
