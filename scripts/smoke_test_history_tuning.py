"""Smoke test for catboost history tuning changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.config import DATA_PATH
from modeling.data import build_history_tree_matrices, load_fatigue_data, prepare_splits
from modeling.models.ordinal import build_catboost_history
from modeling.registry import HISTORY_ORDINAL_MODELS
from modeling.runner import tune_and_benchmark_model
from modeling.validation import run_stability_study, summarize_stability

df = load_fatigue_data(DATA_PATH)
bundle = prepare_splits(df, ewma_alpha=0.25, rolling_window=5)
X_tv, X_te = build_history_tree_matrices(df, bundle.train_val_mask, bundle.test_mask, 0.25, 5)
assert X_tv.shape[1] == X_te.shape[1]
print("history matrices", X_tv.shape)

m = build_catboost_history(iterations=10, depth=4, loss_mode="multiclass")
m.fit(X_tv.iloc[:200], bundle.y_ord_train_val.iloc[:200])
print("multiclass preds sample", m.predict(X_te.iloc[:50])[:5])

m2 = build_catboost_history(iterations=10, depth=4, loss_mode="rmse")
m2.fit(X_tv.iloc[:200], bundle.y_ord_train_val.iloc[:200])
print("rmse preds sample", m2.predict(X_te.iloc[:50])[:5])

result, params = tune_and_benchmark_model(
    "catboost_history", bundle, HISTORY_ORDINAL_MODELS, n_trials=2, n_splits=3
)
print("tune keys", sorted(params.keys()))
print("test_mae", result["test_metrics"]["mae"])
print("history_params", result.get("history_params"))

stab = run_stability_study(df, seeds=[42, 43], n_trials=2, n_splits=3)
print("stability rows", len(stab))
print(summarize_stability(stab))
print("OK")
