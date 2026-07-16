"""Smoke test multiclass metrics and baseline without optional ML deps."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules["modeling"] = type(sys)("modeling")
load("modeling.config", ROOT / "src/modeling/config.py")
load("modeling.metrics", ROOT / "src/modeling/metrics.py")
data = load("modeling.data", ROOT / "src/modeling/data.py")
cv = load("modeling.cv", ROOT / "src/modeling/cv.py")
baselines = load("modeling.baselines", ROOT / "src/modeling/baselines.py")

df = data.load_fatigue_data()
bundle = data.prepare_splits(df)

result = baselines.run_baseline_benchmark(
    "global_mode",
    baselines.build_global_mode_baseline,
    bundle,
    task="multiclass",
    n_splits=3,
)
assert "weighted_f1" in result["test_metrics"]
assert "macro_f1" in result["test_metrics"]
assert "accuracy" in result["test_metrics"]
print("global_mode weighted_f1", round(result["test_metrics"]["weighted_f1"], 4))
print("ok")
