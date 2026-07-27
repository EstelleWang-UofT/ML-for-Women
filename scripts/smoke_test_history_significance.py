"""Smoke test for participant-level history feature-count significance helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.significance import (
    apply_fdr,
    bootstrap_mean_ci,
    paired_participant_significance_test,
)


def _synthetic_paired_predictions(n_participants=10, rows_per_participant=20, seed=42):
    rng = np.random.default_rng(seed)
    participant_ids = np.repeat(np.arange(n_participants), rows_per_participant)
    y_true = rng.integers(0, 6, size=len(participant_ids))

    noise7 = rng.normal(0, 0.5, size=len(participant_ids))
    noise3 = rng.normal(0, 0.5, size=len(participant_ids))
    pred7 = np.clip(y_true + noise7, 0, 5)
    pred3 = np.clip(y_true + noise3, 0, 5)
    return y_true, pred7, pred3, participant_ids


y_true, pred7, pred3, participant_ids = _synthetic_paired_predictions()
result = paired_participant_significance_test(
    y_true,
    pred7,
    pred3,
    participant_ids,
    n_bootstrap=2000,
    seed=42,
)

print("n_participants", result["n_participants"])
print("mean_delta", round(result["mean_delta"], 4))
print("ci_low", round(result["ci_low"], 4))
print("ci_high", round(result["ci_high"], 4))
print("wilcoxon_p", round(result["p_value"], 4))

assert result["n_participants"] == 10
assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]

mean_only, ci_low, ci_high = bootstrap_mean_ci(np.zeros(8), n_bootstrap=2000, seed=42)
assert ci_low <= mean_only <= ci_high
assert abs(mean_only) < 1e-9

p_values = np.array([0.01, 0.04, 0.10, 0.20, 0.30, 0.40, 0.50])
fdr = apply_fdr(p_values)
print("fdr_p", np.round(fdr, 4))
assert len(fdr) == 7
assert np.all(fdr >= p_values - 1e-12)
assert np.all((fdr >= 0) & (fdr <= 1))

print("OK")
