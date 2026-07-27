"""Participant-level significance tests for paired model comparisons."""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from modeling.config import HISTORY_CANDIDATE_FEATURES, HISTORY_FEATURES, RANDOM_STATE
from modeling.metrics import clip_ordinal_predictions, compute_metrics
from modeling.registry import ORDINAL_MODELS, make_model_factory, resolve_training_data


def per_participant_mae(y_true, y_pred, participant_ids):
    """Mean absolute error per participant on held-out test rows."""
    y_true = np.asarray(y_true)
    y_pred = clip_ordinal_predictions(y_pred)
    participant_ids = np.asarray(participant_ids)
    errors = pd.DataFrame(
        {
            "participant": participant_ids,
            "abs_error": np.abs(y_true - y_pred),
        }
    )
    return errors.groupby("participant")["abs_error"].mean()


def participant_mae_delta(y_true, pred_a, pred_b, participant_ids):
    """Per-participant MAE(pred_b) - MAE(pred_a)."""
    mae_a = per_participant_mae(y_true, pred_a, participant_ids)
    mae_b = per_participant_mae(y_true, pred_b, participant_ids)
    return (mae_b - mae_a).sort_index()


def bootstrap_mean_ci(values, n_bootstrap=10000, seed=RANDOM_STATE):
    """Bootstrap 95% CI for the mean of participant-level values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_means[i] = values[idx].mean()
    return (
        float(values.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def apply_fdr(p_values):
    """Benjamini-Hochberg FDR adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev

    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def paired_participant_significance_test(
    y_true,
    pred_a,
    pred_b,
    participant_ids,
    n_bootstrap=10000,
    seed=RANDOM_STATE,
):
    """Wilcoxon + cluster bootstrap on per-participant MAE(pred_b) - MAE(pred_a)."""
    deltas = participant_mae_delta(y_true, pred_a, pred_b, participant_ids)
    n_participants = len(deltas)
    if n_participants == 0:
        return {
            "n_participants": 0,
            "mean_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value": np.nan,
            "participant_deltas": deltas,
        }

    mean_delta, ci_low, ci_high = bootstrap_mean_ci(
        deltas.values,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if n_participants < 2 or np.allclose(deltas.values, 0.0):
        wilcoxon_stat = np.nan
        p_value = 1.0 if n_participants == 1 else np.nan
    else:
        try:
            wilcoxon_stat, p_value = wilcoxon(deltas.values, alternative="two-sided")
        except ValueError:
            wilcoxon_stat = np.nan
            p_value = 1.0

    return {
        "n_participants": n_participants,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_stat": wilcoxon_stat,
        "p_value": p_value,
        "participant_deltas": deltas,
    }


def predict_on_test_set(model_name, bundle, params, history_cols):
    """Refit on train/val and return held-out test predictions."""
    data_kwargs = resolve_training_data(
        model_name,
        bundle,
        feature_set="history",
        history_cols=history_cols,
    )
    factory = make_model_factory(model_name, params, ORDINAL_MODELS)
    model = factory()
    model.fit(data_kwargs["X_train_val"], data_kwargs["y_train_val"])
    return model.predict(data_kwargs["X_test"])


def compare_history_feature_count_significance(
    bundle,
    model_names,
    history7_params,
    history3_params,
    history7_cols=None,
    history3_cols=None,
    n_bootstrap=10000,
    seed=RANDOM_STATE,
):
    """Compare 3-col vs 7-col history with participant-level paired significance tests."""
    if history7_cols is None:
        history7_cols = list(HISTORY_CANDIDATE_FEATURES)
    if history3_cols is None:
        history3_cols = list(HISTORY_FEATURES)

    if isinstance(model_names, dict):
        model_names = list(model_names.keys())

    rows = []
    y_test = bundle.y_ord_test
    participant_ids = bundle.groups_test

    for base in model_names:
        key7 = f"{base}_history7"
        key3 = f"{base}_history3"
        if key7 not in history7_params or key3 not in history3_params:
            continue

        pred7 = predict_on_test_set(
            base,
            bundle,
            history7_params[key7],
            history7_cols,
        )
        pred3 = predict_on_test_set(
            base,
            bundle,
            history3_params[key3],
            history3_cols,
        )

        test_mae7 = compute_metrics(y_test, pred7)["mae"]
        test_mae3 = compute_metrics(y_test, pred3)["mae"]
        test_result = paired_participant_significance_test(
            y_test,
            pred7,
            pred3,
            participant_ids,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        rows.append(
            {
                "model": base,
                "delta_mae_3_minus_7": test_mae3 - test_mae7,
                "mean_participant_delta": test_result["mean_delta"],
                "ci_low": test_result["ci_low"],
                "ci_high": test_result["ci_high"],
                "wilcoxon_p": test_result["p_value"],
            }
        )

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows).set_index("model")
    summary["fdr_p"] = apply_fdr(summary["wilcoxon_p"].fillna(1.0).to_numpy())
    return summary.sort_values("delta_mae_3_minus_7")
