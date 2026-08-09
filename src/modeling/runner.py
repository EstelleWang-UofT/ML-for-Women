"""Single-model tune + benchmark pipeline for notebook cells."""

import numpy as np
import pandas as pd

from modeling.config import FEATURE_COLUMNS, N_CV_FOLDS, OPTUNA_TRIALS
from modeling.cv import run_group_cv, run_model_benchmark
from modeling.metrics import compute_metrics, compute_regression_metrics
from modeling.registry import (
    CONTINUOUS_REGRESSION_MODELS,
    PIPELINE_ORDINAL_MODELS,
    TREE_ORDINAL_MODELS,
    get_search_space,
    make_model_factory,
    resolve_training_data,
)
from modeling.tuning import tune_model

try:
    from IPython.display import display
except ImportError:
    display = print


def tune_and_benchmark_model(
    name,
    bundle,
    registry,
    feature_set="base",
    display_name=None,
    history_cols=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    params=None,
):
    """Tune (optional) + GroupKFold benchmark + test eval for one model."""
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    metrics_fn = (
        compute_regression_metrics
        if name in CONTINUOUS_REGRESSION_MODELS
        else compute_metrics
    )
    cv_score = None

    if params is None:
        tune_kwargs = {
            k: v
            for k, v in data_kwargs.items()
            if k in {"X_train_val", "y_train_val", "groups"}
        }
        params, cv_score = tune_model(
            name=name,
            registry=registry,
            search_space=get_search_space(name),
            n_trials=n_trials,
            n_splits=n_splits,
            test_ids=bundle.test_ids,
            **tune_kwargs,
        )

    factory = make_model_factory(name, params, registry)
    result = run_model_benchmark(
        name=display_name or name,
        model_factory=factory,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
        y_train_val=data_kwargs["y_train_val"],
        y_test=data_kwargs["y_test"],
        groups=data_kwargs["groups"],
        X_train_val=data_kwargs["X_train_val"],
        X_test=data_kwargs["X_test"],
        metrics_fn=metrics_fn,
    )
    result["name"] = display_name or name
    result["best_params"] = params
    result["best_cv_score"] = cv_score
    result["feature_set"] = feature_set
    result["history_cols"] = history_cols
    result["metric_cols"] = list(result["test_metrics"].keys())
    return result, params


def print_tune_summary(display_name, result, params):
    cv_mae = result["cv_summary"].loc["mean", "mae"]
    test_mae = result["test_metrics"]["mae"]
    print(f"[ok] {display_name}")
    print(f"  best_params: {params}")
    print(f"  cv_mae: {cv_mae:.4f}")
    print(f"  test_mae: {test_mae:.4f}")


DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE = [
    "linear_regression",
    "ordinal_rf",
]


def _refit_test_predictions(
    name,
    bundle,
    registry,
    params,
    feature_set="base",
    history_cols=None,
):
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    model = make_model_factory(name, params, registry)()
    model.fit(data_kwargs["X_train_val"], data_kwargs["y_train_val"])
    return model.predict(data_kwargs["X_test"])


def collect_continuous_test_predictions(
    bundle,
    ordinal_best_params,
    registry,
    models=None,
    feature_set="base",
    history_cols=None,
):
    """Refit each model on train/val and collect continuous test predictions."""
    if models is None:
        models = list(DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE)

    preds_by_name = {}
    y_test = None

    for name in models:
        params = ordinal_best_params.get(name)
        if params is None:
            continue
        data_kwargs = resolve_training_data(
            name, bundle, feature_set=feature_set, history_cols=history_cols
        )
        if y_test is None:
            y_test = np.asarray(data_kwargs["y_test"])
        preds = _refit_test_predictions(
            name,
            bundle,
            registry,
            params,
            feature_set=feature_set,
            history_cols=history_cols,
        )
        preds_by_name[name] = np.asarray(preds, dtype=float)

    if y_test is None or not preds_by_name:
        raise ValueError("No tuned model params found; run §3 first.")

    wide_df = pd.DataFrame({"y_true": y_test})
    for name, preds in preds_by_name.items():
        wide_df[name] = preds
    return y_test, preds_by_name, wide_df


def _continuous_prediction_stats(preds):
    """Summary stats verifying predictions are continuous floats."""
    preds = np.asarray(preds, dtype=float)
    rounded = np.round(preds, 6)
    non_integer = preds != np.rint(preds)
    return {
        "dtype": str(preds.dtype),
        "n_test": len(preds),
        "min": float(preds.min()),
        "max": float(preds.max()),
        "mean": float(preds.mean()),
        "std": float(preds.std()),
        "n_unique_raw": len(np.unique(rounded)),
        "frac_non_integer": float(non_integer.mean()),
    }


def _continuous_prediction_stats_table(preds_by_name):
    rows = []
    for name, preds in preds_by_name.items():
        row = _continuous_prediction_stats(preds)
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_continuous_test_predictions(
    y_test,
    preds_by_name,
    ax=None,
    figsize=(18, 8),
):
    """Histogram + y_true vs pred scatter for each model (2 rows x n_models cols)."""
    import matplotlib.pyplot as plt

    names = list(preds_by_name.keys())
    n_models = len(names)
    if n_models == 0:
        raise ValueError("preds_by_name is empty")

    y_test = np.asarray(y_test)
    if ax is not None:
        fig = ax.figure
        axes = np.array([[ax]])
    else:
        fig, axes = plt.subplots(2, n_models, figsize=figsize, squeeze=False)

    integer_ticks = list(range(6))

    for col, name in enumerate(names):
        preds = np.asarray(preds_by_name[name], dtype=float)
        stats = _continuous_prediction_stats(preds)

        ax_hist = axes[0, col]
        ax_hist.hist(preds, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        for tick in integer_ticks:
            ax_hist.axvline(tick, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_hist.set_title(f"{name}\nfrac_non_integer={stats['frac_non_integer']:.3f}")
        ax_hist.set_xlabel("predicted value")
        if col == 0:
            ax_hist.set_ylabel("count")

        ax_scatter = axes[1, col]
        ax_scatter.scatter(y_test, preds, alpha=0.35, s=12, color="darkorange")
        lim_lo = min(0, float(preds.min()), float(y_test.min())) - 0.25
        lim_hi = max(5, float(preds.max()), float(y_test.max())) + 0.25
        ax_scatter.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", linewidth=0.8)
        ax_scatter.set_xlim(lim_lo, lim_hi)
        ax_scatter.set_ylim(lim_lo, lim_hi)
        ax_scatter.set_xlabel("y_true")
        if col == 0:
            ax_scatter.set_ylabel("predicted value")
        ax_scatter.set_title(f"{name} scatter")

    fig.suptitle(f"Continuous test predictions (n={len(y_test)})", fontsize=12)
    fig.subplots_adjust(top=0.90, hspace=0.45, wspace=0.30)
    return fig, axes


def print_and_plot_continuous_test_predictions(
    bundle,
    ordinal_best_params,
    registry,
    models=None,
    feature_set="base",
    history_cols=None,
    figsize=(18, 8),
):
    """Print per-model stats table and plot histogram/scatter per model."""
    import matplotlib.pyplot as plt

    y_test, preds_by_name, _wide_df = collect_continuous_test_predictions(
        bundle,
        ordinal_best_params,
        registry,
        models=models,
        feature_set=feature_set,
        history_cols=history_cols,
    )

    stats = _continuous_prediction_stats_table(preds_by_name)
    print("[continuous verification] per-model prediction stats")
    display(stats)

    collapsed = stats.loc[stats["frac_non_integer"] == 0.0]
    if not collapsed.empty:
        print("Warning: models with zero non-integer predictions (may be discrete collapse):")
        print(collapsed.index.tolist())

    plot_continuous_test_predictions(
        y_test,
        preds_by_name,
        figsize=figsize,
    )
    plt.show()


def _match_matrix_lower_triangle(match_df):
    mask_upper = np.triu(np.ones(match_df.shape, dtype=bool), k=1)
    return match_df.where(~mask_upper)


def _style_prediction_match(df):
    _SAME = frozenset({"Same", "Same as global_mode"})
    _DIFF = frozenset({"Different", "Different from global_mode"})

    def _cell_style(val):
        if val in _SAME:
            return "background-color: #c6efce; color: #006100"
        if val in _DIFF:
            return "background-color: #ffc7ce; color: #9c0006"
        return ""

    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(_cell_style)
    return styler.applymap(_cell_style)


def print_ordinal_test_prediction_diagnostic(
    bundle,
    ordinal_results,
    ordinal_best_params,
    registry,
    feature_set="base",
    history_cols=None,
):
    """Compare §3 models' continuous test predictions (identical-vector check)."""
    from modeling.baselines import build_global_mode_baseline

    if not ordinal_results:
        print("No ordinal_results to diagnose.")
        return

    preds_by_name = {}
    summary_rows = []
    y_test = None

    for result in ordinal_results:
        name = result["name"]
        params = ordinal_best_params.get(name) or result.get("best_params")
        data_kwargs = resolve_training_data(
            name, bundle, feature_set=feature_set, history_cols=history_cols
        )
        if y_test is None:
            y_test = data_kwargs["y_test"]
        preds = _refit_test_predictions(
            name,
            bundle,
            registry,
            params,
            feature_set=feature_set,
            history_cols=history_cols,
        )
        preds_by_name[name] = np.asarray(preds, dtype=float)
        test_mae = float(compute_regression_metrics(y_test, preds)["mae"])
        summary_rows.append(
            {
                "model": name,
                "test_mae": test_mae,
                "n_unique_preds": len(np.unique(np.round(preds, 4))),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("model")
    print("[diagnostic] §3 continuous test predictions")
    display(summary)

    names = list(preds_by_name.keys())
    match = pd.DataFrame(False, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i <= j:
                same = np.allclose(preds_by_name[a], preds_by_name[b], rtol=0, atol=1e-9)
                match.loc[a, b] = same
                match.loc[b, a] = same

    match_labels = match.replace({True: "Same", False: "Different"})
    match_lower = _match_matrix_lower_triangle(match_labels)
    print("Pairwise identical test predictions (lower triangle; green = same vector)")
    display(_style_prediction_match(match_lower))

    ref_name = names[0]
    data_kwargs = resolve_training_data(
        ref_name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    global_mode = build_global_mode_baseline()
    global_mode.fit(data_kwargs["X_train_val"], data_kwargs["y_train_val"])
    global_preds = global_mode.predict(data_kwargs["X_test"])
    global_mae = float(
        compute_metrics(data_kwargs["y_test"], global_preds)["mae"]
    )

    vs_global = pd.Series(
        {
            name: bool(np.array_equal(np.rint(preds), global_preds))
            for name, preds in preds_by_name.items()
        },
        name="matches_global_mode",
    )
    print(
        f"global_mode test_mae = {global_mae:.4f} "
        "(green = same discretized test vector as global_mode)"
    )
    vs_global_df = vs_global.map(
        {True: "Same as global_mode", False: "Different from global_mode"}
    ).to_frame()
    display(_style_prediction_match(vs_global_df))


def _base_features_from_bundle(bundle):
    return list(FEATURE_COLUMNS)


def _tree_columns_for_base_feature(base_feature, all_columns):
    if base_feature == "phase":
        return [c for c in all_columns if c == "phase" or str(c).startswith("phase_")]
    return [base_feature] if base_feature in all_columns else []


def _columns_to_drop_for_base_feature(base_feature, all_columns):
    return _tree_columns_for_base_feature(base_feature, all_columns)


def _base_features_from_bundle_from_columns(columns):
    cols = set(columns)
    return [f for f in FEATURE_COLUMNS if f in cols or f == "phase" and any(c.startswith("phase_") for c in cols)]


def _make_factory_with_columns(name, params, registry, columns):
    def factory():
        model = make_model_factory(name, params, registry)()
        if name in PIPELINE_ORDINAL_MODELS:
            model.feature_columns = list(columns)
        return model

    return factory


def _cv_mae_for_columns(
    name,
    params,
    registry,
    X,
    y,
    groups,
    columns,
    n_splits,
    test_ids,
    metrics_fn=compute_regression_metrics,
):
    if name in TREE_ORDINAL_MODELS:
        factory = make_model_factory(name, params, registry)
        X_use = X[columns]
    elif name in PIPELINE_ORDINAL_MODELS:
        factory = _make_factory_with_columns(name, params, registry, columns)
        X_use = X
    else:
        factory = make_model_factory(name, params, registry)
        X_use = X[columns]

    _, cv_summary = run_group_cv(
        factory,
        X_use,
        y,
        groups,
        n_splits=n_splits,
        test_ids=test_ids,
        metrics_fn=metrics_fn,
    )
    return float(cv_summary.loc["mean", "mae"])


def run_base_feature_ablation(
    name,
    bundle,
    params,
    registry,
    feature_set="base",
    history_cols=None,
    n_splits=N_CV_FOLDS,
    ablate_base_features=False,
):
    """Leave-one-out GroupKFold ablation over columns or logical base features."""
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    X = data_kwargs["X_train_val"]
    y = data_kwargs["y_train_val"]
    groups = data_kwargs["groups"]
    metrics_fn = (
        compute_regression_metrics
        if name in CONTINUOUS_REGRESSION_MODELS
        else compute_metrics
    )

    all_columns = list(X.columns)

    def cv_mae_for_columns(columns):
        return _cv_mae_for_columns(
            name,
            params,
            registry,
            X,
            y,
            groups,
            columns,
            n_splits,
            bundle.test_ids,
            metrics_fn=metrics_fn,
        )

    all_features_cv_mae = cv_mae_for_columns(all_columns)

    if ablate_base_features:
        base_features = _base_features_from_bundle_from_columns(all_columns)
        rows = []
        for feature in base_features:
            drop_cols = _columns_to_drop_for_base_feature(feature, all_columns)
            subset = [col for col in all_columns if col not in drop_cols]
            cv_mae_without = cv_mae_for_columns(subset)
            rows.append(
                {
                    "feature": feature,
                    "cv_mae_without_feature": cv_mae_without,
                    "cv_mae_increase_vs_all": cv_mae_without - all_features_cv_mae,
                }
            )
    else:
        rows = []
        for feature in all_columns:
            subset = [col for col in all_columns if col != feature]
            cv_mae_without = cv_mae_for_columns(subset)
            rows.append(
                {
                    "feature": feature,
                    "cv_mae_without_feature": cv_mae_without,
                    "cv_mae_increase_vs_all": cv_mae_without - all_features_cv_mae,
                }
            )

    feature_importance = pd.DataFrame(rows).sort_values(
        "cv_mae_increase_vs_all", ascending=False
    )
    return all_features_cv_mae, feature_importance.reset_index(drop=True)


def _feature_rank_map(feature_order):
    return {feature: rank + 1 for rank, feature in enumerate(feature_order)}


def _top_k_overlap_detail(per_model, top_k):
    feature_models = {}
    for model_name, result in per_model.items():
        for rank, feature in enumerate(
            result["ablation_df"]["feature"].head(top_k).tolist(), start=1
        ):
            feature_models.setdefault(feature, {})[model_name] = rank
    return feature_models


def _print_top_k_overlap_detail(per_model, top_k, label):
    n_models = len(per_model)
    feature_models = _top_k_overlap_detail(per_model, top_k)
    by_count = {}
    for feature, model_ranks in feature_models.items():
        by_count.setdefault(len(model_ranks), []).append((feature, model_ranks))

    print(f"{label} (top-{top_k} overlap):")
    for count in sorted(by_count.keys(), reverse=True):
        print(f"  In top-{top_k} for {count} model(s):")
        for feature, model_ranks in sorted(by_count[count], key=lambda x: x[0]):
            parts = [f"{m} (#{r})" for m, r in sorted(model_ranks.items(), key=lambda x: x[1])]
            print(f"    {feature}: {', '.join(parts)}")
    if not feature_models:
        print("  (none)")


def summarize_all_models_feature_importance(
    ordinal_best_params,
    registry,
    bundle,
    models=None,
    top_k=5,
    feature_set="base",
    history_cols=None,
    n_splits=N_CV_FOLDS,
):
    """Leave-one-out ablation importance per model, plus cross-model consensus table."""
    if models is None:
        models = list(DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE)

    base_features = _base_features_from_bundle(bundle)
    per_model = {}

    for model_name in models:
        params = ordinal_best_params.get(model_name)
        if params is None:
            continue

        all_mae, ablation_df = run_base_feature_ablation(
            model_name,
            bundle,
            params,
            registry,
            feature_set=feature_set,
            history_cols=history_cols,
            n_splits=n_splits,
            ablate_base_features=True,
        )
        per_model[model_name] = {
            "ablation_df": ablation_df,
            "all_features_cv_mae": all_mae,
        }

    consensus_rows = []
    for feature in base_features:
        row = {"feature": feature}
        for model_name, result in per_model.items():
            ablation_df = result["ablation_df"]
            val = ablation_df.loc[
                ablation_df["feature"] == feature, "cv_mae_increase_vs_all"
            ]
            row[f"{model_name}_cv_mae_increase_vs_all"] = (
                float(val.iloc[0]) if len(val) else np.nan
            )
        consensus_rows.append(row)

    consensus = pd.DataFrame(consensus_rows)
    sort_cols = [
        f"{model_name}_cv_mae_increase_vs_all"
        for model_name in models
        if model_name in per_model
    ]
    if "linear_regression" in per_model and "ordinal_rf" in per_model:
        sort_cols = [
            "ordinal_rf_cv_mae_increase_vs_all",
            "linear_regression_cv_mae_increase_vs_all",
        ]
    consensus = consensus.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    if "linear_regression" in per_model and "ordinal_rf" in per_model:
        consensus = consensus[["feature"] + sort_cols]
    return per_model, consensus


def print_all_models_feature_importance(
    bundle,
    ordinal_best_params,
    registry,
    models=None,
    top_k=5,
    feature_set="base",
    history_cols=None,
    n_splits=N_CV_FOLDS,
):
    """Print leave-one-out ablation importance for each §3 model and cross-model consensus."""
    if models is None:
        models = list(DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE)

    per_model, consensus = summarize_all_models_feature_importance(
        ordinal_best_params,
        registry,
        bundle,
        models=models,
        top_k=top_k,
        feature_set=feature_set,
        history_cols=history_cols,
        n_splits=n_splits,
    )

    for model_name in models:
        if model_name not in per_model:
            print(f"[skip] {model_name}: no tuned params")
            continue
        result = per_model[model_name]
        print(f"\n{'=' * 60}")
        print(f"{model_name} — ablation (all-{len(_base_features_from_bundle(bundle))}-feature CV MAE: "
              f"{result['all_features_cv_mae']:.4f})")
        print(f"{'=' * 60}")
        display(result["ablation_df"][["feature", "cv_mae_increase_vs_all"]])

    print(f"\n{'=' * 60}")
    print("Cross-model consensus (cv_mae_increase_vs_all)")
    print(f"{'=' * 60}")
    display(consensus.round(4))

    _print_top_k_overlap_detail(per_model, top_k, "Ablation")
