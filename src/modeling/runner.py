"""Single-model tune + benchmark pipeline for notebook cells."""

import numpy as np
import pandas as pd

from modeling.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, N_CV_FOLDS, OPTUNA_TRIALS
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


DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE = [
    "linear_regression",
    "elasticnet_regression",
    "svr_regression",
    "ordinal_rf",
    "catboost_regressor",
]


def _importance_label_for_model(model_name):
    labels = {
        "ordinal_rf": "Gini",
        "catboost_regressor": "CatBoost",
        "linear_regression": "Coef",
        "elasticnet_regression": "Coef",
        "svr_regression": "Permutation",
    }
    return labels.get(model_name, "Importance")


def _base_features_from_bundle(bundle):
    return list(FEATURE_COLUMNS)


def _is_tree_model(name):
    return name in TREE_ORDINAL_MODELS


def _tree_columns_for_base_feature(base_feature, all_columns):
    if base_feature == "phase":
        return [c for c in all_columns if c == "phase" or str(c).startswith("phase_")]
    return [base_feature] if base_feature in all_columns else []


def _columns_to_drop_for_base_feature(base_feature, all_columns):
    return _tree_columns_for_base_feature(base_feature, all_columns)


def _transformed_name_to_base_feature(name):
    if name.startswith("num__"):
        return name.split("__", 1)[1]
    if name.startswith("cat__"):
        rest = name.split("__", 1)[1]
        for base in CATEGORICAL_FEATURES:
            if rest == base or rest.startswith(f"{base}_"):
                return base
        return rest
    for base in CATEGORICAL_FEATURES:
        if name == base or name.startswith(f"{base}_"):
            return base
    return name


def _aggregate_importance_to_base(importance_df):
    rows = []
    for _, row in importance_df.iterrows():
        base = _transformed_name_to_base_feature(row["feature"])
        rows.append({"feature": base, "importance": row["importance"]})
    grouped = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def _tree_native_importance_df(model, X):
    importances = model.regressor.feature_importances_
    raw = pd.DataFrame({"feature": X.columns, "importance": importances})
    rows = []
    base_features = _base_features_from_bundle_from_columns(X.columns)
    for base in base_features:
        cols = _tree_columns_for_base_feature(base, X.columns)
        if not cols:
            continue
        importance = raw.loc[raw["feature"].isin(cols), "importance"].sum()
        rows.append({"feature": base, "importance": importance})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def _base_features_from_bundle_from_columns(columns):
    cols = set(columns)
    return [f for f in FEATURE_COLUMNS if f in cols or f == "phase" and any(c.startswith("phase_") for c in cols)]


def _pipeline_coef_importance_df(model):
    names = model.prep_.get_feature_names_out()
    coefs = np.abs(model.model_.coef_.ravel())
    raw = pd.DataFrame({"feature": names, "importance": coefs})
    return _aggregate_importance_to_base(raw)


def _svr_permutation_importance_df(model, X, y, random_state=42):
    from sklearn.inspection import permutation_importance

    cols = _fit_columns_for_model(model, X)
    X_sub = X[cols]
    result = permutation_importance(
        model,
        X_sub,
        y,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    raw = pd.DataFrame({"feature": cols, "importance": result.importances_mean})
    return raw.sort_values("importance", ascending=False).reset_index(drop=True)


def _fit_columns_for_model(model, X):
    if hasattr(model, "feature_columns") and model.feature_columns is not None:
        return list(model.feature_columns)
    return list(X.columns)


def _native_importance_df(model_name, model, X, y):
    if model_name in {"ordinal_rf", "catboost_regressor"}:
        return _tree_native_importance_df(model, X)
    if model_name in {"linear_regression", "elasticnet_regression"}:
        return _pipeline_coef_importance_df(model)
    if model_name == "svr_regression":
        return _svr_permutation_importance_df(model, X, y)
    raise ValueError(f"No native importance for {model_name!r}")


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


def _top_k_overlap_detail(per_model, top_k, source="native"):
    df_key = "native_df" if source == "native" else "ablation_df"
    feature_models = {}
    for model_name, result in per_model.items():
        for rank, feature in enumerate(result[df_key]["feature"].head(top_k).tolist(), start=1):
            feature_models.setdefault(feature, {})[model_name] = rank
    return feature_models


def _print_top_k_overlap_detail(per_model, top_k, label, source):
    n_models = len(per_model)
    feature_models = _top_k_overlap_detail(per_model, top_k, source=source)
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
    """Native + ablation importance per model, plus cross-model consensus table."""
    if models is None:
        models = list(DEFAULT_ALL_MODELS_FEATURE_IMPORTANCE)

    base_features = _base_features_from_bundle(bundle)
    per_model = {}

    for model_name in models:
        params = ordinal_best_params.get(model_name)
        if params is None:
            continue

        data_kwargs = resolve_training_data(
            model_name, bundle, feature_set=feature_set, history_cols=history_cols
        )
        X = data_kwargs["X_train_val"]
        y = data_kwargs["y_train_val"]

        model = make_model_factory(model_name, params, registry)()
        model.fit(X, y)
        native_df = _native_importance_df(model_name, model, X, y)

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
            "native_df": native_df,
            "ablation_df": ablation_df,
            "all_features_cv_mae": all_mae,
            "importance_label": _importance_label_for_model(model_name),
        }

    consensus_rows = []
    for feature in base_features:
        row = {"feature": feature}
        native_top_counts = 0
        ablation_top_counts = 0
        ablation_ranks = []

        for model_name, result in per_model.items():
            native_rank = _feature_rank_map(result["native_df"]["feature"].tolist()).get(feature)
            ablation_rank = _feature_rank_map(result["ablation_df"]["feature"].tolist()).get(feature)
            row[f"{model_name}_native_rank"] = native_rank
            row[f"{model_name}_ablation_rank"] = ablation_rank
            if native_rank is not None and native_rank <= top_k:
                native_top_counts += 1
            if ablation_rank is not None and ablation_rank <= top_k:
                ablation_top_counts += 1
            if ablation_rank is not None:
                ablation_ranks.append(ablation_rank)

        row["n_models_top5_native"] = native_top_counts
        row["n_models_top5_ablation"] = ablation_top_counts
        row["in_all_top5_ablation"] = ablation_top_counts == len(per_model)
        row["mean_ablation_rank"] = float(np.mean(ablation_ranks)) if ablation_ranks else np.nan
        consensus_rows.append(row)

    consensus = pd.DataFrame(consensus_rows).sort_values(
        ["n_models_top5_ablation", "mean_ablation_rank"],
        ascending=[False, True],
    )
    return per_model, consensus.reset_index(drop=True)


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
    """Print native + ablation importance for each §3 model and cross-model consensus."""
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
        label = result["importance_label"]
        print(f"\n{'=' * 60}")
        print(f"{model_name} — native ({label})")
        print(f"{'=' * 60}")
        display(result["native_df"])
        print(f"\n{model_name} — ablation (all-{len(_base_features_from_bundle(bundle))}-feature CV MAE: "
              f"{result['all_features_cv_mae']:.4f})")
        display(result["ablation_df"])

    n_models = len(per_model)
    n_native_col = "n_models_top5_native"
    n_ablation_col = "n_models_top5_ablation"
    in_all_col = "in_all_top5_ablation"

    print(f"\n{'=' * 60}")
    print(f"Cross-model consensus (top-{top_k}, {n_models} models)")
    print(f"{'=' * 60}")
    display(consensus)

    threshold = max(1, n_models - 1)

    all_native_top = consensus.loc[consensus[n_native_col] == n_models, "feature"].tolist()
    print(f"\nFeatures in top-{top_k} native for all {n_models} models:")
    print(all_native_top if all_native_top else "(none)")

    most_native_top = consensus.loc[
        consensus[n_native_col] >= threshold, "feature"
    ].tolist()
    print(f"\nFeatures in top-{top_k} native for ≥{threshold} models:")
    print(most_native_top if most_native_top else "(none)")

    all_top = consensus.loc[consensus[in_all_col], "feature"].tolist()
    print(f"\nFeatures in top-{top_k} ablation for all {n_models} models:")
    print(all_top if all_top else "(none)")

    most_top = consensus.loc[
        consensus[n_ablation_col] >= threshold, "feature"
    ].tolist()
    print(f"\nFeatures in top-{top_k} ablation for ≥{threshold} models:")
    print(most_top if most_top else "(none)")

    _print_top_k_overlap_detail(
        per_model, top_k, "Native importances", source="native"
    )
    _print_top_k_overlap_detail(
        per_model, top_k, "Ablation", source="ablation"
    )
