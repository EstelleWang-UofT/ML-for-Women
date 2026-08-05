"""Single-model tune + benchmark pipeline for notebook cells."""

import numpy as np
import pandas as pd

from modeling.config import CUTPOINT_ALTERNATING_ITERATIONS, N_CV_FOLDS, OPTUNA_TRIALS
from modeling.cv import run_group_cv, run_model_benchmark
from modeling.calibration import (
    benchmark_cutpoint_calibration,
    cutpoints_differ_from_default,
    make_model_factory_with_cutpoints,
)
from modeling.registry import get_search_space, make_model_factory, resolve_training_data
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
    cutpoints=None,
):
    """Tune (optional) + GroupKFold benchmark + test eval for one model.

    When ``cutpoints`` is provided, hyperparameter search and final benchmark
    use ``CutpointRegressorWrapper`` so CV/test MAE reflects fixed cutpoint
    discretization rather than rint. Used in §3 cell 3 (cutpoint-integrated retune).
    """
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
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
            cutpoints=cutpoints,
            **tune_kwargs,
        )

    X_train_val = data_kwargs["X_train_val"]
    X_test = data_kwargs["X_test"]
    if cutpoints is None:
        factory = make_model_factory(name, params, registry)
    else:
        factory = make_model_factory_with_cutpoints(name, params, cutpoints, registry)
    result = run_model_benchmark(
        name=display_name or name,
        model_factory=factory,
        n_splits=n_splits,
        test_ids=bundle.test_ids,
        y_train_val=data_kwargs["y_train_val"],
        y_test=data_kwargs["y_test"],
        groups=data_kwargs["groups"],
        X_train_val=X_train_val,
        X_test=X_test,
    )
    result["name"] = display_name or name
    result["best_params"] = params
    result["best_cv_score"] = cv_score
    result["feature_set"] = feature_set
    result["history_cols"] = history_cols
    result["cutpoints"] = cutpoints
    return result, params


def print_tune_summary(display_name, result, params):
    cv_mae = result["cv_summary"].loc["mean", "mae"]
    test_mae = result["test_metrics"]["mae"]
    print(f"[ok] {display_name}")
    print(f"  best_params: {params}")
    print(f"  cv_mae: {cv_mae:.4f}")
    print(f"  test_mae: {test_mae:.4f}")


def print_cutpoint_comparison(display_name, tune_result, cutpoint_result):
    """Compare rint pipeline metrics vs post-hoc tuned cutpoints."""
    best_cutpoints = cutpoint_result["best_cutpoints"]
    rint_cv = tune_result["cv_summary"].loc["mean"].to_dict()
    rint_test = tune_result["test_metrics"]
    tuned_cv = cutpoint_result["oof_tuned_metrics"]
    tuned_test = cutpoint_result["test_tuned_metrics"]

    print(f"[cutpoints] {display_name}")
    print(f"  tuned cutpoints: {best_cutpoints.tolist()}")
    delta_test_mae = tuned_test["mae"] - rint_test["mae"]
    print(f"  test MAE delta (tuned - rint): {delta_test_mae:+.4f}")

    cv_compare = pd.DataFrame(
        [rint_cv, tuned_cv],
        index=["rint (pipeline)", "tuned (OOF)"],
    )
    test_compare = pd.DataFrame(
        [rint_test, tuned_test],
        index=["rint", "tuned"],
    )
    print("CV metrics")
    display(cv_compare)
    print("Test metrics")
    display(test_compare)


def run_cutpoint_alternating_iterations(
    name,
    bundle,
    registry,
    n_iterations=CUTPOINT_ALTERNATING_ITERATIONS,
    feature_set="base",
    history_cols=None,
    n_trials=OPTUNA_TRIALS,
    n_splits=N_CV_FOLDS,
    verbose=True,
):
    """Alternating Optuna tune + OOF cutpoint search for up to ``n_iterations`` rounds.

    Always runs iteration 1 (rint tune + cutpoint calibration). Iterations 2–5 run
    only when iteration-1 cutpoints differ from the rint default.
    """
    cutpoints = None
    rows = []
    final_result = None
    final_params = None
    continued = False

    for iteration in range(1, n_iterations + 1):
        if verbose:
            print(f"[iter {iteration}/{n_iterations}] {name}")

        result, params = tune_and_benchmark_model(
            name,
            bundle,
            registry,
            feature_set=feature_set,
            history_cols=history_cols,
            n_trials=n_trials,
            n_splits=n_splits,
            cutpoints=cutpoints,
            display_name=f"{name} (iter {iteration})",
        )
        cv_mae = float(result["cv_summary"].loc["mean", "mae"])
        test_mae = float(result["test_metrics"]["mae"])

        cut = benchmark_cutpoint_calibration(
            name,
            bundle,
            params,
            registry=registry,
            feature_set=feature_set,
            n_splits=n_splits,
        )
        cutpoints = cut["best_cutpoints"]

        rows.append(
            {
                "iteration": iteration,
                "cv_mae": cv_mae,
                "test_mae": test_mae,
                "cutpoints": cutpoints.tolist(),
            }
        )
        final_result = result
        final_params = params

        if iteration == 1 and not cutpoints_differ_from_default(cutpoints):
            if verbose:
                print(
                    f"{name}: cutpoint values unchanged after iteration 1 "
                    f"(rint default); skipping iterations 2–5."
                )
            break

        if iteration == 1:
            continued = True

    if final_result is not None:
        final_result["name"] = name

    history = pd.DataFrame(rows)
    return {
        "history": history,
        "final_result": final_result,
        "final_params": final_params,
        "final_cutpoints": cutpoints,
        "continued": continued,
        "n_iterations_run": len(rows),
    }


def print_cutpoint_iteration_table(display_name, history_df):
    """Display iteration history of pipeline cv_mae and test_mae."""
    table = history_df[["iteration", "cv_mae", "test_mae"]].copy()
    table["cv_mae"] = table["cv_mae"].map(lambda x: f"{x:.4f}")
    table["test_mae"] = table["test_mae"].map(lambda x: f"{x:.4f}")
    print(f"[iterations] {display_name}")
    display(table)


def plot_cutpoint_iteration_mae(
    history_df,
    display_name,
    ax=None,
    figsize=(8, 4),
):
    """Line plot of cv_mae and test_mae vs iteration with value labels on each point."""
    import matplotlib.pyplot as plt

    if history_df.empty:
        raise ValueError("history_df is empty")

    x = history_df["iteration"]
    series = [
        ("cv_mae", "steelblue", (0, 6)),
        ("test_mae", "darkorange", (0, -12)),
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for col, color, offset in series:
        y = history_df[col]
        ax.plot(x, y, marker="o", color=color, linewidth=1.2, label=col)
        for xi, yi in zip(x, y):
            ax.annotate(
                f"{yi:.4f}",
                (xi, yi),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE")
    ax.set_title(f"{display_name}: pipeline MAE by iteration")
    ax.set_xticks(x)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig = ax.figure
    plt.tight_layout()
    return fig, ax


def _refit_test_predictions(
    name,
    bundle,
    registry,
    params,
    cutpoints=None,
    feature_set="base",
    history_cols=None,
):
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    if cutpoints is None:
        factory = make_model_factory(name, params, registry)
    else:
        factory = make_model_factory_with_cutpoints(name, params, cutpoints, registry)
    model = factory()
    model.fit(data_kwargs["X_train_val"], data_kwargs["y_train_val"])
    return model.predict(data_kwargs["X_test"])


def _match_matrix_lower_triangle(match_df):
    """Keep lower triangle (incl. diagonal); mask strict upper triangle."""
    mask_upper = np.triu(np.ones(match_df.shape, dtype=bool), k=1)
    return match_df.where(~mask_upper)


def _style_prediction_match(df):
    """Green/red Styler for Same/Different labels (masked cells unstyled)."""
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
    """Compare §3 models' test predictions and cutpoints (identical-MAE check).

    Pairwise comparison uses a colored lower-triangle matrix (green = identical
    test prediction vectors).
    """
    from modeling.baselines import build_global_mode_baseline
    from modeling.metrics import compute_metrics

    if not ordinal_results:
        print("No ordinal_results to diagnose.")
        return

    preds_by_name = {}
    summary_rows = []
    y_test = None

    for result in ordinal_results:
        name = result["name"]
        params = ordinal_best_params.get(name) or result.get("best_params")
        cutpoints = result.get("cutpoints")
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
            cutpoints=cutpoints,
            feature_set=feature_set,
            history_cols=history_cols,
        )
        preds_by_name[name] = np.asarray(preds)
        test_mae = float(compute_metrics(y_test, preds)["mae"])

        summary_rows.append(
            {
                "model": name,
                "test_mae": test_mae,
                "n_unique_preds": len(np.unique(preds)),
                "cutpoints": None
                if cutpoints is None
                else np.asarray(cutpoints, dtype=float).round(4).tolist(),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("model")
    print("[diagnostic] §3 test predictions")
    display(summary)

    names = list(preds_by_name.keys())
    match = pd.DataFrame(False, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i <= j:
                same = np.array_equal(preds_by_name[a], preds_by_name[b])
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
            name: bool(np.array_equal(preds, global_preds))
            for name, preds in preds_by_name.items()
        },
        name="matches_global_mode",
    )
    print(f"global_mode test_mae = {global_mae:.4f} (green = same test vector as global_mode)")
    vs_global_df = vs_global.map(
        {True: "Same as global_mode", False: "Different from global_mode"}
    ).to_frame()
    display(_style_prediction_match(vs_global_df))


def run_base_feature_ablation(
    name,
    bundle,
    params,
    registry,
    feature_set="base",
    history_cols=None,
    n_splits=N_CV_FOLDS,
):
    """Leave-one-out GroupKFold ablation over base/tree feature columns."""
    data_kwargs = resolve_training_data(
        name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    X = data_kwargs["X_train_val"]
    y = data_kwargs["y_train_val"]
    groups = data_kwargs["groups"]

    def cv_mae_for_columns(columns):
        factory = make_model_factory(name, params, registry)
        _, cv_summary = run_group_cv(
            factory,
            X[columns],
            y,
            groups,
            n_splits=n_splits,
            test_ids=bundle.test_ids,
        )
        return float(cv_summary.loc["mean", "mae"])

    all_columns = list(X.columns)
    all_features_cv_mae = cv_mae_for_columns(all_columns)

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


def print_ordinal_rf_gini_importance(
    bundle,
    ordinal_best_params,
    registry,
    model_name="ordinal_rf",
    feature_set="base",
    history_cols=None,
):
    """Display Gini importances from a refit ordinal_rf on train/val (§6.1)."""
    import matplotlib.pyplot as plt

    params = ordinal_best_params.get(model_name)
    if params is None:
        print(f"No params for {model_name}; run §3 first.")
        return

    data_kwargs = resolve_training_data(
        model_name, bundle, feature_set=feature_set, history_cols=history_cols
    )
    X = data_kwargs["X_train_val"]
    y = data_kwargs["y_train_val"]

    model = make_model_factory(model_name, params, registry)()
    model.fit(X, y)
    importances = model.regressor.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    display(importance_df)

    if importance_df.empty or importance_df["importance"].sum() <= 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importance_df))))
    importance_df.plot.bar(x="feature", y="importance", legend=False, ax=ax)
    ax.set_title(f"{model_name} Gini feature importance — base features")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def print_ordinal_rf_ablation_importance(
    bundle,
    ordinal_best_params,
    registry,
    model_name="ordinal_rf",
    feature_set="base",
    history_cols=None,
    n_splits=N_CV_FOLDS,
    plot=True,
):
    """Leave-one-out CV ablation table (+ bar chart) for ordinal_rf base features (§6.2)."""
    import matplotlib.pyplot as plt

    params = ordinal_best_params.get(model_name)
    if params is None:
        print(f"No params for {model_name}; run §3 first.")
        return None

    all_features_cv_mae, feature_importance = run_base_feature_ablation(
        model_name,
        bundle,
        params,
        registry,
        feature_set=feature_set,
        history_cols=history_cols,
        n_splits=n_splits,
    )
    n_features = len(feature_importance)
    print(f"All-{n_features}-feature CV MAE: {all_features_cv_mae:.4f}")
    print("Removing each feature — higher cv_mae_increase_vs_all = more important:")
    display(feature_importance)

    if plot and not feature_importance.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(feature_importance))))
        feature_importance.plot.bar(
            x="feature", y="cv_mae_increase_vs_all", legend=False, ax=ax
        )
        ax.set_title(f"{model_name} leave-one-out CV importance — base features")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return all_features_cv_mae, feature_importance
