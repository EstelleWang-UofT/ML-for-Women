"""Cross-validation and benchmark runners."""

import pandas as pd
from sklearn.model_selection import GroupKFold

from modeling.metrics import compute_metrics, summarize_fold_metrics


def run_group_cv(
    model_factory,
    X,
    y,
    groups,
    n_splits=5,
    test_ids=None,
):
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    test_id_set = set(test_ids or [])

    y_series = pd.Series(y).reset_index(drop=True)
    groups_series = pd.Series(groups).reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_series, groups_series)):
        train_group_ids = set(groups_series.iloc[train_idx])
        val_group_ids = set(groups_series.iloc[val_idx])
        assert train_group_ids.isdisjoint(val_group_ids)
        if test_id_set:
            assert train_group_ids.isdisjoint(test_id_set)
            assert val_group_ids.isdisjoint(test_id_set)

        model = model_factory()
        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]

        model.fit(X_train, y_series.iloc[train_idx])
        preds = model.predict(X_val)
        metrics = compute_metrics(y_series.iloc[val_idx], preds)
        metrics["fold"] = fold
        metrics["n_train_participants"] = len(train_group_ids)
        metrics["n_val_participants"] = len(val_group_ids)
        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)
    metric_cols = [
        c
        for c in fold_df.columns
        if c not in {"fold", "n_train_participants", "n_val_participants"}
    ]
    return fold_df, summarize_fold_metrics(fold_df, metric_cols)


def evaluate_on_test(
    model_factory,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
):
    model = model_factory()
    model.fit(X_train_val, y_train_val)
    preds = model.predict(X_test)
    return compute_metrics(y_test, preds)


def run_model_benchmark(
    name,
    model_factory,
    y_train_val,
    groups,
    y_test,
    X_train_val=None,
    X_test=None,
    n_splits=5,
    test_ids=None,
):
    fold_df, cv_summary = run_group_cv(
        model_factory,
        X_train_val,
        y_train_val,
        groups,
        n_splits=n_splits,
        test_ids=test_ids,
    )
    test_metrics = evaluate_on_test(
        model_factory,
        X_train_val,
        y_train_val,
        X_test,
        y_test,
    )

    return {
        "status": "ok",
        "name": name,
        "fold_df": fold_df,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }
