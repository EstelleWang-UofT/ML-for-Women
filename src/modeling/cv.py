"""Cross-validation and benchmark runners."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from modeling.data import SequenceData
from modeling.metrics import compute_metrics, summarize_fold_metrics


def _slice_sequence(seq_data: SequenceData, idx) -> SequenceData:
    idx = np.asarray(idx)
    return SequenceData(
        X=seq_data.X[idx],
        lengths=seq_data.lengths[idx],
        y=seq_data.y[idx],
        groups=seq_data.groups[idx],
        y_lag1=seq_data.y_lag1[idx] if seq_data.y_lag1 is not None else None,
        y_expanding_mean=seq_data.y_expanding_mean[idx]
        if seq_data.y_expanding_mean is not None
        else None,
    )


def run_group_cv(
    model_factory,
    X,
    y,
    groups,
    n_splits=5,
    task="ordinal",
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
        metrics = compute_metrics(y_series.iloc[val_idx], preds, task=task)
        metrics["fold"] = fold
        metrics["n_train_participants"] = len(train_group_ids)
        metrics["n_val_participants"] = len(val_group_ids)
        if task == "multiclass":
            metrics["val_mode_class"] = int(pd.Series(y_series.iloc[val_idx]).mode().iloc[0])
        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)
    metric_cols = [
        c
        for c in fold_df.columns
        if c
        not in {"fold", "n_train_participants", "n_val_participants", "val_mode_class"}
    ]
    return fold_df, summarize_fold_metrics(fold_df, metric_cols)


def run_group_cv_sequences(
    model_factory,
    seq_data: SequenceData,
    n_splits=5,
    task="ordinal",
    test_ids=None,
):
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    test_id_set = set(test_ids or [])
    groups_series = pd.Series(seq_data.groups)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(seq_data.X, seq_data.y, groups_series)):
        train_group_ids = set(groups_series.iloc[train_idx])
        val_group_ids = set(groups_series.iloc[val_idx])
        assert train_group_ids.isdisjoint(val_group_ids)
        if test_id_set:
            assert train_group_ids.isdisjoint(test_id_set)
            assert val_group_ids.isdisjoint(test_id_set)

        train_seq = _slice_sequence(seq_data, train_idx)
        val_seq = _slice_sequence(seq_data, val_idx)
        model = model_factory()
        model.fit(train_seq)
        preds = model.predict(val_seq)
        metrics = compute_metrics(val_seq.y, preds, task=task)
        metrics["fold"] = fold
        metrics["n_train_participants"] = len(train_group_ids)
        metrics["n_val_participants"] = len(val_group_ids)
        if task == "multiclass":
            metrics["val_mode_class"] = int(pd.Series(val_seq.y).mode().iloc[0])
        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)
    metric_cols = [
        c
        for c in fold_df.columns
        if c
        not in {"fold", "n_train_participants", "n_val_participants", "val_mode_class"}
    ]
    return fold_df, summarize_fold_metrics(fold_df, metric_cols)


def evaluate_on_test(
    model_factory,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
    task="ordinal",
):
    model = model_factory()
    model.fit(X_train_val, y_train_val)
    preds = model.predict(X_test)
    return compute_metrics(y_test, preds, task=task)


def evaluate_sequence_on_test(model_factory, seq_train_val, seq_test, task="ordinal"):
    model = model_factory()
    model.fit(seq_train_val)
    preds = model.predict(seq_test)
    return compute_metrics(seq_test.y, preds, task=task)


def run_model_benchmark(
    name,
    model_factory,
    y_train_val,
    groups,
    y_test,
    X_train_val=None,
    X_test=None,
    task="ordinal",
    n_splits=5,
    test_ids=None,
    use_sequences=False,
    seq_train_val=None,
    seq_test=None,
):
    if use_sequences:
        if seq_train_val is None or seq_test is None:
            raise ValueError("seq_train_val and seq_test are required when use_sequences=True")
        fold_df, cv_summary = run_group_cv_sequences(
            model_factory,
            seq_train_val,
            n_splits=n_splits,
            task=task,
            test_ids=test_ids,
        )
        test_metrics = evaluate_sequence_on_test(
            model_factory, seq_train_val, seq_test, task=task
        )
    else:
        fold_df, cv_summary = run_group_cv(
            model_factory,
            X_train_val,
            y_train_val,
            groups,
            n_splits=n_splits,
            task=task,
            test_ids=test_ids,
        )
        test_metrics = evaluate_on_test(
            model_factory,
            X_train_val,
            y_train_val,
            X_test,
            y_test,
            task=task,
        )

    return {
        "status": "ok",
        "name": name,
        "fold_df": fold_df,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }
