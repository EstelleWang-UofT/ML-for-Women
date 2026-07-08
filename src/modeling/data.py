"""Data loading, splitting, and feature matrix construction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modeling.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EWMA_ALPHA,
    FEATURE_COLUMNS,
    HIGH_FATIGUE_THRESHOLD,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    PHASE_ORDER,
    PHASE_TO_IDX,
    PRIOR_COL,
    RANDOM_STATE,
    ROLLING_WINDOW,
    SEQUENCE_FEATURE_COLUMNS,
    TEST_SIZE,
)


@dataclass
class SequenceData:
    X: np.ndarray
    lengths: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    y_lag1: np.ndarray | None = None
    y_expanding_mean: np.ndarray | None = None


@dataclass
class SplitBundle:
    df: pd.DataFrame
    train_val_ids: set
    test_ids: set
    train_val_mask: pd.Series
    test_mask: pd.Series
    X_train_val: pd.DataFrame
    X_test: pd.DataFrame
    X_train_val_tree: pd.DataFrame
    X_test_tree: pd.DataFrame
    X_train_val_history: pd.DataFrame
    X_test_history: pd.DataFrame
    X_train_val_history_tree: pd.DataFrame
    X_test_history_tree: pd.DataFrame
    y_ord_train_val: pd.Series
    y_ord_test: pd.Series
    y_clf_train_val: pd.Series
    y_clf_test: pd.Series
    groups_train_val: pd.Series
    groups_test: pd.Series
    y_lag1_train_val: pd.Series
    y_lag1_test: pd.Series
    y_expanding_train_val: pd.Series
    y_expanding_test: pd.Series
    seq_train_val: SequenceData
    seq_test: SequenceData


def load_fatigue_data(path=DATA_PATH):
    return pd.read_csv(path)


def participant_strata(
    df,
    id_col="id",
    target_col="fatigue_num",
    threshold=HIGH_FATIGUE_THRESHOLD,
):
    """Per-participant stratification labels based on high-fatigue rate."""
    stats = df.groupby(id_col)[target_col].agg(
        high_fatigue_rate=lambda s: (s >= threshold).mean(),
        mean_fatigue="mean",
    )
    try:
        stats["stratum"] = pd.qcut(
            stats["high_fatigue_rate"],
            q=3,
            duplicates="drop",
        )
    except ValueError:
        median_rate = stats["high_fatigue_rate"].median()
        stats["stratum"] = (stats["high_fatigue_rate"] >= median_rate).astype(int)
    return stats["stratum"]


def split_participant_ids(
    unique_ids,
    test_size=TEST_SIZE,
    seed=RANDOM_STATE,
    strata=None,
):
    """Hold out a participant-level test set, optionally stratified by `strata`."""
    ids = np.array(sorted(unique_ids))
    n_test = max(1, int(round(len(ids) * test_size)))
    test_fraction = n_test / len(ids)

    if strata is not None:
        from sklearn.model_selection import StratifiedShuffleSplit

        stratum_labels = np.array([str(strata[id_]) for id_ in ids])
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(ids, stratum_labels))
        test_ids = set(ids[test_idx])
        train_val_ids = set(ids[train_idx])
        return train_val_ids, test_ids

    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    test_ids = set(shuffled[:n_test])
    train_val_ids = set(shuffled[n_test:])
    return train_val_ids, test_ids


def make_feature_matrix(
    data,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
):
    X = data[numeric_features + categorical_features].copy()
    if "phase" in X.columns:
        X["phase"] = pd.Categorical(X["phase"], categories=PHASE_ORDER, ordered=True)
    if "is_weekend" in X.columns:
        X["is_weekend"] = X["is_weekend"].astype(int)
    return X


def build_tree_matrix(X):
    X_tree = X.copy()
    if "phase" in X_tree.columns:
        X_tree["phase"] = X_tree["phase"].astype(str)
    return pd.get_dummies(X_tree, columns=["phase"], prefix="phase", dtype=int)


def compute_fatigue_lag1(
    df,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
):
    """Previous-day fatigue per participant (NaN on each participant's first day)."""
    ordered = df.sort_values([group_col, time_col])
    lag1 = ordered.groupby(group_col)[target_col].shift(1)
    return lag1.reindex(df.index)


def compute_expanding_mean_prior(
    df,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
):
    """Mean fatigue from earlier days in the same participant (NaN on first day)."""
    ordered = df.sort_values([group_col, time_col])

    def _expanding_prior(series):
        shifted = series.shift(1)
        return shifted.expanding(min_periods=1).mean()

    expanding = ordered.groupby(group_col)[target_col].transform(_expanding_prior)
    return expanding.reindex(df.index)


def compute_fatigue_lag2(
    df,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
):
    """Fatigue from two days ago per participant (NaN on first two days)."""
    ordered = df.sort_values([group_col, time_col])
    lag2 = ordered.groupby(group_col)[target_col].shift(2)
    return lag2.reindex(df.index)


def compute_fatigue_ewma_prior(
    df,
    alpha=EWMA_ALPHA,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
):
    """EWMA of prior fatigue per participant (NaN until first prior day exists)."""
    ordered = df.sort_values([group_col, time_col])
    ewma = ordered.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
    )
    return ewma.reindex(df.index)


def compute_fatigue_delta_lag1(
    df,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
):
    """Change in fatigue from two days ago to yesterday (lag1 - lag2)."""
    lag1 = compute_fatigue_lag1(df, group_col, time_col, target_col)
    lag2 = compute_fatigue_lag2(df, group_col, time_col, target_col)
    return lag1 - lag2


def compute_rolling_mean_prior(
    df,
    col,
    window=ROLLING_WINDOW,
    group_col="id",
    time_col="day_in_study",
):
    """Rolling mean of a column over prior days within each participant."""
    ordered = df.sort_values([group_col, time_col])
    rolled = ordered.groupby(group_col)[col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return rolled.reindex(df.index)


def compute_active_minutes(df):
    return df["lightly"] + df["moderately"] + df["very"]


def compute_history_features(
    df,
    group_col="id",
    time_col="day_in_study",
    target_col="fatigue_num",
    ewma_alpha=None,
    rolling_window=None,
):
    """Return leakage-safe history columns aligned to df.index."""
    if ewma_alpha is None:
        ewma_alpha = EWMA_ALPHA
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW

    active = compute_active_minutes(df)
    work = pd.DataFrame(index=df.index)
    work["fatigue_lag1"] = compute_fatigue_lag1(df, group_col, time_col, target_col)
    work["fatigue_expanding_mean"] = compute_expanding_mean_prior(
        df, group_col, time_col, target_col
    )
    work["fatigue_ewma"] = compute_fatigue_ewma_prior(
        df, ewma_alpha, group_col, time_col, target_col
    )
    active_df = df.assign(_active_minutes=active)
    work["active_minutes_roll3_mean"] = compute_rolling_mean_prior(
        active_df,
        "_active_minutes",
        rolling_window,
        group_col,
        time_col,
    )
    work["calories_sum_roll3_mean"] = compute_rolling_mean_prior(
        df, "calories_sum", rolling_window, group_col, time_col
    )
    work["very_roll3_mean"] = compute_rolling_mean_prior(
        df, "very", rolling_window, group_col, time_col
    )
    work["fatigue_delta_lag1"] = compute_fatigue_delta_lag1(
        df, group_col, time_col, target_col
    )
    return work[HISTORY_FEATURES]


def make_feature_matrix_with_history(data, ewma_alpha=None, rolling_window=None):
    X = make_feature_matrix(data)
    history = compute_history_features(
        data, ewma_alpha=ewma_alpha, rolling_window=rolling_window
    )
    return pd.concat([X, history], axis=1)


def build_history_tree_matrices(
    df,
    train_val_mask,
    test_mask,
    ewma_alpha=None,
    rolling_window=None,
):
    """Build tree-encoded history feature matrices for train/val and test splits."""
    X_all_history = make_feature_matrix_with_history(
        df, ewma_alpha=ewma_alpha, rolling_window=rolling_window
    )
    X_train_val_history = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test_history = X_all_history.loc[test_mask].reset_index(drop=True)
    return (
        build_tree_matrix(X_train_val_history),
        build_tree_matrix(X_test_history),
    )


def attach_prior_frame(X, prior_series, col=PRIOR_COL):
    out = X.copy()
    out[col] = prior_series.values if hasattr(prior_series, "values") else prior_series
    return out


def build_hybrid_residual_matrix(X_history, prior_series, prior_col=PRIOR_COL):
    """Tree matrix for hybrid models: drop expanding mean from X, attach prior column."""
    out = X_history.drop(columns=["fatigue_expanding_mean"], errors="ignore")
    out_tree = build_tree_matrix(out)
    return attach_prior_frame(out_tree, prior_series, col=prior_col)


def build_hybrid_tree_matrices(
    df,
    train_val_mask,
    test_mask,
    ewma_alpha=None,
    rolling_window=None,
):
    """Build tree-encoded hybrid feature matrices for train/val and test splits."""
    X_all_history = make_feature_matrix_with_history(
        df, ewma_alpha=ewma_alpha, rolling_window=rolling_window
    )
    y_expanding = compute_expanding_mean_prior(df)
    X_train_val_history = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test_history = X_all_history.loc[test_mask].reset_index(drop=True)
    y_expanding_train_val = y_expanding.loc[train_val_mask].reset_index(drop=True)
    y_expanding_test = y_expanding.loc[test_mask].reset_index(drop=True)
    return (
        build_hybrid_residual_matrix(X_train_val_history, y_expanding_train_val),
        build_hybrid_residual_matrix(X_test_history, y_expanding_test),
    )


def _encode_phase_series(phase_series):
    return phase_series.map(PHASE_TO_IDX).fillna(0).astype(int)


def build_sequence_tensors(
    data,
    feature_cols=SEQUENCE_FEATURE_COLUMNS,
    target_col="fatigue_num",
    group_col="id",
    time_col="day_in_study",
):
    work = data[[group_col, time_col, target_col] + [c for c in feature_cols if c in data.columns]].copy()
    if "phase" in data.columns:
        work["phase"] = _encode_phase_series(data["phase"])
        if "phase" not in feature_cols:
            feature_cols = list(feature_cols) + ["phase"]
    if "is_weekend" in work.columns:
        work["is_weekend"] = work["is_weekend"].astype(int)

    feature_cols = [c for c in feature_cols if c in work.columns]
    grouped = {}
    for group_id, gdf in work.groupby(group_col, sort=False):
        grouped[group_id] = gdf.sort_values(time_col)

    seq_rows = []
    y_list = []
    y_lag1_list = []
    y_expanding_list = []
    groups_list = []

    lag1_all = compute_fatigue_lag1(data, group_col, time_col, target_col)
    expanding_all = compute_expanding_mean_prior(data, group_col, time_col, target_col)

    for idx, row in work.iterrows():
        group_id = row[group_col]
        gdf = grouped[group_id]
        pos = gdf.index.get_loc(idx)
        if isinstance(pos, slice):
            pos = pos.start
        elif isinstance(pos, np.ndarray):
            pos = int(np.where(pos)[0][0])
        feats = gdf[feature_cols].astype(float).values
        seq_rows.append(feats[: pos + 1])
        y_list.append(int(row[target_col]))
        lag_val = lag1_all.loc[idx]
        exp_val = expanding_all.loc[idx]
        y_lag1_list.append(np.nan if pd.isna(lag_val) else float(lag_val))
        y_expanding_list.append(np.nan if pd.isna(exp_val) else float(exp_val))
        groups_list.append(group_id)

    n_samples = len(seq_rows)
    n_features = len(feature_cols)
    max_len = max(len(seq) for seq in seq_rows)
    X = np.zeros((n_samples, max_len, n_features), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int64)
    for i, seq in enumerate(seq_rows):
        lengths[i] = len(seq)
        X[i, : len(seq)] = seq

    return SequenceData(
        X=X,
        lengths=lengths,
        y=np.array(y_list, dtype=np.int64),
        groups=np.array(groups_list),
        y_lag1=np.array(y_lag1_list, dtype=np.float64),
        y_expanding_mean=np.array(y_expanding_list, dtype=np.float64),
    )


def _subset_sequence_data(seq_data: SequenceData, mask: np.ndarray) -> SequenceData:
    mask = np.asarray(mask, dtype=bool)
    return SequenceData(
        X=seq_data.X[mask],
        lengths=seq_data.lengths[mask],
        y=seq_data.y[mask],
        groups=seq_data.groups[mask],
        y_lag1=seq_data.y_lag1[mask] if seq_data.y_lag1 is not None else None,
        y_expanding_mean=seq_data.y_expanding_mean[mask]
        if seq_data.y_expanding_mean is not None
        else None,
    )


def prepare_splits(
    df,
    test_size=TEST_SIZE,
    seed=RANDOM_STATE,
    stratify=True,
    ewma_alpha=None,
    rolling_window=None,
):
    if ewma_alpha is None:
        ewma_alpha = EWMA_ALPHA
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW

    strata = participant_strata(df) if stratify else None
    train_val_ids, test_ids = split_participant_ids(
        df["id"].unique(),
        test_size,
        seed,
        strata=strata,
    )
    assert train_val_ids.isdisjoint(test_ids)

    train_val_mask = df["id"].isin(train_val_ids)
    test_mask = df["id"].isin(test_ids)

    X_all = make_feature_matrix(df)
    X_all_history = make_feature_matrix_with_history(
        df, ewma_alpha=ewma_alpha, rolling_window=rolling_window
    )
    seq_all = build_sequence_tensors(df)

    X_train_val = X_all.loc[train_val_mask].reset_index(drop=True)
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    X_train_val_tree = build_tree_matrix(X_train_val)
    X_test_tree = build_tree_matrix(X_test)

    X_train_val_history = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test_history = X_all_history.loc[test_mask].reset_index(drop=True)
    X_train_val_history_tree = build_tree_matrix(X_train_val_history)
    X_test_history_tree = build_tree_matrix(X_test_history)

    y_ordinal = df["fatigue_num"].astype(int)
    y_high_fatigue = (df["fatigue_num"] >= HIGH_FATIGUE_THRESHOLD).astype(int)
    y_lag1 = compute_fatigue_lag1(df)
    y_expanding = compute_expanding_mean_prior(df)

    y_expanding_train_val = y_expanding.loc[train_val_mask].reset_index(drop=True)
    y_expanding_test = y_expanding.loc[test_mask].reset_index(drop=True)

    return SplitBundle(
        df=df,
        train_val_ids=train_val_ids,
        test_ids=test_ids,
        train_val_mask=train_val_mask,
        test_mask=test_mask,
        X_train_val=X_train_val,
        X_test=X_test,
        X_train_val_tree=X_train_val_tree,
        X_test_tree=X_test_tree,
        X_train_val_history=X_train_val_history,
        X_test_history=X_test_history,
        X_train_val_history_tree=X_train_val_history_tree,
        X_test_history_tree=X_test_history_tree,
        y_ord_train_val=y_ordinal.loc[train_val_mask].reset_index(drop=True),
        y_ord_test=y_ordinal.loc[test_mask].reset_index(drop=True),
        y_clf_train_val=y_high_fatigue.loc[train_val_mask].reset_index(drop=True),
        y_clf_test=y_high_fatigue.loc[test_mask].reset_index(drop=True),
        groups_train_val=df.loc[train_val_mask, "id"].reset_index(drop=True),
        groups_test=df.loc[test_mask, "id"].reset_index(drop=True),
        y_lag1_train_val=y_lag1.loc[train_val_mask].reset_index(drop=True),
        y_lag1_test=y_lag1.loc[test_mask].reset_index(drop=True),
        y_expanding_train_val=y_expanding_train_val,
        y_expanding_test=y_expanding_test,
        seq_train_val=_subset_sequence_data(seq_all, train_val_mask.values),
        seq_test=_subset_sequence_data(seq_all, test_mask.values),
    )


def split_summary_table(bundle: SplitBundle, y_high_fatigue: pd.Series):
    rows = []
    for name, ids, mask in [
        ("train_val", bundle.train_val_ids, bundle.df["id"].isin(bundle.train_val_ids)),
        ("test", bundle.test_ids, bundle.df["id"].isin(bundle.test_ids)),
    ]:
        rows.append(
            {
                "split": name,
                "participants": len(ids),
                "rows": int(mask.sum()),
                "mean_fatigue": float(bundle.df.loc[mask, "fatigue_num"].mean()),
                "high_fatigue_rate": float(y_high_fatigue[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
