"""Data loading, splitting, and feature matrix construction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modeling.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EWMA_ALPHA,
    FEATURE_COLUMNS,
    HISTORY_CANDIDATE_FEATURES,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    PHASE_ORDER,
    PRIOR_COL,
    RANDOM_STATE,
    ROLLING_WINDOW_COLUMNS,
    ROLLING_WINDOWS,
    TEST_SIZE,
    TIME_COL,
    TIME_SERIES_GROUP_COLS,
)


def resolve_rolling_windows(rolling_windows=None):
    """Return per-column rolling windows from config or an override dict."""
    if rolling_windows is None:
        return {col: int(ROLLING_WINDOWS[col]) for col in ROLLING_WINDOW_COLUMNS}
    return {
        col: int(rolling_windows.get(col, ROLLING_WINDOWS[col]))
        for col in ROLLING_WINDOW_COLUMNS
    }


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
    groups_train_val: pd.Series
    groups_test: pd.Series
    y_lag1_train_val: pd.Series
    y_lag1_test: pd.Series
    y_expanding_train_val: pd.Series
    y_expanding_test: pd.Series


def load_fatigue_data(path=DATA_PATH):
    return pd.read_csv(path)


def _resolve_group_cols(df, group_cols=None):
    """Return wave-aware grouping columns for temporal features."""
    if group_cols is None:
        group_cols = TIME_SERIES_GROUP_COLS
    group_cols = list(group_cols)
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns required for wave-aware temporal features: "
            f"{missing}. Re-export physical_activity_merged_processed.csv "
            "with study_interval from merged preprocess.ipynb."
        )
    return group_cols


def _sort_by_group_time(df, group_cols, time_col):
    return df.sort_values(group_cols + [time_col])


def participant_strata(
    df,
    id_col="id",
    target_col="fatigue_num",
):
    """Per-participant stratification labels based on mean fatigue_num."""
    mean_fatigue = df.groupby(id_col)[target_col].mean()
    try:
        return pd.qcut(mean_fatigue, q=3, duplicates="drop")
    except ValueError:
        median = mean_fatigue.median()
        return (mean_fatigue >= median).astype(int)


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


def preprocess_after_split(
    df,
    train_val_mask,
    literacy_col="menstrual_health_literacy_num",
):
    """Post-split preprocessing fit on train/val rows only, applied to all rows."""
    if literacy_col not in df.columns or not df[literacy_col].isna().any():
        return df
    df = df.copy()
    literacy_median = df.loc[train_val_mask, literacy_col].median()
    df[literacy_col] = df[literacy_col].fillna(literacy_median)
    return df


def build_split_bundle(
    df,
    train_val_ids,
    test_ids,
    train_val_mask,
    test_mask,
    ewma_alpha=EWMA_ALPHA,
    rolling_windows=None,
):
    """Build feature matrices and targets from a preprocessed dataframe and split masks."""
    windows = resolve_rolling_windows(rolling_windows)
    X_all = make_feature_matrix(df)
    X_all_history = make_feature_matrix_with_history(
        df, ewma_alpha=ewma_alpha, rolling_windows=windows
    )

    X_train_val = X_all.loc[train_val_mask].reset_index(drop=True)
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    X_train_val_tree = build_tree_matrix(X_train_val)
    X_test_tree = build_tree_matrix(X_test)

    X_train_val_history = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test_history = X_all_history.loc[test_mask].reset_index(drop=True)
    X_train_val_history_tree = build_tree_matrix(X_train_val_history)
    X_test_history_tree = build_tree_matrix(X_test_history)

    y_ordinal = df["fatigue_num"].astype(int)
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
        groups_train_val=df.loc[train_val_mask, "id"].reset_index(drop=True),
        groups_test=df.loc[test_mask, "id"].reset_index(drop=True),
        y_lag1_train_val=y_lag1.loc[train_val_mask].reset_index(drop=True),
        y_lag1_test=y_lag1.loc[test_mask].reset_index(drop=True),
        y_expanding_train_val=y_expanding_train_val,
        y_expanding_test=y_expanding_test,
    )


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
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
):
    """Previous-day fatigue per participant wave (NaN on each wave's first day)."""
    group_cols = _resolve_group_cols(df, group_cols)
    ordered = _sort_by_group_time(df, group_cols, time_col)
    lag1 = ordered.groupby(group_cols, sort=False)[target_col].shift(1)
    return lag1.reindex(df.index)


def compute_expanding_mean_prior(
    df,
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
):
    """Mean fatigue from earlier days in the same wave (NaN on first day)."""
    group_cols = _resolve_group_cols(df, group_cols)
    ordered = _sort_by_group_time(df, group_cols, time_col)

    def _expanding_prior(series):
        shifted = series.shift(1)
        return shifted.expanding(min_periods=1).mean()

    expanding = ordered.groupby(group_cols, sort=False)[target_col].transform(
        _expanding_prior
    )
    return expanding.reindex(df.index)


def compute_fatigue_lag2(
    df,
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
):
    """Fatigue from two days ago per wave (NaN on first two days)."""
    group_cols = _resolve_group_cols(df, group_cols)
    ordered = _sort_by_group_time(df, group_cols, time_col)
    lag2 = ordered.groupby(group_cols, sort=False)[target_col].shift(2)
    return lag2.reindex(df.index)


def compute_fatigue_ewma_prior(
    df,
    alpha=EWMA_ALPHA,
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
):
    """EWMA of prior fatigue per wave (NaN until first prior day exists)."""
    group_cols = _resolve_group_cols(df, group_cols)
    ordered = _sort_by_group_time(df, group_cols, time_col)
    ewma = ordered.groupby(group_cols, sort=False)[target_col].transform(
        lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
    )
    return ewma.reindex(df.index)


def compute_fatigue_delta_lag1(
    df,
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
):
    """Change in fatigue from two days ago to yesterday (lag1 - lag2)."""
    lag1 = compute_fatigue_lag1(df, group_cols, time_col, target_col)
    lag2 = compute_fatigue_lag2(df, group_cols, time_col, target_col)
    return lag1 - lag2


def compute_rolling_mean_prior(
    df,
    col,
    window=3,
    group_cols=None,
    time_col=TIME_COL,
):
    """Rolling mean of a column over prior days within each wave."""
    group_cols = _resolve_group_cols(df, group_cols)
    ordered = _sort_by_group_time(df, group_cols, time_col)
    rolled = ordered.groupby(group_cols, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    return rolled.reindex(df.index)


def compute_log1p_activity_sum(df):
    """Sum of log1p-transformed lightly, moderately, and very columns."""
    return df["lightly"] + df["moderately"] + df["very"]


def compute_history_features(
    df,
    group_cols=None,
    time_col=TIME_COL,
    target_col="fatigue_num",
    ewma_alpha=None,
    rolling_windows=None,
):
    """Return leakage-safe history columns aligned to df.index."""
    if ewma_alpha is None:
        ewma_alpha = EWMA_ALPHA
    windows = resolve_rolling_windows(rolling_windows)

    activity_logsum = compute_log1p_activity_sum(df)
    work = pd.DataFrame(index=df.index)
    work["fatigue_lag1"] = compute_fatigue_lag1(df, group_cols, time_col, target_col)
    work["fatigue_expanding_mean"] = compute_expanding_mean_prior(
        df, group_cols, time_col, target_col
    )
    work["fatigue_ewma"] = compute_fatigue_ewma_prior(
        df, ewma_alpha, group_cols, time_col, target_col
    )
    activity_df = df.assign(_activity_logsum=activity_logsum)
    rolling_mean_sources = {
        "activity_logsum_roll_mean": (activity_df, "_activity_logsum"),
        "calories_sum_roll_mean": (df, "calories_sum"),
        "very_roll_mean": (df, "very"),
    }
    for col in ROLLING_WINDOW_COLUMNS:
        source_df, source_col = rolling_mean_sources[col]
        work[col] = compute_rolling_mean_prior(
            source_df,
            source_col,
            windows[col],
            group_cols,
            time_col,
        )
    work["fatigue_delta_lag1"] = compute_fatigue_delta_lag1(
        df, group_cols, time_col, target_col
    )
    return work[HISTORY_CANDIDATE_FEATURES]


def make_feature_matrix_with_history(data, ewma_alpha=None, rolling_windows=None):
    X = make_feature_matrix(data)
    history = compute_history_features(
        data,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
    )
    return pd.concat([X, history], axis=1)


def build_history_tree_matrices(
    df,
    train_val_mask,
    test_mask,
    ewma_alpha=None,
    rolling_windows=None,
):
    """Build tree-encoded history feature matrices for train/val and test splits."""
    X_all_history = make_feature_matrix_with_history(
        df,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
    )
    X_train_val_history = X_all_history.loc[train_val_mask].reset_index(drop=True)
    X_test_history = X_all_history.loc[test_mask].reset_index(drop=True)
    return (
        build_tree_matrix(X_train_val_history),
        build_tree_matrix(X_test_history),
    )


def impute_history_features(
    X_train_val,
    X_test,
    history_cols=HISTORY_FEATURES,
):
    """Fill history-column NaNs using train_val medians only."""
    X_tv = X_train_val.copy()
    X_te = X_test.copy()
    present = [col for col in history_cols if col in X_tv.columns]
    if not present:
        return X_tv, X_te
    medians = X_tv[present].median(numeric_only=True)
    for col in present:
        X_tv[col] = X_tv[col].fillna(medians[col])
        X_te[col] = X_te[col].fillna(medians[col])
    return X_tv, X_te


def _subset_history_columns(X_train_val, X_test, history_cols):
    """Drop history columns not in history_cols (None keeps all present history cols)."""
    if history_cols is None:
        return X_train_val, X_test
    drop = [
        col
        for col in HISTORY_CANDIDATE_FEATURES
        if col in X_train_val.columns and col not in history_cols
    ]
    if not drop:
        return X_train_val, X_test
    return X_train_val.drop(columns=drop), X_test.drop(columns=drop)


def history_feature_matrices(bundle, history_cols=None):
    """Return imputed raw and tree history matrices for train/val and test."""
    if history_cols is not None and len(history_cols) == 0:
        return (
            bundle.X_train_val,
            bundle.X_test,
            bundle.X_train_val_tree,
            bundle.X_test_tree,
        )
    X_tv, X_te = impute_history_features(
        bundle.X_train_val_history,
        bundle.X_test_history,
        history_cols=history_cols or HISTORY_FEATURES,
    )
    X_tv, X_te = _subset_history_columns(X_tv, X_te, history_cols)
    return X_tv, X_te, build_tree_matrix(X_tv), build_tree_matrix(X_te)


def attach_prior_frame(X, prior_series, col=PRIOR_COL):
    out = X.copy()
    out[col] = prior_series.values if hasattr(prior_series, "values") else prior_series
    return out


def make_wave_groups(df, mask):
    """Composite cluster id: one participant-interval (id × study_interval)."""
    wave = df.loc[mask, TIME_SERIES_GROUP_COLS]
    return (
        wave["id"].astype(int).astype(str)
        + "_"
        + wave["study_interval"].astype(int).astype(str)
    ).reset_index(drop=True)


def prepare_splits(
    df,
    test_size=TEST_SIZE,
    seed=RANDOM_STATE,
    stratify=True,
    ewma_alpha=None,
    rolling_windows=None,
):
    if ewma_alpha is None:
        ewma_alpha = EWMA_ALPHA

    df = df.sort_values(TIME_SERIES_GROUP_COLS + [TIME_COL]).reset_index(drop=True)

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

    df = preprocess_after_split(df, train_val_mask)

    return build_split_bundle(
        df,
        train_val_ids,
        test_ids,
        train_val_mask,
        test_mask,
        ewma_alpha=ewma_alpha,
        rolling_windows=rolling_windows,
    )


def split_summary_table(bundle: SplitBundle):
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
            }
        )
    return pd.DataFrame(rows)
