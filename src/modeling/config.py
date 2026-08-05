"""Configuration constants for fatigue modeling."""

import numpy as np

DATA_PATH = "mcphases/merged/physical_activity_merged_processed.csv"

NUMERIC_FEATURES = [
    "lightly_activity",
    "moderately_activity",
    "very_activity",
    "calories_sum",
    "filtered_demographic_vo2_max",
    "cardio_zone",
    "fat_burn_zone",
    "below_fat_burn_zone",
    "lh_smooth",
    "estrogen_smooth",
    "age_of_first_menarche",
    "age",
    "daily_glucose",
    "daily_hrv",
    "sleep_score",
]

CATEGORICAL_FEATURES = [
    "is_weekend",
    "phase",
    "exerciselevel_num",
    "menstrual_health_literacy_num",
    "sexually_active_num",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TIME_SERIES_GROUP_COLS = ["id", "study_interval"]
TIME_COL = "day_in_study"

PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_FOLDS = 5
OPTUNA_TRIALS = 30
GEE_OPTUNA_TRIALS = 10
ELASTICNET_OPTUNA_TRIALS = 50
SVR_OPTUNA_TRIALS = 50
CUTPOINT_ALTERNATING_ITERATIONS = 5

POST_SPLIT_MEDIAN_COLS = (
    "menstrual_health_literacy_num",
    "daily_hrv",
)

ORDINAL_METRIC = "mae"

HISTORY_CANDIDATE_FEATURES = [
    "fatigue_lag1",
    "fatigue_expanding_mean",
    "fatigue_ewma",
    "activity_logsum_roll_mean",
    "calories_sum_roll_mean",
    "very_active_roll_mean",
    "fatigue_delta_lag1",
]

# FE forward-selection subset; use HISTORY_CANDIDATE_FEATURES for full history runs.
HISTORY_FEATURES = [
    "fatigue_ewma",
    "fatigue_lag1",
    "fatigue_expanding_mean",
]

# Best construction from history feature engineering.ipynb (ordinal_rf proxy; §1 grid + §2 forward selection).
EWMA_ALPHA = 0.142997
# Window length per rolling column is in ROLLING_WINDOWS; column names are window-agnostic.
ROLLING_WINDOW_COLUMNS = [
    "activity_logsum_roll_mean",
    "calories_sum_roll_mean",
    "very_active_roll_mean",
]
ROLLING_WINDOWS = {
    "activity_logsum_roll_mean": 2,
    "calories_sum_roll_mean": 2,
    "very_active_roll_mean": 3,
}
ROLLING_WINDOW_PARAM_NAMES = {
    "activity_logsum_roll_mean": "activity_roll_window",
    "calories_sum_roll_mean": "calories_roll_window",
    "very_active_roll_mean": "very_active_roll_window",
}
EWMA_ALPHA_RANGE = (0.1, 0.5)
EWMA_ALPHA_GRID = np.geomspace(EWMA_ALPHA_RANGE[0], EWMA_ALPHA_RANGE[1], 10).tolist()
ROLLING_WINDOW_CHOICES = [2, 3, 5, 7]
HISTORY_ABLATION_MODEL = "ordinal_rf"

# Fixed proxy hyperparams for history FE notebook grid search and ablation (not full model Optuna).
# Optuna best from ordinal_rf_history in 4 fatigue regression.ipynb (min_samples_leaf not in saved output).
HISTORY_PROXY_PARAMS = {
    "n_estimators": 396,
    "max_depth": 3,
    "min_samples_leaf": 1,
}
PRIOR_COL = "__prior__"
