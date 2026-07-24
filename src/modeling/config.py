"""Configuration constants for fatigue modeling."""

DATA_PATH = "mcphases/merged/physical_activity_merged_processed.csv"

NUMERIC_FEATURES = [
    "lightly",
    "moderately",
    "very",
    "calories_sum",
    "filtered_demographic_vo2_max",
    "cardio_zone",
    "fat_burn_zone",
    "below_fat_burn_zone",
    "lh_smooth",
    "estrogen_smooth",
    "age_of_first_menarche",
    "age",
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

ORDINAL_METRIC = "mae"

HISTORY_FEATURES = [
    "fatigue_ewma",
    "fatigue_expanding_mean",
    "fatigue_lag1",
]

EWMA_ALPHA = 0.366976
ROLLING_WINDOW_COLUMNS = [
    "activity_logsum_roll3_mean",
    "calories_sum_roll3_mean",
    "very_roll3_mean",
]
ROLLING_WINDOWS = {
    "activity_logsum_roll3_mean": 2,
    "calories_sum_roll3_mean": 2,
    "very_roll3_mean": 5,
}
ROLLING_WINDOW_PARAM_NAMES = {
    "activity_logsum_roll3_mean": "activity_roll_window",
    "calories_sum_roll3_mean": "calories_roll_window",
    "very_roll3_mean": "very_roll_window",
}
EWMA_ALPHA_RANGE = (0.1, 0.5)
ROLLING_WINDOW_CHOICES = [2, 3, 5, 7]
HISTORY_TUNING_TRIALS = 20
HISTORY_ABLATION_MODEL = "catboost_ordinal"

# Optuna best from 3 fatigue_modeling.ipynb §3 catboost_ordinal_history
# (default history construction; update after re-tuning main notebook)
HISTORY_PROXY_PARAMS = {
    "iterations": 366,
    "depth": 4,
    "learning_rate": 0.034143215054019314,
    "l2_leaf_reg": 3.5798039531025863,
}
PRIOR_COL = "__prior__"
