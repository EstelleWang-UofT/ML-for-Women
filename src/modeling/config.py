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
SEQUENCE_FEATURE_COLUMNS = NUMERIC_FEATURES + [
    "is_weekend",
    "exerciselevel_num",
    "menstrual_health_literacy_num",
    "sexually_active_num",
]

PHASE_ORDER = ["Menstrual", "Follicular", "Fertility", "Luteal"]
PHASE_TO_IDX = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_FOLDS = 5
HIGH_FATIGUE_THRESHOLD = 4
OPTUNA_TRIALS = 30

ORDINAL_METRIC = "mae"
MULTICLASS_METRIC = "weighted_f1"

HISTORY_FEATURES = [
    "fatigue_lag1",
    "fatigue_expanding_mean",
    "fatigue_ewma",
    "activity_logsum_roll3_mean",
    "calories_sum_roll3_mean",
    "very_roll3_mean",
    "fatigue_delta_lag1",
]

EWMA_ALPHA = 0.3
ROLLING_WINDOW = 3
EWMA_ALPHA_RANGE = (0.1, 0.5)
ROLLING_WINDOW_CHOICES = [2, 3, 5, 7]
PRIOR_COL = "__prior__"

STABILITY_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
STABILITY_MODELS = ["expanding_mean", "catboost_history", "catboost_residual_expanding"]
