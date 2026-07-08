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
CLASSIFICATION_METRIC = "f1"

HISTORY_FEATURES = [
    "fatigue_lag1",
    "fatigue_expanding_mean",
    "fatigue_ewma",
    "active_minutes_roll3_mean",
    "calories_sum_roll3_mean",
    "very_roll3_mean",
    "fatigue_delta_lag1",
]

HYBRID_HISTORY_FEATURES = [
    "fatigue_lag1",
    "fatigue_ewma",
    "fatigue_delta_lag1",
    "active_minutes_roll3_mean",
    "calories_sum_roll3_mean",
    "very_roll3_mean",
]

EWMA_ALPHA = 0.3
ROLLING_WINDOW = 3
PRIOR_COL = "__prior__"
