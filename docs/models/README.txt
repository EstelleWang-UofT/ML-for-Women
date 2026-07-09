Fatigue Modeling — Documentation
=================================

Shared pipeline context for all models. Per-model details:
  baselines.txt      — persistence and naive baselines
  ordinal.txt        — ordinal regression and history/hybrid models
  classification.txt — high-fatigue classifiers and history/hybrid models

Implementations live under src/modeling/.


TASKS
-----

Ordinal regression:
  Target: fatigue_num (integer 0–5)
  Primary metric: MAE (mean absolute error)
  Additional metrics: RMSE, R2, quadratic weighted Cohen's kappa (QWK)

Binary classification:
  Target: high_fatigue (1 if fatigue_num >= 4, else 0)
  Primary metric: F1 (pos_label=1)
  Additional metrics: accuracy, precision, recall


DATA
----

Source file: mcphases/merged/physical_activity_merged_processed.csv
Loader: modeling.data.load_fatigue_data()

Preprocessing (merged preprocess.ipynb):
  1. Participant-level median for 9 activity/HR columns (pre-split, no cross-participant leak)
  2. menstrual_health_literacy_num left as NaN through export

Post-split imputation (modeling.data.prepare_splits):
  After the participant hold-out, literacy NaNs are filled with the train_val median only.
  Test rows never contribute to that median.

Rough scale: ~3,331 daily rows, 42 participants.
study_interval is retained for wave-aware temporal features but is not a model input.


FEATURES
--------

Numeric (NUMERIC_FEATURES):
  lightly, moderately, very, calories_sum,
  filtered_demographic_vo2_max, cardio_zone, fat_burn_zone,
  below_fat_burn_zone, lh_smooth, estrogen_smooth,
  age_of_first_menarche, age
  Note: lightly, moderately, very are log1p-transformed in the processed CSV.

Categorical (CATEGORICAL_FEATURES):
  is_weekend, phase, exerciselevel_num,
  menstrual_health_literacy_num, sexually_active_num

Sequence features (SEQUENCE_FEATURE_COLUMNS):
  NUMERIC_FEATURES plus is_weekend, exerciselevel_num,
  menstrual_health_literacy_num, sexually_active_num
  (phase encoded as integer index at sequence build time)

Phase order: Menstrual, Follicular, Fertility, Luteal

History features (HISTORY_FEATURES — past days only, per participant wave;
  grouped by id and study_interval):
  fatigue_lag1, fatigue_expanding_mean, fatigue_ewma,
  activity_logsum_roll3_mean (rolling mean of prior log1p lightly + log1p moderately + log1p very),
  calories_sum_roll3_mean, very_roll3_mean, fatigue_delta_lag1

Constants: EWMA_ALPHA=0.3, ROLLING_WINDOW=3 (defaults; tuned for catboost_history),
  EWMA_ALPHA_RANGE=(0.1, 0.5), ROLLING_WINDOW_CHOICES=[2, 3, 5, 7],
  STABILITY_SEEDS=[42..51], PRIOR_COL="__prior__"


TRAIN / TEST SPLIT
------------------

Function: modeling.data.prepare_splits(stratify=True by default)

After splitting, fills menstrual_health_literacy_num NaNs using the train_val median.

1. Compute per-participant high-fatigue rate (fatigue_num >= 4).
2. Assign participants to tertile strata via pd.qcut on that rate.
3. Hold out ~20% of participants (TEST_SIZE=0.2, RANDOM_STATE=42) using
   StratifiedShuffleSplit on strata labels.
4. All rows from held-out participants go to test; no row-level leakage.


CROSS-VALIDATION
----------------

Function: modeling.cv.run_group_cv / run_group_cv_sequences

Protocol: GroupKFold (N_CV_FOLDS=5) on train_val participants only.
Test participant ids are excluded from all CV folds.
Each fold trains on some participants, validates on disjoint participants.


HYPERPARAMETER TUNING
---------------------

Function: modeling.tuning.tune_model / tune_all_models
Library: Optuna (OPTUNA_TRIALS=30 per model)
Objective: minimize mean CV MAE (ordinal) or maximize mean CV F1 (classification)
Search spaces: src/modeling/registry.py get_search_space()

After tuning, final models are refit on all train_val data and evaluated once
on the held-out test participants.
