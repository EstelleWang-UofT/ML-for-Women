Fatigue Modeling — Model Documentation Index
============================================

This folder contains implementation documentation for each model used in the
fatigue prediction pipeline. See individual .txt files for model-specific
details. All implementations live under src/modeling/.


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

Rough scale: ~3,331 daily rows, 42 participants.


FEATURES (from src/modeling/config.py)
--------------------------------------

Numeric (NUMERIC_FEATURES):
  lightly, moderately, very, calories_sum,
  filtered_demographic_vo2_max, cardio_zone, fat_burn_zone,
  below_fat_burn_zone, lh_smooth, estrogen_smooth,
  age_of_first_menarche, age

Categorical (CATEGORICAL_FEATURES):
  is_weekend, phase, exerciselevel_num,
  menstrual_health_literacy_num, sexually_active_num

Sequence features (SEQUENCE_FEATURE_COLUMNS):
  NUMERIC_FEATURES plus is_weekend, exerciselevel_num,
  menstrual_health_literacy_num, sexually_active_num
  (phase encoded as integer index at sequence build time)

Phase order: Menstrual, Follicular, Fertility, Luteal

History features (HISTORY_FEATURES — past days only, per participant):
  fatigue_lag1, fatigue_expanding_mean, fatigue_ewma,
  active_minutes_roll3_mean, calories_sum_roll3_mean,
  very_roll3_mean, fatigue_delta_lag1

Constants: EWMA_ALPHA=0.3, ROLLING_WINDOW=3, PRIOR_COL="__prior__"


TRAIN / TEST SPLIT
------------------

Function: modeling.data.prepare_splits(stratify=True by default)

1. Compute per-participant high-fatigue rate (fatigue_num >= 4).
2. Assign participants to tertile strata via pd.qcut on that rate.
3. Hold out ~20% of participants (TEST_SIZE=0.2, RANDOM_STATE=42) using
   StratifiedShuffleSplit on strata labels.
4. All rows from held-out participants go to test; no row-level leakage.

SplitBundle exposes train/val matrices, tree-encoded matrices, history
matrices (X_*_history_tree), hybrid residual matrices (X_*_hybrid with
__prior__), sequence tensors, lag1/expanding priors, and group columns.


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


DATA FLOW (text)
----------------

merged_processed.csv
  -> load_and_prepare (prepare_splits)
  -> participant-stratified train/test split
  -> build_feature_matrix (tabular, for sklearn pipelines)
  -> build_tree_matrix (phase one-hot for tree models)
  -> build_sequence_tensors (padded per-participant histories for LSTM)
  -> compute_fatigue_lag1 / compute_expanding_mean_prior (baseline priors)

Tabular ordinal: ordered_logistic, mixed_effects (raw matrix + groups)
Tree ordinal: ordinal_rf, catboost_ordinal (tree matrix)
Sequence ordinal: lstm (SequenceData)
Classification: lightgbm, random_forest (tree matrix)
Baselines: rule-based predictors (see baseline_*.txt)


MODEL FILES
-----------

Tuned ordinal models:
  ordered_logistic.txt
  ordinal_rf.txt
  catboost_ordinal.txt
  mixed_effects.txt
  lstm.txt

History / hybrid ordinal models:
  catboost_history.txt
  catboost_residual_expanding.txt

Tuned classification models:
  lightgbm.txt
  random_forest.txt

Baselines:
  baseline_global_mean.txt
  baseline_global_mode.txt
  baseline_lag1_fatigue.txt
  baseline_expanding_mean.txt
  baseline_majority_class.txt


NOTEBOOK
--------

End-to-end pipeline: notebooks/physical activity/fatigue_modeling.ipynb
Baselines run in section 1b; tuned models in section 2+.
Section 2a tunes catboost_history and catboost_residual_expanding.
Delta columns compare test metrics vs expanding_mean and lag1_fatigue.

Hybrid formula (catboost_residual_expanding):
  prediction = clip(expanding_mean + CatBoost_residual)
