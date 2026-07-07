import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")   # change to your folder path
OUTPUT_FILE = "mcphases/merged/sleep.csv"

# =========================================================
# LOAD CSV FILES
# =========================================================

sleep_data = pd.read_csv(DATA_DIR / "sleep.csv")

sleep_score = pd.read_csv(DATA_DIR / "sleep_score.csv")

hormones = pd.read_csv(
    DATA_DIR / "hormones_and_selfreport.csv"
)

# =========================================================
# KEEP ONLY RELEVANT COLUMNS
# (based on README feature descriptions)
# =========================================================

# ---------------------------------------------------------
# sleep.csv
# ---------------------------------------------------------

sleep_data = sleep_data[[
    "id",
    "sleep_start_day_in_study",
    "minutesasleep",
    "minutesawake",
    "minutesafterwakeup",
    "timeinbed",
    "efficiency",
    "minutestofallasleep",
    "duration",
    "mainsleep"
]]

sleep_data = sleep_data.rename(columns={
    "sleep_start_day_in_study": "day_in_study"
})

# Keep only main sleep session
sleep_data = sleep_data[sleep_data["mainsleep"] == True]

# Aggregate daily
sleep_data = (
    sleep_data
    .groupby(
        ["id", "day_in_study"],
        as_index=False
    )
    .agg({
        # cumulative quantities
        "minutesasleep": "sum",
        "minutesawake": "sum",
        "minutesafterwakeup": "sum",
        "timeinbed": "sum",
        "duration": "sum",

        # quality metrics
        "efficiency": "mean",
        "minutestofallasleep": "mean"
    })
    .rename(columns={
        "minutesasleep": "minutes_asleep",
        "minutesawake": "minutes_awake",
        "minutesafterwakeup": "minutes_after_wakeup",
        "timeinbed": "time_in_bed",
        "minutestofallasleep": "minutes_to_fall_asleep"
    })
)

# ---------------------------------------------------------
# sleep_score.csv
# ---------------------------------------------------------

sleep_score = sleep_score[[
    "id",
    "day_in_study",
    "overall_score",
    "composition_score",
    "revitalization_score",
    "duration_score",
    "deep_sleep_in_minutes",
    "resting_heart_rate",
    "restlessness"
]]

sleep_score = (
    sleep_score
    .groupby(["id", "day_in_study"], as_index=False)
    .agg({
        "overall_score": "mean",
        "composition_score": "mean",
        "revitalization_score": "mean",
        "duration_score": "mean",
        "deep_sleep_in_minutes": "mean",
        "resting_heart_rate": "mean",
        "restlessness": "mean"
    })
)

# ---------------------------------------------------------
# hormones_and_selfreport.csv
# (PARTIAL FEATURES ONLY)
# ---------------------------------------------------------

hormones = hormones[[
    "id",
    "day_in_study",
    "sleepissue",
    "fatigue",
    "stress", 
    "phase"
]]

# =========================================================
# MERGE EVERYTHING
# =========================================================

merged = sleep_data.copy()

dfs_to_merge = [
    sleep_score,
    hormones
]

for df in dfs_to_merge:

    merged["day_in_study"] = pd.to_numeric(
        merged["day_in_study"],
        errors="coerce"
    )

    df["day_in_study"] = pd.to_numeric(
        df["day_in_study"],
        errors="coerce"
    )

    merged = merged.merge(
        df,
        on=["id", "day_in_study"],
        how="left"
    )

# =========================================================
# OPTIONAL CLEANING
# =========================================================

# Fill missing numeric values with 0 if desired
# numeric_cols = merged.select_dtypes(include="number").columns
# merged[numeric_cols] = merged[numeric_cols].fillna(0)

# =========================================================
# SAVE
# =========================================================

merged.to_csv(OUTPUT_FILE, index=False)

print("Merged dataset shape:", merged.shape)
print("Saved to:", OUTPUT_FILE)

print("\nColumns:")
print(merged.columns.tolist())

print("\nPreview:")
print(merged.head())

# =========================================================
# IMPORTANT:
# The following files were intentionally NOT merged here
# because temporal/dynamic variation matters:
#
# - heart_rate_variability_details.csv
# - computed_temperature.csv
# - wrist_temperature.csv
# - respiratory_rate_summary.csv
# - estimated_oxygen_variation.csv
#
# These should later be engineered separately using:
# - std
# - min/max
# - rolling baselines
# - delta features
# - nighttime variability
# =========================================================