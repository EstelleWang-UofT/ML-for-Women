import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")   # change to your folder path
OUTPUT_FILE = "mcphases/merged/physical_activity.csv"

# =========================================================
# LOAD CSV FILES
# =========================================================

active_minutes = pd.read_csv(DATA_DIR / "active_minutes.csv")
active_zone_minutes = pd.read_csv(DATA_DIR / "active_zone_minutes.csv")
calories = pd.read_csv(DATA_DIR / "calories.csv")
distance = pd.read_csv(DATA_DIR / "distance.csv")
exercise = pd.read_csv(DATA_DIR / "exercise.csv")
steps = pd.read_csv(DATA_DIR / "steps.csv")
hormones = pd.read_csv(DATA_DIR / "hormones_and_selfreport.csv")

# =========================================================
# KEEP ONLY RELEVANT COLUMNS
# (based on README feature descriptions)
# =========================================================

# -------------------------
# active_minutes.csv
# -------------------------
active_minutes = active_minutes[[
    "id",
    "day_in_study",
    "study_interval",
    "is_weekend",
    "sedentary",
    "lightly",
    "moderately",
    "very"
]]

# -------------------------
# active_zone_minutes.csv
# -------------------------
active_zone_minutes = active_zone_minutes[[
    "id",
    "day_in_study",
    "heart_zone_id",
    "total_minutes"
]]

# Convert long -> wide
active_zone_minutes = (
    active_zone_minutes
    .pivot_table(
        index=["id", "day_in_study"],
        columns="heart_zone_id",
        values="total_minutes",
        aggfunc="sum"
    )
    .reset_index()
)

# Rename columns
active_zone_minutes.columns.name = None
active_zone_minutes = active_zone_minutes.rename(columns={
    "fat burn": "fat_burn_minutes",
    "cardio": "cardio_minutes",
    "peak": "peak_minutes"
})

# -------------------------
# calories.csv
# -------------------------
calories = calories[[
    "id",
    "day_in_study",
    "calories"
]]

# Aggregate to daily total
calories = (
    calories
    .groupby(["id", "day_in_study"], as_index=False)
    .agg({
        "calories": "sum"
    })
    .rename(columns={
        "calories": "daily_calories"
    })
)

# -------------------------
# distance.csv
# -------------------------
distance = distance[[
    "id",
    "day_in_study",
    "distance"
]]

distance = (
    distance
    .groupby(["id", "day_in_study"], as_index=False)
    .agg({
        "distance": "sum"
    })
    .rename(columns={
        "distance": "daily_distance"
    })
)

# -------------------------
# exercise.csv
# -------------------------
exercise = exercise[[
    "id",
    "start_day_in_study",
    "activityname",
    "averageheartrate",
    "calories",
    "activeduration",
    "steps",
    "activezoneminutes",
    "elevationgain"
]]

exercise = exercise.rename(columns={
    "start_day_in_study": "day_in_study"
})

exercise["weighted_hr"] = (
    exercise["averageheartrate"] *
    exercise["activeduration"]
)

# Daily aggregation
exercise = (
    exercise
    .groupby(["id", "day_in_study"], as_index=False)
    .agg({
        "activeduration": "sum",
        "weighted_hr": "sum",
        "calories": "sum",
        "steps": "sum",
        "activezoneminutes": "sum",
        "elevationgain": "sum",
        "activityname": "count"
    })
)

exercise["avg_exercise_hr"] = (
    exercise["weighted_hr"] /
    exercise["activeduration"]
)

exercise = exercise.drop(columns=["weighted_hr"])

exercise = exercise.rename(columns={
    "calories": "exercise_calories",
    "steps": "exercise_steps",
    "activezoneminutes": "exercise_zone_minutes",
    "elevationgain": "exercise_elevation_gain",
    "activeduration": "exercise_duration"
})
# -------------------------
# steps.csv
# -------------------------
steps = steps[[
    "id",
    "day_in_study",
    "steps"
]]

steps = (
    steps
    .groupby(["id", "day_in_study"], as_index=False)
    .agg({
        "steps": "sum"
    })
    .rename(columns={
        "steps": "daily_steps"
    })
)

# -------------------------
# hormones_and_selfreport.csv
# (PARTIAL FEATURES ONLY)
# -------------------------
hormones = hormones[[
    "id",
    "day_in_study",
    "exerciselevel",
    "appetite"
]]

# =========================================================
# MERGE EVERYTHING
# =========================================================

merged = active_minutes.copy()

dfs_to_merge = [
    active_zone_minutes,
    calories,
    distance,
    exercise,
    steps,
    hormones
]

for df in dfs_to_merge:
    # coerce day_in_study to numeric on both sides to avoid dtype mismatches
    if 'day_in_study' in merged.columns and 'day_in_study' in df.columns:
        merged['day_in_study'] = pd.to_numeric(merged['day_in_study'], errors='coerce')
        df['day_in_study'] = pd.to_numeric(df['day_in_study'], errors='coerce')
    merged = merged.merge(
        df,
        on=["id", "day_in_study"],
        how="left"
    )

# =========================================================
# OPTIONAL CLEANING
# =========================================================

# Fill missing numeric values with 0
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