import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")
OUTPUT_FILE = "mcphases/merged/stress_recovery.csv"

# =========================================================
# LOAD CSV FILES
# =========================================================

heart = pd.read_csv(DATA_DIR / "heart_rate.csv")
hrv = pd.read_csv(DATA_DIR / "heart_rate_variability_details.csv")
resting = pd.read_csv(DATA_DIR / "resting_heart_rate.csv")
stress_score = pd.read_csv(DATA_DIR / "stress_score.csv")
rr = pd.read_csv(DATA_DIR / "respiratory_rate_summary.csv")
sleep_score = pd.read_csv(DATA_DIR / "sleep_score.csv")
hormones = pd.read_csv(DATA_DIR / "hormones_and_selfreport.csv")

# =========================================================
# KEEP ONLY RELEVANT COLUMNS
# =========================================================

# -------------------------
# heart_rate.csv
# -------------------------
heart = heart[[c for c in ["id", "day_in_study", "bpm"] if c in heart.columns]]

if "bpm" in heart.columns:
    if "day_in_study" in heart.columns:
        heart = (
            heart
            .groupby(["id", "day_in_study"], as_index=False)
            .agg({"bpm": "mean"})
            .rename(columns={"bpm": "mean_bpm"})
        )
    else:
        heart = (
            heart
            .groupby("id", as_index=False)
            .agg({"bpm": "mean"})
            .rename(columns={"bpm": "mean_bpm"})
        )

# -------------------------
# heart_rate_variability_details.csv
# -------------------------
if "timestamp" in hrv.columns and "day_in_study" not in hrv.columns:
    hrv = hrv.rename(columns={"timestamp": "day_in_study"})

hrv = hrv[[c for c in ["id", "day_in_study", "rmssd"] if c in hrv.columns]]
if hrv.columns.duplicated().any():
    hrv = hrv.loc[:, ~hrv.columns.duplicated()]

if {"id", "day_in_study", "rmssd"}.issubset(hrv.columns):
    hrv = (
        hrv
        .groupby(["id", "day_in_study"], as_index=False)
        .agg({"rmssd": "mean"})
        .rename(columns={"rmssd": "mean_rmssd"})
    )

# -------------------------
# resting_heart_rate.csv
# -------------------------
if "value" in resting.columns:
    resting = resting.rename(columns={"value": "resting_hr"})
resting = resting[[c for c in ["id", "resting_hr"] if c in resting.columns]]

# -------------------------
# stress_score.csv
# -------------------------
if "timestamp" in stress_score.columns and "day_in_study" not in stress_score.columns:
    stress_score = stress_score.rename(columns={"timestamp": "day_in_study"})
stress_score = stress_score[[c for c in ["id", "day_in_study", "overall_score"] if c in stress_score.columns]]
if stress_score.columns.duplicated().any():
    stress_score = stress_score.loc[:, ~stress_score.columns.duplicated()]
stress_score = stress_score.rename(columns={"overall_score": "stress_overall_score"})

# -------------------------
# respiratory_rate_summary.csv
# -------------------------
if "timestamp" in rr.columns and "day_in_study" not in rr.columns:
    rr = rr.rename(columns={"timestamp": "day_in_study"})
rr = rr[[c for c in ["id", "day_in_study", "full_sleep_breathing_rate"] if c in rr.columns]]
if rr.columns.duplicated().any():
    rr = rr.loc[:, ~rr.columns.duplicated()]
rr = rr.rename(columns={"full_sleep_breathing_rate": "sleep_breathing_rate"})

# -------------------------
# sleep_score.csv
# -------------------------
if "timestamp" in sleep_score.columns and "day_in_study" not in sleep_score.columns:
    sleep_score = sleep_score.rename(columns={"timestamp": "day_in_study"})
sleep_score = sleep_score[[c for c in ["id", "day_in_study", "overall_score"] if c in sleep_score.columns]]
if sleep_score.columns.duplicated().any():
    sleep_score = sleep_score.loc[:, ~sleep_score.columns.duplicated()]
sleep_score = sleep_score.rename(columns={"overall_score": "sleep_overall_score"})

# -------------------------
# hormones_and_selfreport.csv
# -------------------------
hormones = hormones[[c for c in [
    "id",
    "day_in_study",
    "stresslevel",
    "recovery",
    "mood",
    "fatigue"
] if c in hormones.columns]]

# =========================================================
# MERGE EVERYTHING
# =========================================================

merged = hormones.copy()

dfs_to_merge = [
    heart,
    hrv,
    resting,
    stress_score,
    rr,
    sleep_score
]

for df in dfs_to_merge:
    if df.empty:
        continue

    if "day_in_study" in merged.columns and "day_in_study" in df.columns:
        merged["day_in_study"] = pd.to_numeric(merged["day_in_study"], errors="coerce")
        df["day_in_study"] = pd.to_numeric(df["day_in_study"], errors="coerce")
        merged = merged.merge(df, on=["id", "day_in_study"], how="left")
    elif "id" in df.columns:
        merged = merged.merge(df, on=["id"], how="left")

# =========================================================
# OPTIONAL CLEANING
# =========================================================

numeric_cols = merged.select_dtypes(include="number").columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)

# =========================================================
# SAVE
# =========================================================

merged.to_csv(OUTPUT_FILE, index=False)

print("Merged dataset shape:", merged.shape)
print("Saved to:", OUTPUT_FILE)
