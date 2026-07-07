import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")
OUTPUT_FILE = DATA_DIR / "merged" / "stress_recovery.csv"

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
# HEART RATE
# =========================================================

heart = heart[["id", "day_in_study", "bpm"]]

heart = (
    heart
    .groupby(["id", "day_in_study"], as_index=False)
    .agg(mean_bpm=("bpm", "mean"))
)

# =========================================================
# HRV
# =========================================================

hrv = hrv[["id", "day_in_study", "rmssd", "coverage", "low_frequency", "high_frequency"]]

hrv = (
    hrv
    .groupby(["id", "day_in_study"], as_index=False)
    .agg(
        mean_rmssd=("rmssd", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_low_frequency=("low_frequency", "mean"),
        mean_high_frequency=("high_frequency", "mean")
    )
)

# =========================================================
# RESTING HEART RATE
# =========================================================

if "value" in resting.columns:
    resting = resting.rename(columns={"value": "resting_hr"})

resting = resting[["id", "day_in_study", "resting_hr"]]

# =========================================================
# STRESS SCORE
# =========================================================

stress_score = stress_score[stress_score["calculation_failed"] == False]

stress_score = stress_score[
    [
        "id",
        "day_in_study",
        "stress_score",
        "sleep_points",
        "responsiveness_points",
        "exertion_points"
    ]
]

stress_score = stress_score.groupby(
    ["id", "day_in_study"],
    as_index=False
).agg(
    stress_score=("stress_score", "mean"), 
    stress_sleep_points=("sleep_points", "mean"),
    stress_responsiveness_points=("responsiveness_points", "mean"),
    stress_exertion_points=("exertion_points", "mean")
)

# =========================================================
# RESPIRATORY RATE
# =========================================================

rr = rr[
    ["id", "day_in_study", "full_sleep_breathing_rate", "deep_sleep_breathing_rate", 
     "light_sleep_breathing_rate", "rem_sleep_breathing_rate"]
]

rr = (
    rr
    .groupby(["id", "day_in_study"], as_index=False)
    .agg(
        sleep_breathing_rate=("full_sleep_breathing_rate","mean"),
        deep_sleep_breathing_rate=("deep_sleep_breathing_rate","mean"),
        light_sleep_breathing_rate=("light_sleep_breathing_rate","mean"),
        rem_sleep_breathing_rate=("rem_sleep_breathing_rate","mean")
        )
    )

# =========================================================
# SLEEP SCORE
# =========================================================

sleep_score = sleep_score[
    ["id", "day_in_study", "overall_score", "revitalization_score", "composition_score", "duration_score"]]

sleep_score = (
    sleep_score
    .groupby(["id", "day_in_study"], as_index=False)
    .agg(sleep_overall_score=("overall_score", "mean"), 
         sleep_revitalization_score=("revitalization_score", "mean"), 
         sleep_composition_score=("composition_score", "mean"), 
         sleep_duration_score=("duration_score", "mean"))
)

# =========================================================
# HORMONES + SELF REPORT
# =========================================================

hormones = hormones[
    [
        "id",
        "day_in_study",
        "phase",
        "stress",
        "sleepissue",
        "fatigue"
    ]
]


# =========================================================
# MERGE
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

    merged = merged.merge(
        df,
        on=["id", "day_in_study"],
        how="left"
    )


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

merged.to_csv(OUTPUT_FILE, index=False)

print("Merged dataset shape:", merged.shape)
print("Saved to:", OUTPUT_FILE)