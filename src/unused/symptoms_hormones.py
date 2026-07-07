import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")
OUTPUT_FILE = "mcphases/merged/symptoms_hormones.csv"

# =========================================================
# LOAD CSV FILES
# =========================================================

hormones = pd.read_csv(DATA_DIR / "hormones_and_selfreport.csv")
computed_temp = pd.read_csv(DATA_DIR / "computed_temperature.csv")

# =========================================================
# HORMONES + SELF REPORT
# =========================================================

h_cols = [
    "id",
    "day_in_study",
    "phase",

    # Hormones
    "lh",
    "estrogen",
    "pdg",

    # Menstrual flow
    "flow_volume",
    "flow_color",

    # Self-report symptoms
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating"
]

hormones = hormones[
    [c for c in h_cols if c in hormones.columns]
]

# =========================================================
# COMPUTED TEMPERATURE
# =========================================================

if "sleep_start_day_in_study" in computed_temp.columns:
    computed_temp = computed_temp.rename(columns={
        "sleep_start_day_in_study": "day_in_study"
    })

temp_cols = [
    c for c in [
        "id",
        "day_in_study",
        "nightly_temperature",
        "baseline_relative_sample_sum"
    ]
    if c in computed_temp.columns
]

computed_temp = computed_temp[temp_cols]

# ---------------------------------------------------------
# Temperature aggregation
# Average nightly temperature per day
# ---------------------------------------------------------

if not computed_temp.empty:

    temp_agg = {}

    if "nightly_temperature" in computed_temp.columns:
        temp_agg["nightly_temperature"] = "mean"

    if "baseline_relative_sample_sum" in computed_temp.columns:
        temp_agg["baseline_relative_sample_sum"] = "mean"

    computed_temp = (
        computed_temp
        .groupby(["id", "day_in_study"], as_index=False)
        .agg(temp_agg)
    )

# =========================================================
# MERGE
# =========================================================

if not computed_temp.empty:

    hormones["day_in_study"] = pd.to_numeric(
        hormones["day_in_study"],
        errors="coerce"
    )

    computed_temp["day_in_study"] = pd.to_numeric(
        computed_temp["day_in_study"],
        errors="coerce"
    )

    merged = hormones.merge(
        computed_temp,
        on=["id", "day_in_study"],
        how="left"
    )

else:
    merged = hormones.copy()

# =========================================================
# OPTIONAL CLEANING
# =========================================================

# Fill ONLY numeric columns if desired
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