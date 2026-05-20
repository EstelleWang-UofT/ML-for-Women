import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("mcphases")   # change to your folder path
OUTPUT_FILE = "mcphases/merged/background.csv"

# =========================================================
# LOAD CSV FILES
# =========================================================

demographic = pd.read_csv(DATA_DIR / "demographic_vo2_max.csv")
height_weight = pd.read_csv(DATA_DIR / "height_and_weight.csv")
subject_info = pd.read_csv(DATA_DIR / "subject-info.csv")

# =========================================================
# KEEP ONLY RELEVANT COLUMNS
# (based on README feature descriptions)
# =========================================================

# -------------------------
# demographic_vo2_max.csv
# -------------------------
demographic = demographic[[
    "id",
    "day_in_study",
    "demographic_vo2_max",
    "demographic_vo2_max_error",
    "filtered_demographic_vo2_max", 
    "filtered_demographic_vo2_max_error"
]]

# -------------------------
# height_and_weight.csv
# -------------------------
height_weight = height_weight[[
    "id",
    "height_2022",
    "weight_2022",
    "height_2024",
    "weight_2024"
]]

# -------------------------
# subject-info.csv
# -------------------------
subject_info = subject_info[[
    "id",
    "birth_year",
    "ethnicity",
    "education", 
    "sexually_active",
    "self_report_menstrual_health_literacy",
    "age_of_first_menarche"
]]

# =========================================================
# MERGE ALL DATAFRAMES
# =========================================================

merged = (
    subject_info
    .merge(height_weight, on="id", how="left")
    .merge(demographic, on="id", how="left")
)

# # Fill numeric columns with 0
# numeric_cols = merged.select_dtypes(include="number").columns
# merged[numeric_cols] = merged[numeric_cols].fillna(0)

# =========================================================
# SAVE OUTPUT
# =========================================================

merged.to_csv(OUTPUT_FILE, index=False)
print("Saved", OUTPUT_FILE)
