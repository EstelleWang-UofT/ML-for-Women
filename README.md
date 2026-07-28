# ML for Women — mcPHASES fatigue modeling

## Setup

1. Download the [mcPHASES dataset](https://www.physionet.org/content/mcphases/1.0.0/) locally into `mcphases/` (raw CSVs are too large for GitHub).
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data pipeline (notebooks)

Notebooks live under `notebooks/physical activity/`:

| Notebook | Purpose |
|----------|---------|
| `0 target_identification.ipynb` | Target variable selection |
| `1 preprocessing and merging.ipynb` | Merge raw mcPHASES physical-activity tables |
| `2 merged preprocess.ipynb` | Clean, transform, export processed CSV |
| `history feature engineering.ipynb` | Tune history construction; forward-select `HISTORY_FEATURES` |
| `3 fatigue_modeling.ipynb` | Train/tune ordinal models (base + 3 history features) |
| `3.5 history features.ipynb` | Archived 3-col vs 7-col history comparison and significance |

**Processed modeling CSV:** `mcphases/merged/physical_activity_merged_processed.csv`

## Modeling code

Implementation package: [`src/modeling/`](src/modeling/)

- Load/split: `modeling.data.load_fatigue_data`, `prepare_splits`
- Models: 7 ordinal models with optional history feature ablation (see `docs/models/`)
- Tuning: Optuna via `modeling.runner.tune_and_benchmark_model`

Run the modeling notebook with the project root on `PYTHONPATH` or use the `%pip` / `sys.path` setup in the notebook (adds `../../src`).

## Documentation

Model and pipeline details: [`docs/models/README.txt`](docs/models/README.txt)
