# ML for Women — mcPHASES fatigue modeling

## Setup

1. Download the [mcPHASES dataset](https://www.physionet.org/content/mcphases/1.0.0/) locally into `mcphases/` (raw CSVs are too large for GitHub).
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data pipeline (notebooks)

Notebooks live under `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `notebooks/physical activity/preprocessing and merging.ipynb` | Merge raw mcPHASES physical-activity tables |
| `notebooks/physical activity/merged preprocess.ipynb` | Clean, transform, export processed CSV |
| `notebooks/physical activity/fatigue_modeling.ipynb` | Train/tune ordinal fatigue models |

**Processed modeling CSV:** `mcphases/merged/physical_activity_merged_processed.csv`

## Modeling code

Implementation package: [`src/modeling/`](src/modeling/)

- Load/split: `modeling.data.load_fatigue_data`, `prepare_splits`
- Models: 7 ordinal models with optional history feature ablation (see `docs/models/`)
- Tuning: Optuna via `modeling.runner.tune_and_benchmark_model`

Run the modeling notebook with the project root on `PYTHONPATH` or use the `%pip` / `sys.path` setup in the notebook (adds `../../src`).

## Documentation

Model and pipeline details: [`docs/models/README.txt`](docs/models/README.txt)
