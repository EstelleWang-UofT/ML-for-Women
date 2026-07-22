"""Move GEE model cells from 3 fatigue_modeling.ipynb to unused models.ipynb."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "notebooks/physical activity/3 fatigue_modeling.ipynb"
UNUSED = ROOT / "notebooks/physical activity/unused models.ipynb"

GEE_MARKERS = ("#### `gee_gaussian`", "#### `gee_ordinal`", "_name = 'gee_gaussian'", "_name = 'gee_ordinal'")


def is_gee_cell(cell: dict) -> bool:
    src = "".join(cell.get("source", []))
    return any(marker in src for marker in GEE_MARKERS)


def clear_code_outputs(cell: dict) -> dict:
    cell = deepcopy(cell)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def remove_gee_import(source_lines: list[str]) -> list[str]:
    return [line for line in source_lines if "GEE_OPTUNA_TRIALS" not in line]


def build_unused_notebook(gee_cells: list[dict], metadata: dict) -> dict:
    intro = """# Unused models

These **Generalized Estimating Equation (GEE)** benchmarks live outside [`3 fatigue_modeling.ipynb`](3%20fatigue_modeling.ipynb).

## Why unused?

The main notebook targets **prediction accuracy** (held-out test MAE). GEE is designed for **longitudinal inference**: coefficient estimates with **statistically valid standard errors and p-values after accounting for within-cluster correlation** (repeated daily rows within each participant-interval). That is valuable for interpretability and hypothesis testing, not for ranking predictors by test error.

On this dataset both GEE models land mid-pack (~1.27 test MAE) versus stronger tree and history-feature models (~0.9–1.2 MAE). They stay here for reference and optional re-runs, not the primary accuracy comparison.

Same participant-level split, GroupKFold CV, and Optuna tuning (`GEE_OPTUNA_TRIALS=10`) as the main notebook.
"""

    gee_gaussian_md = """### `gee_gaussian` (ordinal regression)

**What it does:** Fits a **Gaussian GEE** on `fatigue_num`, treating the ordinal score as continuous. Uses **autoregressive working correlation** within each cluster (participant × `study_interval`, rows sorted by `day_in_study`). Includes `study_interval` as a covariate plus the 17 base daily features. Optuna tunes `maxiter`; the predicted mean is clipped to [0, 5].

**Why unused:** Inference-oriented (valid SEs/p-values under correlation). Not prioritized while we optimize prediction accuracy only.
"""

    gee_ordinal_md = """### `gee_ordinal` (ordinal classification)

**What it does:** Fits **five cumulative-threshold Binomial GEE** models (P(y > k) for k = 0…4) with autoregressive working correlation (Exchangeable fallback when AR is unstable). Clustered by participant-interval; uses the 17 base daily features only (wave captured by clustering, not a `study_interval` covariate). Class probabilities from threshold survival are combined and clipped to [0, 5].

**Why unused:** Same inference vs accuracy tradeoff as `gee_gaussian`; kept for reference, not the main leaderboard.
"""

    gee_by_kind: dict[str, dict] = {}
    for cell in gee_cells:
        src = "".join(cell["source"])
        if "gee_gaussian" in src:
            gee_by_kind["gaussian"] = clear_code_outputs(cell)
        elif "gee_ordinal" in src:
            gee_by_kind["ordinal"] = clear_code_outputs(cell)

    cells: list[dict] = [
        {"cell_type": "markdown", "metadata": {}, "source": [intro]},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["%pip install -q -r ../../requirements.txt\n"],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "_src = Path('../../src').resolve()\n",
                "if str(_src) not in sys.path:\n",
                "    sys.path.insert(0, str(_src))\n",
                "\n",
                "for _mod in [k for k in list(sys.modules) if k == 'modeling' or k.startswith('modeling.')]:\n",
                "    del sys.modules[_mod]\n",
                "\n",
                "from modeling.config import DATA_PATH, GEE_OPTUNA_TRIALS, N_CV_FOLDS\n",
                "from modeling.data import load_fatigue_data, prepare_splits, split_summary_table\n",
                "from modeling.registry import ORDINAL_MODELS\n",
                "from modeling.runner import tune_and_benchmark_model\n",
                "from modeling.summaries import collect_summaries\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Load data and split\n"]},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "df = load_fatigue_data('../../' + DATA_PATH)\n",
                "bundle = prepare_splits(df)\n",
                "\n",
                "print(f\"Rows: {len(df):,}  Participants: {df['id'].nunique()}\")\n",
                "display(split_summary_table(bundle))\n",
                "print('Test participant ids:', sorted(bundle.test_ids))\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["ordinal_results = []\n", "ordinal_best_params = {}\n"],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## GEE models\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": [gee_gaussian_md]},
        gee_by_kind["gaussian"],
        {"cell_type": "markdown", "metadata": {}, "source": [gee_ordinal_md]},
        gee_by_kind["ordinal"],
        {"cell_type": "markdown", "metadata": {}, "source": ["## Summary\n"]},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "_, test_summary = collect_summaries(ordinal_results)\n",
                "display(test_summary.sort_values('test_mae'))\n",
            ],
        },
    ]
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    nb = json.loads(MAIN.read_text(encoding="utf-8"))
    gee_cells: list[dict] = []
    main_cells: list[dict] = []
    pointer = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "GEE models (`gee_gaussian`, `gee_ordinal`) are in "
            "[`unused models.ipynb`](unused%20models.ipynb) — kept for longitudinal "
            "inference benchmarks, excluded from the main prediction comparison.\n"
        ],
    }
    inserted_pointer = False

    for cell in nb["cells"]:
        if is_gee_cell(cell):
            gee_cells.append(cell)
            continue
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "from modeling.config import" in src and "GEE_OPTUNA_TRIALS" in src:
                lines = cell["source"]
                cell = deepcopy(cell)
                cell["source"] = remove_gee_import(lines)
        main_cells.append(cell)
        if (
            not inserted_pointer
            and cell.get("cell_type") == "code"
            and "_name = 'catboost_regressor'" in "".join(cell.get("source", []))
            and "history" not in "".join(cell.get("source", []))
        ):
            main_cells.append(pointer)
            inserted_pointer = True

    if len(gee_cells) != 4:
        raise RuntimeError(f"Expected 4 GEE cells, found {len(gee_cells)}")

    for cell in main_cells:
        if cell.get("cell_type") == "code" and "collect_categorized_summaries" in "".join(
            cell.get("source", [])
        ):
            cell["outputs"] = []
            cell["execution_count"] = None

    nb["cells"] = main_cells
    MAIN.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    unused_nb = build_unused_notebook(gee_cells, nb["metadata"])
    UNUSED.write_text(json.dumps(unused_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {UNUSED.relative_to(ROOT)}")
    print(f"Updated {MAIN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
