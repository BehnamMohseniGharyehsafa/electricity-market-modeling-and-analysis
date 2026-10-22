"""Create a portable, output-free research notebook using the project's real pipeline."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell(
        "markdown",
        """# Electricity Market Lab — research walkthrough

**Real NYISO day-ahead prices • 11 zones • 2022–2025**

The notebook reads the same audited artifacts as the command line. Run the download and
benchmark stages first. No results are invented or embedded as unexecuted output.

See [the methodology](../docs/METHODOLOGY.md) for auction timing, information access,
DST policy, calibration and the separation of the foundation-model experiment.
""",
    ),
    cell(
        "code",
        """import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_market.data import TARGET
from electricity_market.features import build_samples

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Open the notebook from the repository"
""",
    ),
    cell(
        "markdown",
        """## 1. Inspect provenance and data contracts

Official source: <https://mis.nyiso.com/public/P-2Alist.htm>.
The recorded manifest includes the URL and SHA-256 of every source archive.
""",
    ),
    cell(
        "code",
        """manifest = json.loads((ROOT / "reports/data_manifest.json").read_text())
display(pd.DataFrame(manifest["archives"]).head())
quality = json.loads((ROOT / "reports/data_quality.json").read_text())
quality
""",
    ),
    cell(
        "markdown",
        """## 2. Inspect the inputs and target

Each sample uses seven published daily curves, a DST history flag and six calendar features.
The 24 target prices are strictly outside the historical input window. D−1's day-ahead
prices have already been published at the D−1 pre-auction forecast origin.
""",
    ),
    cell(
        "code",
        """samples = build_samples(ROOT / "data/processed/nyiso_daily.npz")
print("History:", samples.history.shape)
print("Calendar:", samples.calendar.shape)
print("Target:", samples.target.shape)
i = samples.dates.get_loc("2025-01-15")
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].plot(samples.history[i, :, TARGET])
axes[0].set(title="Known NYC historical price slots", xlabel="Context slot", ylabel="USD/MWh")
axes[1].plot(samples.target[i])
axes[1].set(title="Held-out target: 2025-01-15", xlabel="Delivery hour", ylabel="USD/MWh")
plt.tight_layout()
""",
    ),
    cell(
        "markdown",
        """## 3. Compare every method

Rank by held-out MAE, but identify the model chosen using validation only.
Inspect weak methods and interval undercoverage rather than hiding them.
""",
    ),
    cell(
        "code",
        """metrics = pd.read_csv(ROOT / "reports/metrics.csv")
info = json.loads((ROOT / "reports/run.json").read_text())
print("Validation-selected model:", info["validation_selected_model"])
display(metrics[["model", "mae", "rmse", "bias", "coverage", "spike_mae"]])
""",
    ),
    cell(
        "markdown",
        """## 4. Read paired model comparisons

Daily loss differences preserve the pairing of forecasts for the same delivery day.
The supplied seven-day-block bootstrap approximates temporal dependence; it does not
capture all sources of uncertainty or correct multiple model comparisons.
""",
    ),
    cell(
        "code",
        """from electricity_market.evaluation import block_bootstrap_delta
losses = pd.read_csv(ROOT / "reports/daily_mae.csv")
chosen = info["validation_selected_model"]
block_bootstrap_delta(losses[chosen], losses["WeeklyNaive"])
""",
    ),
    cell(
        "markdown",
        """## 5. Inspect genuine forecast curves

These examples are specified in advance in the plotting code. Their missed peaks and
interval violations are useful diagnostics.
""",
    ),
    cell(
        "code",
        """from IPython.display import Image, display
display(Image(filename=str(ROOT / "docs/assets/forecast_examples.png")))
display(Image(filename=str(ROOT / "docs/assets/risk_diagnostics.png")))
""",
    ),
    cell(
        "markdown",
        """## 6. Foundation model: separate evidence

Chronos-2 uses official pretrained weights with past zone covariates. Its pretraining
overlap with public NYISO data is unknown. This is not evidence of contamination-free
zero-shot superiority. Native quantile intervals are not conformally calibrated here.
""",
    ),
    cell(
        "code",
        """foundation_path = ROOT / "reports/foundation_experiment.json"
if foundation_path.exists():
    display(json.loads(foundation_path.read_text()))
else:
    print("Optional: python -m electricity_market.foundation --full-test")
""",
    ),
    cell(
        "markdown",
        """## 7. Reproduce or extend

From the repository root:

```bash
electricity-lab all --epochs 100 --seed 42
python -m electricity_market.foundation --full-test
python scripts/write_readme.py
pytest -q
ruff check .
```

New research should reserve a fresh future holdout. Use genuine forecast vintages when
adding load or weather; never replace them with future realized values.
""",
    ),
]
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, item in enumerate(cells):
    item["id"] = f"research-{i:02d}"
destination = root / "notebooks/01_research_walkthrough.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
