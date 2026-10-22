"""Render README and results from the computed reports, preventing invented metrics."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
info = json.loads((ROOT / "reports/run.json").read_text())
quality = json.loads((ROOT / "reports/data_quality.json").read_text())
scores = pd.read_csv(ROOT / "reports/metrics.csv")
selected = info["validation_selected_model"]
selected_row = scores.set_index("model").loc[selected]
table = [
    "| Model | MAE ↓ | RMSE ↓ | Skill vs weekly ↑ | 80% interval coverage |",
    "|---|---:|---:|---:|---:|",
]
for row in scores.itertuples():
    label = f"**{row.model}**" if row.model == selected else row.model
    table.append(
        f"| {label} | {row.mae:.2f} | {row.rmse:.2f} | "
        f"{row.skill_vs_weekly_pct:.1f}% | {100 * row.coverage:.1f}% |"
    )
table = "\n".join(table)
foundation = ROOT / "reports/foundation_experiment.json"
foundation_line = "The optional Chronos-2 experiment is not part of this table."
if foundation.exists():
    f = json.loads(foundation.read_text())
    period = (
        "all eligible 2025 delivery days"
        if f["experiment"] == "full_test"
        else "pre-specified monthly 15th days in 2025"
    )
    foundation_line = (
        f"Chronos-2 also ran on **{f['days']} days** ({period}): "
        f"MAE **{f['metrics']['mae']:.2f} USD/MWh**, native interval coverage "
        f"**{100 * f['metrics']['coverage']:.1f}%**. "
        "Its public pretraining overlap is unknown, so it is **reported separately** "
        "and excluded from supervised model selection. "
        "[Exact record and pinned checkpoint](reports/foundation_experiment.json)."
    )

readme = f"""<p align="center"><img src="docs/assets/hero.png" alt="Electricity Market Lab: real US prices, modern models, auditable forecasting experiments" width="100%"></p>

# Electricity market modeling and analysis

[![Validate](https://github.com/BehnamMohseniGharyehsafa/electricity-market-modeling-and-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/BehnamMohseniGharyehsafa/electricity-market-modeling-and-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11–3.13-3776AB)
![License](https://img.shields.io/badge/code-MIT-00a88f)
![Data](https://img.shields.io/badge/data-NYISO%20official-10283f)

**Predict a full day of wholesale electricity prices before the auction.** A reproducible
US electricity-market research project combining classical machine learning, modern neural
forecasting, probabilistic calibration, and a separate foundation-model experiment.

Built and maintained by **Behnam Mohseni Gharyehsafa**.

> This repository reports computed results, including weak models and failed interval
> coverage. It does not claim universal state of the art, perfect accuracy, or trading profit.

## At a glance

| Real market data | Forecast | Validation |
|---|---|---|
| NYISO, 11 internal zones, 2022–2025 | Tomorrow's 24 NYC hourly LBMPs, USD/MWh | Train / validation / calibration / held-out test |
| {quality["internal_zone_rows"]:,} source observations; 48 monthly ZIP archives | 7 daily price curves + calendar; before 05:00 auction close | {info["split_counts"]["test"]} test days; {info["test_hours"]:,} hourly targets |

**What makes the project useful:** official downloadable data with checksums; eleven tested
baseline/ML/deep-learning/ensemble methods; current Chronos-2 integration; transparent
auction timing; DST-aware data handling; calibrated intervals; and figures generated from
the saved experiment rather than fabricated examples.

<img src="docs/assets/pipeline.png" alt="Data, historical inputs, model families, forecast outputs, and four chronological partitions" width="100%">

## Real US data — links included

The source is the **New York Independent System Operator's P-2A day-ahead zonal LBMP**:

- [Official archive index](https://mis.nyiso.com/public/P-2Alist.htm)
- [Current NYISO portal](https://oasis.nyiso.com/DAM_LBMP_ZONE)
- [Download an example: January 2024 ZIP](https://mis.nyiso.com/public/csv/damlbmp/20240101damlbmp_zone_csv.zip)
- [All 48 exact URLs, byte sizes and SHA-256 fingerprints](reports/data_manifest.json)
- [Data card: units, zones, timing, DST and limitations](docs/DATA_CARD.md)

The downloader fetches real prices directly from NYISO without an API key. Source data
stays local; this repository includes code, provenance, aggregate scores and research figures.
The MIT license applies to original project code/documentation, not NYISO source data or
third-party pretrained weights. No synthetic data is substituted in the published experiment.

<img src="docs/assets/market_landscape.png" alt="Real NYISO price history, monthly hourly price heatmap, cross-zone correlations, and NYC–WEST spread" width="100%">

## Models, inputs and outputs

| Family | Methods | Inputs → outputs |
|---|---|---|
| Seasonal references | DailyNaive, WeeklyNaive | Previous day/week NYC curve → 24 prices |
| Regularized ML | Ridge | Lagged curves, spatial summaries, calendar → 24 prices |
| Tree ensembles | ExtraTrees, XGBoost | Lagged curves, spatial summaries, calendar → 24 prices |
| Recurrent / decomposition | LSTM, DLinear | Multizone sequence / NYC decomposition → 24 prices |
| Attention-based DL | PatchTST, iTransformer | Temporal patches / cross-zone tokens → 24 prices |
| Probabilistic ML | QuantileBoosting + CQR | Hourly lag/spatial/calendar features → median and interval |
| Ensemble | ValidationEnsemble | Three models chosen using validation only → weighted curve |
| Foundation model, separate track | Chronos-2 (120M pretrained parameters) | NYC + past spatial covariates → native quantile forecasts |

Every supervised point model also receives separate per-hour split-conformal intervals.
The neural implementations are compact adaptations, with architectural differences and
training settings documented in the [methodology](docs/METHODOLOGY.md).

<img src="docs/assets/neural_architectures.png" alt="DLinear decomposition, LSTM sequence encoder, PatchTST temporal patches, and iTransformer spatial tokens" width="100%">

## Measured results

The full-year test is **2025**. Model weights are fitted on 2022–2023; settings and neural
epochs use 2024 H1; intervals use 2024 H2. No refitting on test labels is performed.
All eight DST transition days across the source period are excluded as targets; historical
DST curves are normalized and flagged. Negative prices are preserved.

{table}

MAE/RMSE are USD/MWh. Bold identifies the **validation-selected** model, not a model
selected after viewing test scores. {selected} achieves a test MAE of
**{selected_row.mae:.2f} USD/MWh**, versus WeeklyNaive, with a paired seven-day-block
bootstrap MAE difference of **{selected_row.delta_mae_vs_weekly:.2f}**
(95% descriptive interval **[{selected_row.ci95_low:.2f}, {selected_row.ci95_high:.2f}]**).
The nominal 80% interval achieved **{selected_row.coverage * 100:.1f}%** hourly coverage.
Time dependence and regime shifts limit conformal coverage guarantees.

{foundation_line}

<img src="docs/assets/foundation_examples.png" alt="Separate Chronos-2 forecasts with native quantile intervals on three pre-specified 2025 dates; pretraining overlap is unknown" width="100%">

<img src="docs/assets/leaderboard.png" alt="All measured 2025 model mean absolute errors, with the validation-selected model highlighted" width="100%">

<img src="docs/assets/forecast_examples.png" alt="Actual prices, the validation-selected model, and calibrated intervals for three pre-specified 2025 dates" width="100%">

<img src="docs/assets/risk_diagnostics.png" alt="Forecast error by delivery hour and measured prediction-interval coverage" width="100%">

Explore [all metrics](reports/metrics.csv), [daily paired losses](reports/daily_mae.csv),
[hourly errors](reports/hourly_mae.csv), [run settings and versions](reports/run.json),
and the [research interpretation](docs/RESULTS.md).

Verification: `pytest -q` covers data contracts, future-input isolation, scoring, neural
gradients and inference inputs. `python scripts/verify_artifacts.py` reloads the fitted
models and compares their forecasts and calibrated bounds to the saved backtest.

## Reproduce locally

Python 3.11–3.13. The recorded run uses Python {info["python"]} on CPU.

```bash
git clone https://github.com/BehnamMohseniGharyehsafa/electricity-market-modeling-and-analysis.git
cd electricity-market-modeling-and-analysis
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
python -m pip install -e '.[dev]'
electricity-lab all --epochs 100 --seed 42
pytest -q
ruff check .
```

Alternatively run stages individually:

```bash
electricity-lab download
electricity-lab benchmark --epochs 100 --seed 42
electricity-lab figures
python scripts/write_readme.py
```

Reload the saved validation-selected model and write an actual 24-row forecast CSV:

```bash
electricity-lab forecast --date 2025-01-15
# Also supports 2026-01-01 using only the final seven published 2025 curves.
```

Forecast files are saved locally under `artifacts/`. Inference requires no target-day
prices; serialization round-trip checks are recorded in [the artifact audit](reports/artifact_audit.json).
The forecast intervals retain the measured calibration limitations described above.
For the guided notebook, install `python -m pip install -e '.[notebook]'` and open
[`notebooks/01_research_walkthrough.ipynb`](notebooks/01_research_walkthrough.ipynb).

Archives total only a few MB; training requires scientific Python/PyTorch packages and
several minutes on a modern CPU. The supplied [environment lock](requirements-lock.txt)
captures the actual run, including optional foundation packages. Editable project metadata
is excluded from that lock: install `-e .` separately after installing it. Exact numerical
results can vary across hardware/library versions. ZIP corrections can change future reruns;
compare the recorded source hashes.

For the advanced foundation-model track:

```bash
python -m pip install -e '.[foundation]'
python -m electricity_market.foundation --full-test --revision 29ec3766d36d6f73f0696f85560a422f50e8498c
# Omit --revision to resolve and record the current official weight revision.
# Without --full-test, run only the 12 illustrative monthly 15th-day examples.
python scripts/write_readme.py
```

The first foundation run downloads official model weights (hundreds of MB). Pretraining
overlap is unknown, so those results remain separate from the supervised leaderboard.

## Repository guide

```text
src/electricity_market/   Download, features, model training, scoring, charts, CLI
tests/                   Data contracts, leakage tests, scores, neural backward passes
docs/                    Data card, methodology, interpretation, model card and figures
reports/                 Provenance, aggregate metrics and exact run metadata
notebooks/               Guided research walkthrough (no invented output cells)
scripts/                 Documentation generation
data/                    Local-only raw archives and processed arrays
artifacts/               Local-only fitted models and row-level backtest arrays
.github/workflows/       Python 3.11/3.12 CI checks
```

## Limits and next research steps

NYISO archives contain later price corrections; this is not a historical release-vintage
simulation. The benchmark covers one US market and one target zone, has two training
years, one full test year and one random seed. It does not model realized demand, market
clearing, bid curves, dispatch, or profits. Non-24-hour targets are excluded. Weather/load/gas
forecast vintages, multi-zone targets, adaptive calibration, rolling retraining, multiple seeds
and a newer untouched holdout are natural extensions. See the
[model card](docs/MODEL_CARD.md) and [contributing guide](CONTRIBUTING.md).

## Research references

[DLinear](https://arxiv.org/abs/2205.13504) ·
[PatchTST](https://arxiv.org/abs/2211.14730) ·
[iTransformer](https://arxiv.org/abs/2310.06625) ·
[Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222) ·
[Chronos-2](https://arxiv.org/abs/2510.15821)

Original code and documentation: MIT. See [LICENSE](LICENSE) and [citation metadata](CITATION.cff).
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")
results = f"""# Results and interpretation

The validation-selected model is **{selected}**. All numbers below were computed from
official NYISO data with the recorded seed, splits and package versions.

{table}

The selected model's MAE is {selected_row.mae:.4f} USD/MWh; its RMSE is
{selected_row.rmse:.4f}. RMSE weights rare, large misses more strongly. Its mean signed
error is {selected_row.bias:.4f} USD/MWh (positive means overprediction).
The training-derived high-price threshold is {info["train_price_95pct_threshold"]:.2f}
USD/MWh; its MAE on test hours above that threshold is {selected_row.spike_mae:.2f}.
Exact daily peak-hour accuracy is {selected_row.peak_hour_accuracy * 100:.1f}%.

For the selected model, calibrated intervals cover {selected_row.coverage * 100:.1f}% of
test hourly outcomes against an 80% nominal level, with mean width
{selected_row.mean_width:.2f} USD/MWh. This is empirical coverage, not an operational
reliability promise. The frozen calibration period can differ substantially from 2025.

The paired daily MAE difference versus WeeklyNaive is {selected_row.delta_mae_vs_weekly:.2f}
USD/MWh; its seven-day moving-block bootstrap 95% interval is
[{selected_row.ci95_low:.2f}, {selected_row.ci95_high:.2f}]. A negative difference favors
the model. This interval does not measure training-seed variability, protect against
all model comparisons, or establish performance in another year/market.

## Read the comparison carefully

The top row is the best observed test score. The bold row is the model chosen using
validation only. Transformer labels do not imply that those models must win. With only
two training years and a price-history-only information set, simpler approaches can be
competitive. Poor interval coverage and large spike errors are retained in the report.
The experiment supports a comparison for this stated task, not a broad claim of superiority.

{foundation_line.replace("reports/", "../reports/")}

## Audit and visual evidence

- [Machine-readable metrics](../reports/metrics.csv)
- [Run configuration, neural epoch traces and package versions](../reports/run.json)
- [Hourly errors](../reports/hourly_mae.csv), [daily paired losses](../reports/daily_mae.csv)
- [Download fingerprints](../reports/data_manifest.json), [quality report](../reports/data_quality.json)
- [Market landscape](assets/market_landscape.png), [full leaderboard](assets/leaderboard.png)
- [Pre-specified example curves](assets/forecast_examples.png), [risk diagnostics](assets/risk_diagnostics.png)

All charts are generated by `electricity-lab figures` from the local processed data and
saved backtest arrays. The illustrative forecast dates were specified in code, rather than
chosen by minimum error. Raw values, checkpoints and source ZIPs are local-only.
"""
(ROOT / "docs/RESULTS.md").write_text(results, encoding="utf-8")
print("Updated README.md and docs/RESULTS.md from measured results")
