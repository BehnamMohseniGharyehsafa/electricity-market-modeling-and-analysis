# Model card

**System:** Electricity Market Lab, v1.0.0
**Author:** Behnam Mohseni Gharyehsafa
**Task:** Retrospective pre-auction, 24-output NYC day-ahead wholesale price forecasting.

## Intended use

Research, education, reproducible model comparison, exploratory market analysis, and
testing new price forecasting architectures. Inputs are historical published day-ahead
price curves and deterministic calendar information. Outputs are point forecasts in
USD/MWh plus separately calibrated hourly intervals. See [methodology](METHODOLOGY.md)
for exact information access, training, splits and architecture adaptations.

The primary model is selected by validation error. Its identity and measured performance
are generated into the [results report](RESULTS.md) from saved experiment output. This
avoids claiming that the most complicated model must be the best model.

## Evaluation evidence

- Official NYISO price archives, 2022–2025; NYC target, 11-zone historical context.
- Training: 719 eligible days in 2022–2023 after initial context and DST exclusions.
- Validation: 181 days; calibration: 183 days; test: 363 days / 8,712 hourly targets.
- Eleven supervised/reference/ensemble methods and a separate Chronos-2 research track.
- [MAE/RMSE/bias, spike error, peak-hour accuracy, interval scores](../reports/metrics.csv).
- Daily paired seven-day-block bootstrap comparisons against the weekly baseline.
- Data contracts, causality checks and neural forward/backward tests.

## Known limitations and failure modes

The 2025 regime differs from the 2024 calibration period. **All main-benchmark interval
methods fall below their nominal 80% test coverage**, so these intervals should not be
treated as dependable operational risk limits. Price spikes remain difficult to predict.

The archive includes revisions made after original auctions; exact historical vintages
are unavailable in this pipeline. There is one target zone, one seed, one test year and
limited hyperparameter tuning. All test forecasts use frozen fitted weights. NYC results
do not establish transfer to PJM, ERCOT, CAISO, other NYISO zones or retail tariffs.
The 23- and 25-hour target days are excluded, so year-round deployment requires further work.
Weather, load, gas, outages and bids are absent from the model information set.

The foundation model is pretrained externally. Its overlap with this public dataset is
unknown, and its results are excluded from supervised selection and ranking.
Third-party weights retain their own license. No checkpoint is bundled in the public repo.

## Operational boundary

This is a research artifact, without a live service, a market-clearing engine, a validated
trading policy or a storage-optimization policy. Before any operational use, obtain
historical release vintages, support DST target days, validate additional years/seeds/markets,
use archived exogenous forecasts, monitor drift, calibrate adaptively and define fallback
behavior for missing or delayed data. Transaction costs and economic constraints need a
separate decision-focused evaluation.

## Reproducibility and artifacts

The repo includes source URLs and SHA-256 fingerprints, actual aggregate scores, exact
package versions and neural epoch traces. The raw archives, fitted joblib/PyTorch files,
row-level predictions, and downloaded foundation weights remain local in ignored folders.
Only load model serialization from trusted sources. Re-run the documented commands to
reconstruct artifacts and figures.
