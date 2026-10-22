# Forecasting protocol and model specification

## The experiment

For delivery date D, estimate the entire NYC day-ahead curve from the previous seven
published daily price curves and a deterministic calendar vector. All models receive the
same dates and target values. Feature access differs intentionally by architecture.

| Partition | Delivery dates | Purpose |
|---|---|---|
| Train | 2022–2023 | Fit model weights, feature scaling, spike threshold |
| Validation | January–June 2024 | Select tree/ridge settings, neural epochs and ensemble weights |
| Calibration | July–December 2024 | Estimate interval correction, without updating point models |
| Test | January–December 2025 | Evaluate once, with frozen fitted parameters |

Seven initial context days and non-24-hour target days are excluded. Every test forecast
uses the newly observed historical price curves available for that origin. This is rolling
prediction with a **frozen fit**, not a repeatedly retrained walk-forward model. Calibration
and test labels never fit network weights or scalers. Code and exact settings are in
[`benchmark.py`](../src/electricity_market/benchmark.py) and [run metadata](../reports/run.json).

## Models actually evaluated

| Model | Representation | Fitting / selection |
|---|---|---|
| DailyNaive | Copy D−1 NYC curve | No fitting |
| WeeklyNaive | Copy D−7 NYC curve | No fitting |
| Ridge | NYC 168-hour history; 24×11 latest zone prices; zone means/std; calendar/DST | Standardize training only; alpha ∈ {10,100,1000} |
| ExtraTrees | Same direct, multi-output features | 250 trees; leaf minimum ∈ {2,5} |
| XGBoost | Same direct, multi-output features | 220 rounds; depth ∈ {2,4}; learning rate .035 |
| DLinear | NYC only, 25-hour moving-average decomposition | Linear seasonal/trend heads + calendar MLP |
| LSTM | 168×12 sequence | 48-unit recurrent encoder + 24-output head + calendar MLP |
| PatchTST | NYC only, instance-scaled 24-hour patches, stride 12 | Two 48-dimensional, four-head attention blocks; learned positions |
| iTransformer | 11 spatial tokens, each embeds 168 price values | Two 48-dimensional, four-head attention blocks; NYC token readout |
| ValidationEnsemble | Three best learned models before quantile boosting | Inverse validation-MAE weights, fixed before calibration/test |
| QuantileBoosting | Pooled hourly lag/spatial/calendar features | Histogram gradient boosting at .1/.5/.9; 180 iterations; 15 leaves |

The neural models are compact **adaptations of published architectures**, implemented here
for inspection. They are not official reference implementations, exact paper reproductions,
or claims of record-setting accuracy. PatchTST is a univariate adaptation; iTransformer
uses cross-zone attention. A calendar MLP is added to all neural variants. DLinear/LSTM
use training-global price scaling; PatchTST/iTransformer also normalize each context.
The latter instance scales have a .1 floor in globally normalized units.

Neural training: AdamW, learning rate .001, weight decay .01, batch size 64,
absolute-error loss, gradient clipping at 1, up to 100 epochs, and patience 15.
The best validation-MAE state is restored. Seed 42 and deterministic CPU operations are
used. Only one seed is run; across-seed variability is not quantified. Limited tuning and
only two training years can disadvantage high-capacity models. No architecture should be
declared universally superior from this comparison.

## Uncertainty

All point models receive separate per-delivery-hour symmetric split-conformal intervals:
`prediction ± q`, where q is the corrected empirical 80th percentile of absolute
calibration residuals. For n calibration days, use order statistic
`min(ceil((n+1)×0.8), n)` (one-based rank).

QuantileBoosting first sorts its .1/.5/.9 outputs to avoid quantile crossing.
For its interval, compute calibration scores `max(q10−y, y−q90)` and widen the
quantile bounds by a nonnegative corrected score quantile. The median is the point forecast.
This conservative CQR variation never shrinks native quantile intervals.

Nominal 80% coverage is **per hour**, not a simultaneous guarantee for a complete daily
curve. Electricity prices are serially dependent and undergo regime changes, so ordinary
exchangeability assumptions need not hold. We report empirical test coverage, mean width
and interval score; coverage below nominal is a meaningful failure, not hidden.

## Scores and statistical comparisons

- MAE, RMSE, signed bias, and high-price MAE, all in USD/MWh.
- High-price threshold: training target 95th percentile, frozen before test.
- Peak-hour accuracy: exact match of the highest predicted versus actual hour; ties use
  the first maximum. This is deliberately strict.
- Relative skill: `100 × (1 − model MAE / WeeklyNaive MAE)`.
- Interval score: width plus penalties `2/alpha` for values outside bounds (alpha .2).
- A paired circular moving-block bootstrap uses daily mean absolute losses, seven-day
  blocks, 2,000 repetitions and seed 42. The 95% interval concerns MAE difference versus
  WeeklyNaive, not the MAE itself. Adjacent eligible days omit DST targets. These are
  descriptive comparisons; no multiple-comparison correction is applied.

MAPE is deliberately absent because zero/negative wholesale prices make it unsuitable.
No trading profit, storage dispatch benefit or causal market effect is inferred from MAE.
The charts show the validation-selected model even when another model ranks first on test.
Illustrative daily forecasts always use January 15, July 15 and December 15, 2025.

## Foundation-model research track

[`foundation.py`](../src/electricity_market/foundation.py) uses the official Chronos-2
120M pretrained checkpoint with past zone covariates. The default experiment forecasts
the 15th of each month in 2025; `--full-test` evaluates every eligible test day.
Its model timestamps are synthetic hourly positions encoding canonical clock-hour slots.
Native q10/q90 intervals are evaluated without conformal calibration. This experiment is
reported separately because the public checkpoint's pretraining overlap with NYISO is
unknown. Zero-shot does not establish a contamination-free holdout. It is excluded from
the supervised leaderboard and ensemble. Record the resolved weight revision when supplied
by the model library and use `--revision HASH` for exact weight pinning.

Chronos-2 is a current foundation-model option, not a claim that it is the newest model in
every family. For example, TimesFM 3.0 is available as of October 2026; its self-hosted
weights carry distinct use restrictions. It is not part of this experiment.

## References

- [NYISO official price archives](https://mis.nyiso.com/public/P-2Alist.htm).
- [DLinear: Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504).
- [PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730).
- [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625).
- [XGBoost official documentation](https://xgboost.readthedocs.io/en/stable/).
- [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222).
- [Chronos-2 technical report](https://arxiv.org/abs/2510.15821) and
  [official implementation](https://github.com/amazon-science/chronos-forecasting).
- [TimesFM official implementation and license notice](https://github.com/google-research/timesfm).
