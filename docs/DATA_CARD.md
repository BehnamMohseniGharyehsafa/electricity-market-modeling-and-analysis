# NYISO day-ahead zonal price data

## Source and access

- **Publisher:** New York Independent System Operator (NYISO).
- **Product:** P-2A, Day-Ahead Market LBMP — Zonal.
- **Official index:** <https://mis.nyiso.com/public/P-2Alist.htm>.
- **Current portal:** <https://oasis.nyiso.com/DAM_LBMP_ZONE>.
- **Example monthly archive:** <https://mis.nyiso.com/public/csv/damlbmp/20240101damlbmp_zone_csv.zip>.
- **Template:** `https://mis.nyiso.com/public/csv/damlbmp/YYYYMM01damlbmp_zone_csv.zip`.
- **Coverage used:** 1 January 2022 through 31 December 2025, inclusive.
- **Access:** public ZIP/CSV downloads, no API key. Raw files remain local and are not committed.
- **Audit:** [archive URLs and SHA-256 hashes](../reports/data_manifest.json),
  [observed quality report](../reports/data_quality.json).

The MIT license covers this project's original code and documentation. It does not assign
ownership or an open-data license to NYISO source data. Download source data from NYISO and
consult its applicable terms for your intended use. Aggregate research metrics and rendered
figures are included; bulk source files and row-level target/prediction arrays are excluded.

## Schema and economic meaning

Each daily file contains local delivery timestamps, zone names, PTIDs, LBMP, marginal losses,
and marginal congestion components. The modeled value is **LBMP in USD/MWh**, an hourly
wholesale energy price. It is not household retail electricity cost, demand in MW, or energy
consumption in MWh. Negative prices are valid observations and are retained.

The model uses the 11 internal zones: CAPITL, CENTRL, DUNWOD, GENESE, HUD VL, LONGIL,
MHK VL, MILLWD, N.Y.C., NORTH, and WEST. External interface locations H Q, NPX, O H,
and PJM are excluded. The target is N.Y.C. (PTID 61761). Other internal zones supply
historical spatial information. Price correlations and spreads describe associations, not
causal transmission constraints or a full congestion decomposition.

## Time convention and availability

Timestamps represent hour-beginning delivery intervals in `America/New_York`, with DST.
The experiment predicts delivery day **D** at **04:45 local time on D−1**, before the
05:00 day-ahead market close shown in NYISO's
[market timeline](https://www.nyiso.com/documents/20142/25467833/Bids-and-Offers-Evaluation-Process.pdf/7b893955-3df4-bddf-078e-c6c50a940cee).
The 24 prices for D−1 were already cleared and published on D−2. They can therefore be
used even though some physical D−1 delivery hours are still in the future at forecast time.
No D price, realized D load, or realized D weather is an input.

**Important research limitation:** NYISO's archived files include later corrections and
updates. This repository downloads the current archive, not the exact vintage visible at
each historical auction. It is an availability-conscious retrospective benchmark, not a
certified point-in-time trading simulation. An operational study needs timestamped release
vintages and publication-outage handling.

## DST and validation

Normal delivery days have 24 hours; spring and autumn transition days have 23 and 25.
For a fixed 24-output model:

1. Exclude all eight DST transition days **as targets**. No fabricated target is scored.
2. Historical spring curves interpolate the missing hour between adjacent hours in an
   already-published day; historical autumn curves average the repeated hour.
3. Add a historical DST-normalization flag. DLinear and univariate PatchTST use NYC prices
   only; iTransformer uses the 11 price channels; tabular/LSTM models can use the flag.
4. Verify each zone's exact local clock-hour multiplicity against the timezone calendar,
   continuous day coverage, finite prices, and complete internal-zone coverage.

The normalized context is seven calendar days × 24 clock-hour slots, not always 168 elapsed
physical hours. A production system should support variable-length 23/24/25-hour targets.

## Deliberate scope

This first benchmark isolates price-history and calendar information. It does not incorporate
weather forecasts, gas forward prices, outages, renewable forecasts, bid curves, or forecast
load. Adding realized versions of those variables would create an unrealistic information set.
Future extensions should use genuinely archived, issue-time-stamped forecasts.
