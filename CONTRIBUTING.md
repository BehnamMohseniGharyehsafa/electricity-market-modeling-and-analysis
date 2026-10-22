# Contributing

Install `python -m pip install -e '.[dev]'`, then run `pytest -q` and `ruff check .`.
Keep changes focused, document the information available at forecast time, and preserve
source provenance. A new model should use the same delivery dates and target values.

Do not use test labels for tuning, feature selection, ensemble weights or interval radii.
The published 2025 holdout has now been inspected. Substantial future research should add
a newer untouched holdout instead of repeatedly optimizing against this year's scores.
Separate exploratory experiments from confirmatory comparisons, and record seed, package
versions, data hashes and the exact model/checkpoint revision.

For new external variables, provide archived forecast issuance timestamps. Realized future
load, temperature, fuel prices or target-day prices are not pre-auction inputs. Keep negative
prices, audit missing observations and handle DST explicitly. Include meaningful tests for
data alignment, causal feature construction and scoring changes.

Regenerate figures with `electricity-lab figures` and documentation with
`python scripts/write_readme.py`. Do not hand-edit measured values or invent results.
Avoid committing raw NYISO data, credentials, model weights, local environments or
large artifacts. Use issues to describe reproducible bugs, with minimal inputs and versions.
