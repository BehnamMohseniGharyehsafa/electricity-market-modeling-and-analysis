"""Separate Chronos-2 research experiment with explicit pretraining-contamination caveat.

Official API: https://github.com/amazon-science/chronos-forecasting
This experiment never participates in the supervised leaderboard or model selection.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import TARGET, ZONES
from .evaluation import interval_metrics, point_metrics
from .features import build_samples


def run(root: Path, full_test=False, revision=None):
    from chronos import Chronos2Pipeline
    from huggingface_hub import HfApi

    torch.set_num_threads(4)
    torch.manual_seed(42)
    samples = build_samples(root / "data/processed/nyiso_daily.npz")
    test = samples.mask("2025-01-01", "2026-01-01")
    mask = test if full_test else (test & (samples.dates.day == 15))
    indices = np.flatnonzero(mask)
    started = time.monotonic()
    requested_revision = revision
    revision = revision or HfApi().model_info("amazon/chronos-2").sha
    kwargs = {"device_map": "cpu", "revision": revision}
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", **kwargs)
    predictions, lows, highs = [], [], []
    for i in indices:
        # An integer-step synthetic hourly index keeps the canonical 24-hour local
        # curve contract; these stamps are model positions, not UTC event times.
        stamp = pd.date_range("2000-01-01", periods=168, freq="h")
        history = samples.history[i, :, :-1]
        context = pd.DataFrame({"id": "NYC", "timestamp": stamp, "target": history[:, TARGET]})
        for j, zone in enumerate(ZONES):
            if j != TARGET:
                context[f"past_{zone}"] = history[:, j]
        prediction = pipeline.predict_df(
            context,
            prediction_length=24,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        predictions.append(prediction["0.5"].to_numpy())
        lows.append(prediction["0.1"].to_numpy())
        highs.append(prediction["0.9"].to_numpy())
        print(f"Chronos-2: {samples.dates[i].date()}", flush=True)
    pred, lo, hi = np.array(predictions), np.array(lows), np.array(highs)
    actual = samples.target[mask]
    if pred.shape != actual.shape or not np.isfinite(pred).all():
        raise ValueError("Unexpected Chronos output shape or non-finite forecasts")
    threshold = float(np.quantile(samples.target[samples.mask("2022-01-01", "2024-01-01")], 0.95))
    model_config = pipeline.model.config.to_dict()
    result = {
        "model_id": "amazon/chronos-2",
        "requested_revision": requested_revision,
        "resolved_revision": revision,
        "experiment": "full_test" if full_test else "pre-specified monthly 15th-day examples",
        "days": len(indices),
        "dates": samples.dates[mask].strftime("%Y-%m-%d").tolist(),
        "pretraining_overlap": "UNKNOWN: public NYISO history may appear in pretraining.",
        "headline_leaderboard_eligible": False,
        "intervals": "Native 10th/90th quantiles; not conformally calibrated",
        "metrics": {**point_metrics(actual, pred, threshold), **interval_metrics(actual, lo, hi)},
        "elapsed_seconds": time.monotonic() - started,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["chronos-forecasting", "transformers", "torch"]
        },
    }
    (root / "reports/foundation_experiment.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "artifacts/chronos_config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    np.savez_compressed(
        root / "artifacts/foundation.npz",
        dates=samples.dates[mask].to_numpy(),
        actual=actual,
        predictions=pred,
        lower=lo,
        upper=hi,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument(
        "--revision", help="Hugging Face model commit hash for exact weight pinning"
    )
    args = parser.parse_args()
    run(args.root, args.full_test, args.revision)
