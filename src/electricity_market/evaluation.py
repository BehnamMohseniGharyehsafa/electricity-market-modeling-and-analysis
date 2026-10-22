"""Price-safe scores and dependence-aware uncertainty on daily losses."""

import numpy as np


def point_metrics(actual, predicted, threshold):
    error = predicted - actual
    high = actual >= threshold
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "spike_mae": float(np.abs(error)[high].mean()) if high.any() else None,
        "peak_hour_accuracy": float((actual.argmax(1) == predicted.argmax(1)).mean()),
    }


def conformal_radius(scores, alpha=0.2):
    """Finite-sample corrected, per-hour quantile; accepts absolute or CQR scores."""
    n = len(scores)
    if n < 2 or not 0 < alpha < 1:
        raise ValueError("Need >=2 calibration days and 0 < alpha < 1")
    rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores, axis=0)[rank - 1]


def interval_metrics(actual, lower, upper, alpha=0.2):
    if np.any(lower > upper):
        raise ValueError("Crossed prediction intervals")
    width = upper - lower
    score = width + (2 / alpha) * (lower - actual) * (actual < lower)
    score += (2 / alpha) * (actual - upper) * (actual > upper)
    return {
        "coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "mean_width": float(width.mean()),
        "interval_score": float(score.mean()),
        "nominal_coverage": 1 - alpha,
    }


def block_bootstrap_delta(loss, reference, seed=42, block=7, repetitions=2000):
    """Circular moving-block bootstrap of paired DAILY mean absolute errors.

    CI is descriptive for this held-out period; block length is pre-specified at 7 days.
    Non-24-hour days are absent, so blocks use adjacent eligible market dates.
    """
    delta = np.asarray(loss) - np.asarray(reference)
    n = len(delta)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block))))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    boot = delta[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "delta_mae_vs_weekly": float(delta.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "block_days": block,
        "repetitions": repetitions,
    }
