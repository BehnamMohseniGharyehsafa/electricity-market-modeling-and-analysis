"""One frozen chronological holdout experiment with a separate calibration set."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data import TARGET
from .evaluation import (
    block_bootstrap_delta,
    conformal_radius,
    interval_metrics,
    point_metrics,
)
from .features import build_samples, tabular_features
from .neural import ARCHITECTURES, train_neural

SPLITS = {
    "train": ("2022-01-01", "2024-01-01"),
    "validation": ("2024-01-01", "2024-07-01"),
    "calibration": ("2024-07-01", "2025-01-01"),
    "test": ("2025-01-01", "2026-01-01"),
}


def pooled_features(samples):
    """Small hourly feature set for pooled quantile regression (known before auction)."""
    h = samples.history[:, :, :-1].reshape(len(samples.dates), 7, 24, 11)
    last, week = h[:, -1], h[:, 0]
    hour = np.tile(np.arange(24), len(h))
    return np.column_stack(
        [
            last.reshape(-1, 11),
            week[:, :, TARGET].reshape(-1),
            h[:, :, :, TARGET].mean(1).reshape(-1),
            h[:, :, :, TARGET].std(1).reshape(-1),
            np.repeat(last[:, :, TARGET].mean(1), 24),
            np.repeat(last[:, :, TARGET].max(1), 24),
            np.repeat(samples.calendar, 24, axis=0),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ]
    ).astype(np.float32)


def run(root: Path, epochs=100, seed=42):
    started = time.monotonic()
    samples = build_samples(root / "data/processed/nyiso_daily.npz")
    masks = {name: samples.mask(*dates) for name, dates in SPLITS.items()}
    if any(mask.sum() < 30 for mask in masks.values()):
        raise ValueError("All four chronological splits require at least 30 eligible days")
    tr, va, ca, te = [masks[k] for k in ["train", "validation", "calibration", "test"]]
    x, y = tabular_features(samples), samples.target
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report = root / "reports"
    report.mkdir(exist_ok=True)
    predictions = {
        "DailyNaive": samples.history[:, -24:, TARGET],
        "WeeklyNaive": samples.history[:, :24, TARGET],
    }
    metadata = {}
    # Tiny, pre-specified tuning grids; validation only. No tuning against 2025.
    families = {
        "Ridge": [make_pipeline(StandardScaler(), Ridge(alpha=a)) for a in [10, 100, 1000]],
        "ExtraTrees": [
            ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=seed,
                n_jobs=4,
            )
            for leaf in [2, 5]
        ],
        "XGBoost": [
            XGBRegressor(
                n_estimators=220,
                max_depth=depth,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=10,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=seed,
                n_jobs=4,
            )
            for depth in [2, 4]
        ],
    }
    for name, candidates in families.items():
        best, fitted, scores = float("inf"), None, []
        for model in candidates:
            model.fit(x[tr], y[tr])
            score = float(np.abs(model.predict(x[va]) - y[va]).mean())
            scores.append(score)
            if score < best:
                best, fitted = score, model
        predictions[name] = fitted.predict(x)
        joblib.dump(fitted, artifacts / f"{name}.joblib")
        metadata[name] = {
            "candidate_validation_mae": scores,
            "selected_params": {k: str(v) for k, v in fitted.get_params().items()},
        }
        print(f"{name}: validation MAE {best:.3f}", flush=True)
    for name in ARCHITECTURES:
        pred, info, checkpoint = train_neural(name, samples, tr, va, seed, epochs)
        predictions[name] = pred
        metadata[name] = info
        torch.save(checkpoint, artifacts / f"{name}.pt")
        print(f"{name}: selected epoch {info['best_epoch']}", flush=True)

    # The ensemble has a fixed rule: top 3 learned models by validation MAE,
    # inverse-validation-MAE weights. It never sees calibration or test labels.
    learned = [n for n in predictions if not n.endswith("Naive")]
    selected = sorted(learned, key=lambda n: np.abs(predictions[n][va] - y[va]).mean())[:3]
    weights = np.array([1 / np.abs(predictions[n][va] - y[va]).mean() for n in selected])
    weights /= weights.sum()
    predictions["ValidationEnsemble"] = sum(w * predictions[n] for w, n in zip(weights, selected))
    metadata["ValidationEnsemble"] = {"weights": dict(zip(selected, weights.tolist()))}

    # Quantile boosting offers heteroscedastic intervals; quantiles are rearranged
    # before CQR to prevent crossing. Calibration only widens (never shrinks).
    pooled = pooled_features(samples)
    q_predictions = []
    for q in [0.1, 0.5, 0.9]:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=180,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=10,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(pooled[np.repeat(tr, 24)], y[tr].ravel())
        q_predictions.append(model.predict(pooled).reshape(-1, 24))
        joblib.dump(model, artifacts / f"quantile_{q}.joblib")
    q_predictions = np.sort(np.stack(q_predictions), axis=0)
    predictions["QuantileBoosting"] = q_predictions[1]
    cqr_score = np.maximum(q_predictions[0][ca] - y[ca], y[ca] - q_predictions[2][ca])
    cqr_radius = np.maximum(conformal_radius(cqr_score), 0)
    intervals = {"QuantileBoosting": (q_predictions[0] - cqr_radius, q_predictions[2] + cqr_radius)}
    for name, pred in predictions.items():
        if name not in intervals:
            radius = conformal_radius(np.abs(y[ca] - pred[ca]))
            intervals[name] = (pred - radius, pred + radius)

    calibration = {
        name: {
            "method": "cqr" if name == "QuantileBoosting" else "symmetric",
            "alpha": 0.2,
            "radius": (
                cqr_radius
                if name == "QuantileBoosting"
                else conformal_radius(np.abs(y[ca] - pred[ca]))
            ).tolist(),
        }
        for name, pred in predictions.items()
    }
    (artifacts / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    threshold = float(np.quantile(y[tr], 0.95))
    weekly_loss = np.abs(predictions["WeeklyNaive"][te] - y[te]).mean(1)
    rows = []
    for name, pred in predictions.items():
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid predictions from {name}")
        lower, upper = intervals[name]
        row = {
            "model": name,
            **point_metrics(y[te], pred[te], threshold),
            **interval_metrics(y[te], lower[te], upper[te]),
            **block_bootstrap_delta(np.abs(pred[te] - y[te]).mean(1), weekly_loss, seed),
            "validation_mae": float(np.abs(pred[va] - y[va]).mean()),
        }
        row["skill_vs_weekly_pct"] = 100 * (1 - row["mae"] / weekly_loss.mean())
        rows.append(row)
    scores = pd.DataFrame(rows).sort_values("mae")
    scores.to_csv(report / "metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame({n: np.abs(p[te] - y[te]).mean(0) for n, p in predictions.items()}).to_csv(
        report / "hourly_mae.csv", index_label="hour", float_format="%.8f"
    )
    daily_losses = pd.DataFrame(
        {
            "date": samples.dates[te].strftime("%Y-%m-%d"),
            **{n: np.abs(p[te] - y[te]).mean(1) for n, p in predictions.items()},
        }
    )
    daily_losses.to_csv(report / "daily_mae.csv", index=False, float_format="%.8f")
    validation_winner = min(predictions, key=lambda n: np.abs(predictions[n][va] - y[va]).mean())
    run_info = {
        "protocol": "Frozen fit, rolling observed history, no test-period refitting",
        "seed": seed,
        "epochs_cap": epochs,
        "splits": SPLITS,
        "split_counts": {n: int(m.sum()) for n, m in masks.items()},
        "train_price_95pct_threshold": threshold,
        "validation_selected_model": validation_winner,
        "test_hours": int(te.sum() * 24),
        "models": metadata,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "scipy"]
        },
    }
    (report / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    np.savez_compressed(
        artifacts / "backtest.npz",
        actual=y,
        dates=samples.dates.to_numpy(),
        test_mask=te,
        calibration_mask=ca,
        **predictions,
        lower=intervals[validation_winner][0],
        upper=intervals[validation_winner][1],
    )
    print(scores[["model", "mae", "rmse", "coverage"]].to_string(index=False), flush=True)
    return scores
