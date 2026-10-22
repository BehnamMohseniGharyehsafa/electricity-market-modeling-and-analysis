import json
from pathlib import Path

import joblib
import numpy as np

from electricity_market.benchmark import pooled_features
from electricity_market.evaluation import conformal_radius, point_metrics
from electricity_market.features import build_samples
from electricity_market.inference import date_inputs, forecast, predict_model

root = Path(__file__).resolve().parents[1]
info = json.loads((root / "reports/run.json").read_text())
samples = build_samples(root / "data/processed/nyiso_daily.npz")
with np.load(root / "artifacts/backtest.npz", allow_pickle=False) as data:
    saved = {k: data[k] for k in data.files}
ca = saved["calibration_mask"]
names = [
    n
    for n in saved
    if n in list(info["models"]) + ["DailyNaive", "WeeklyNaive", "QuantileBoosting"]
]
calibration = {
    n: {
        "method": "symmetric",
        "alpha": 0.2,
        "radius": conformal_radius(np.abs(saved["actual"][ca] - saved[n][ca])).tolist(),
    }
    for n in names
}
pooled = pooled_features(samples)
qp = np.sort(
    np.array(
        [
            joblib.load(root / "artifacts" / f"quantile_{q}.joblib")
            .predict(pooled[np.repeat(ca, 24)])
            .reshape(-1, 24)
            for q in [0.1, 0.5, 0.9]
        ]
    ),
    axis=0,
)
radius = np.maximum(
    conformal_radius(np.maximum(qp[0] - saved["actual"][ca], saved["actual"][ca] - qp[2])), 0
)
calibration["QuantileBoosting"] = {"method": "cqr", "alpha": 0.2, "radius": radius.tolist()}
(root / "artifacts/calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
checks = []
for day in ["2025-01-15", "2025-07-15", "2025-12-15"]:
    inputs = date_inputs(root / "data/processed/nyiso_daily.npz", day)
    i = samples.dates.get_loc(day)
    for name in names:
        pred = predict_model(name, inputs, root, info)
        error = float(np.max(np.abs(pred[0] - saved[name][i])))
        assert error < 0.001, (day, name, error)
        checks.append({"date": day, "model": name, "max_reload_difference": error})
    frame = forecast(root, day)
    assert np.max(np.abs(frame.lower_usd_mwh - saved["lower"][i])) < 0.001
    assert np.max(np.abs(frame.upper_usd_mwh - saved["upper"][i])) < 0.001
test = saved["test_mask"]
metrics = point_metrics(
    saved["actual"][test],
    saved[info["validation_selected_model"]][test],
    info["train_price_95pct_threshold"],
)
result = {
    "status": "passed",
    "reloaded_model_date_checks": len(checks),
    "maximum_allowed_difference_usd_mwh": 0.001,
    "checks": checks,
    "selected_model_recomputed_metrics": metrics,
    "point_and_interval_inference_roundtrip": "passed on three specified dates",
}
(root / "reports/artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print("Artifact audit passed:", len(checks), "reloaded model/date comparisons")
