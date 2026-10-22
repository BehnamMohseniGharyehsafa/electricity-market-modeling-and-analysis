"""Reload trusted local fitted models and forecast a date from prior published curves."""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar

from .benchmark import pooled_features
from .data import TARGET
from .features import Samples, tabular_features
from .neural import ARCHITECTURES


def date_inputs(path, delivery_date):
    day = pd.Timestamp(delivery_date).normalize()
    slots = pd.date_range(
        day, day + pd.Timedelta(days=1), freq="h", inclusive="left", tz="America/New_York"
    )
    if len(slots) != 24:
        raise ValueError("This fixed-output model does not forecast DST transition target days")
    with np.load(path, allow_pickle=False) as data:
        dates = pd.DatetimeIndex(data["dates"])
        prior = pd.date_range(day - pd.Timedelta(days=7), periods=7, freq="D")
        positions = dates.get_indexer(prior)
        if (positions < 0).any():
            raise ValueError("Seven complete prior price curves are required")
        curves = data["curves"][positions]
        normal = data["normal"][positions]
    history = np.column_stack([curves.reshape(168, 11), np.repeat(~normal, 24)])
    holiday = len(USFederalHolidayCalendar().holidays(day, day)) > 0
    calendar = [
        np.sin(2 * np.pi * day.dayofweek / 7),
        np.cos(2 * np.pi * day.dayofweek / 7),
        np.sin(2 * np.pi * (day.dayofyear - 1) / 365.25),
        np.cos(2 * np.pi * (day.dayofyear - 1) / 365.25),
        float(day.dayofweek >= 5),
        float(holiday),
    ]
    return Samples(
        pd.DatetimeIndex([day]),
        history[None].astype(np.float32),
        np.array([calendar], dtype=np.float32),
        np.zeros((1, 24), dtype=np.float32),
    )


def predict_model(name, samples, root, info):
    artifacts = root / "artifacts"
    if name == "ValidationEnsemble":
        return sum(
            float(weight) * predict_model(component, samples, root, info)
            for component, weight in info["models"][name]["weights"].items()
        )
    if name in ["DailyNaive", "WeeklyNaive"]:
        if name == "DailyNaive":
            return samples.history[:, -24:, TARGET]
        return samples.history[:, :24, TARGET]
    if name in ARCHITECTURES:
        checkpoint = torch.load(artifacts / f"{name}.pt", weights_only=True, map_location="cpu")
        net = ARCHITECTURES[name](context=checkpoint["context"])
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        h = samples.history.copy()
        h[:, :, :-1] = (h[:, :, :-1] - checkpoint["center"]) / checkpoint["scale"]
        with torch.no_grad():
            pred = net(torch.tensor(h), torch.tensor(samples.calendar)).numpy()
        return pred * checkpoint["scale"] + checkpoint["center"]
    if name == "QuantileBoosting":
        values = [
            joblib.load(artifacts / f"quantile_{q}.joblib").predict(pooled_features(samples))
            for q in [0.1, 0.5, 0.9]
        ]
        return np.sort(np.array(values), axis=0)[1].reshape(-1, 24)
    return joblib.load(artifacts / f"{name}.joblib").predict(tabular_features(samples))


def forecast(root, delivery_date):
    if not delivery_date:
        raise ValueError("Supply --date YYYY-MM-DD")
    info = json.loads((root / "reports/run.json").read_text())
    samples = date_inputs(root / "data/processed/nyiso_daily.npz", delivery_date)
    name = info["validation_selected_model"]
    prediction = predict_model(name, samples, root, info)[0]
    day = samples.dates[0]
    calibration = json.loads((root / "artifacts/calibration.json").read_text())[name]
    radius = np.array(calibration["radius"])
    if calibration["method"] == "cqr":
        q = np.sort(
            np.array(
                [
                    joblib.load(root / "artifacts" / f"quantile_{q}.joblib").predict(
                        pooled_features(samples)
                    )
                    for q in [0.1, 0.5, 0.9]
                ]
            ),
            axis=0,
        )
        lower, upper = q[0] - radius, q[2] + radius
    else:
        lower, upper = prediction - radius, prediction + radius
    origin = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=4, minutes=45)).tz_localize(
        "America/New_York"
    )
    frame = pd.DataFrame(
        {
            "delivery_date": str(day.date()),
            "local_hour": range(24),
            "forecast_origin": origin.isoformat(),
            "model": name,
            "point_usd_mwh": prediction,
            "lower_usd_mwh": lower,
            "upper_usd_mwh": upper,
        }
    )
    path = root / "artifacts" / f"forecast_{day:%Y%m%d}.csv"
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print(f"Saved {path}")
    return frame
