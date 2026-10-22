"""Construct pre-auction samples; no actual future load, weather or target prices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .data import TARGET


@dataclass
class Samples:
    dates: pd.DatetimeIndex
    history: np.ndarray
    calendar: np.ndarray
    target: np.ndarray

    def mask(self, start: str, end: str) -> np.ndarray:
        return np.asarray((self.dates >= start) & (self.dates < end))


def build_samples(path, lookback_days: int = 7) -> Samples:
    if lookback_days < 7:
        raise ValueError("At least seven history days are required for the weekly baseline")
    with np.load(path, allow_pickle=False) as data:
        curves, normal = data["curves"], data["normal"]
        dates = pd.DatetimeIndex(data["dates"])
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    histories, calendars, targets, selected = [], [], [], []
    for i in range(lookback_days, len(dates)):
        if not normal[i]:
            continue
        d = dates[i]
        hist = curves[i - lookback_days : i].reshape(lookback_days * 24, -1)
        dst_flag = np.repeat(~normal[i - lookback_days : i], 24).astype(np.float32)
        history = np.column_stack([hist, dst_flag])
        calendar = [
            np.sin(2 * np.pi * d.dayofweek / 7),
            np.cos(2 * np.pi * d.dayofweek / 7),
            np.sin(2 * np.pi * (d.dayofyear - 1) / 365.25),
            np.cos(2 * np.pi * (d.dayofyear - 1) / 365.25),
            float(d.dayofweek >= 5),
            float(d.normalize() in holidays),
        ]
        histories.append(history)
        calendars.append(calendar)
        targets.append(curves[i, :, TARGET])
        selected.append(d)
    return Samples(
        pd.DatetimeIndex(selected),
        np.array(histories, dtype=np.float32),
        np.array(calendars, dtype=np.float32),
        np.array(targets, dtype=np.float32),
    )


def tabular_features(samples: Samples) -> np.ndarray:
    """Lagged NYC curves + cross-zone daily means and last-day spatial curve."""
    h = samples.history
    price = h[:, :, TARGET]
    return np.concatenate(
        [
            price,
            h[:, -24:, :-1].reshape(len(h), -1),
            h[:, :, :-1].mean(axis=1),
            h[:, :, :-1].std(axis=1),
            h[:, :, -1].max(axis=1, keepdims=True),
            samples.calendar,
        ],
        axis=1,
    )
