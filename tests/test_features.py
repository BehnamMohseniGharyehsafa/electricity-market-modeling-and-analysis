import numpy as np
import pandas as pd

from electricity_market.data import TARGET
from electricity_market.features import build_samples, tabular_features


def test_future_prices_cannot_change_past_inputs(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    curves = np.arange(30 * 24 * 11, dtype=np.float32).reshape(30, 24, 11)
    path = tmp_path / "daily.npz"
    np.savez(path, dates=dates.to_numpy(), curves=curves, normal=np.ones(30, dtype=bool))
    before = build_samples(path)
    curves[20:] = -999
    np.savez(path, dates=dates.to_numpy(), curves=curves, normal=np.ones(30, dtype=bool))
    after = build_samples(path)
    # Includes the day whose target changed: its inputs must still be identical.
    mask = before.dates <= dates[20]
    np.testing.assert_array_equal(before.history[mask], after.history[mask])
    np.testing.assert_array_equal(tabular_features(before)[mask], tabular_features(after)[mask])
    idx = before.dates.get_loc(dates[20])
    np.testing.assert_array_equal(
        before.history[idx, -24:, TARGET],
        np.arange(30 * 24 * 11).reshape(30, 24, 11)[19, :, TARGET],
    )


def test_dst_targets_excluded_history_retained(tmp_path):
    dates = pd.date_range("2024-03-01", periods=20)
    normal = np.ones(20, dtype=bool)
    normal[9] = False
    path = tmp_path / "daily.npz"
    np.savez(path, dates=dates.to_numpy(), curves=np.ones((20, 24, 11)), normal=normal)
    samples = build_samples(path)
    assert pd.Timestamp("2024-03-10") not in samples.dates
    assert samples.history[samples.dates.get_loc("2024-03-11"), -24:, -1].sum() == 24
