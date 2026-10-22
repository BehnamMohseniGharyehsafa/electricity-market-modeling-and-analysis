import numpy as np
import pandas as pd
import pytest

from electricity_market.features import build_samples
from electricity_market.inference import date_inputs


def test_inference_matches_training_inputs_and_needs_no_target(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30)
    rng = np.random.default_rng(42)
    curves = rng.normal(30, 10, size=(30, 24, 11)).astype(np.float32)
    path = tmp_path / "daily.npz"
    np.savez(path, dates=dates.to_numpy(), curves=curves, normal=np.ones(30, dtype=bool))
    full = build_samples(path)
    day = dates[20]
    trained = full.dates.get_loc(day)
    # Omit the target day and every subsequent day from the inference data.
    np.savez(path, dates=dates[:20].to_numpy(), curves=curves[:20], normal=np.ones(20, dtype=bool))
    inferred = date_inputs(path, str(day.date()))
    np.testing.assert_array_equal(inferred.history[0], full.history[trained])
    np.testing.assert_array_equal(inferred.calendar[0], full.calendar[trained])


def test_inference_rejects_unsupported_dst_target():
    with pytest.raises(ValueError, match="DST transition"):
        date_inputs("not-read.npz", "2025-03-09")
