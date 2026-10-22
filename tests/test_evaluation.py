import numpy as np
import pytest

from electricity_market.evaluation import (
    block_bootstrap_delta,
    conformal_radius,
    interval_metrics,
    point_metrics,
)


def test_scores_work_at_zero_and_negative_prices():
    y = np.array([[-10.0, 0.0, 10.0]])
    scores = point_metrics(y, y + 2, 5)
    assert scores["mae"] == 2
    assert scores["rmse"] == 2
    assert scores["bias"] == 2


def test_conformal_corrected_order_statistic():
    scores = np.arange(10)[:, None]
    # ceil((10+1)*.8)=9; 9th order statistic equals 8.
    assert conformal_radius(scores)[0] == 8


def test_interval_score_and_crossing():
    y = np.array([[5.0, 15.0]])
    scores = interval_metrics(y, np.zeros_like(y), np.ones_like(y) * 10)
    assert scores["coverage"] == 0.5
    assert scores["interval_score"] == 35
    with pytest.raises(ValueError, match="Crossed"):
        interval_metrics(y, y + 1, y - 1)


def test_paired_bootstrap_constant_difference():
    result = block_bootstrap_delta(np.ones(30), np.ones(30) * 3)
    assert result["delta_mae_vs_weekly"] == -2
    assert result["ci95_low"] == -2
    assert result["ci95_high"] == -2
