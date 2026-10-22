import numpy as np
import pandas as pd
import pytest

from electricity_market.data import PRICE, ZONES, parse_daily


def make_day(day):
    start = pd.Timestamp(day)
    hours = pd.date_range(
        start, start + pd.Timedelta(days=1), freq="h", inclusive="left", tz="America/New_York"
    )
    return pd.DataFrame(
        [
            {"Time Stamp": t.strftime("%m/%d/%Y %H:%M"), "Name": zone, PRICE: float(t.hour + i)}
            for t in hours
            for i, zone in enumerate(ZONES)
        ]
    )


@pytest.mark.parametrize(
    "day,regular", [("2024-01-01", True), ("2024-03-10", False), ("2024-11-03", False)]
)
def test_dst_clock_day(day, regular):
    curve, normal, _ = parse_daily(make_day(day))
    assert curve.shape == (24, 11)
    assert normal == regular
    assert np.isfinite(curve).all()


def test_reject_non_dst_duplicate():
    frame = make_day("2024-01-01")
    with pytest.raises(ValueError, match="hour multiplicity"):
        parse_daily(pd.concat([frame, frame.iloc[:11]]))


def test_reject_missing_zone():
    frame = make_day("2024-01-01")
    with pytest.raises(ValueError):
        parse_daily(frame.loc[frame.Name != "WEST"])


def test_negative_prices_preserved():
    frame = make_day("2024-01-01")
    frame.loc[0, PRICE] = -100
    curve, _, _ = parse_daily(frame)
    assert curve.min() == -100
