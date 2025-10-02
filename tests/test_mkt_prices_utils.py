"""Tests for the `market_analy.utils.mkt_price_utils` module."""

import pandas as pd

from market_analy.utils import mkt_prices_utils as m


def test_request_daily_prices():
    f = m.request_daily_prices
    dt = pd.Timestamp("2023-01-10 14:45")
    date = pd.Timestamp("2023-01-10")

    assert f(start=date)
    assert f(end=date)
    assert f(start=date, end=date + pd.Timedelta("1D"))
    assert f(start=date, days=3)
    assert f(start=date, weeks=3)
    assert f(start=date, years=1, months=2, weeks=3)

    assert not f(start=dt)
    assert not f(end=dt)
    assert not f(start=dt, end=dt)
    assert not f(start=dt, days=3)
    assert not f(end=dt, days=3)
    assert not f(start=dt, months=3)
    assert not f(start=dt, years=1)

    for v in [date, dt]:
        assert not f(start=v, minutes=5)
        assert not f(start=v, hours=1)
        assert not f(start=v, hours=1, minutes=20)
        assert not f(end=v, minutes=5)
        assert not f(end=v, hours=1)
        assert not f(end=v, hours=1, minutes=20)


def test_period_string():
    f = m.period_string

    # test start and end
    rtrn = f(start=pd.Timestamp("2022-01-07"), end=pd.Timestamp("2023-01-10"))
    assert rtrn == "2022-01-07 to 2023-01-10"
    rtrn = f(start=pd.Timestamp("2023-01-07"), end=pd.Timestamp("2023-01-10"))
    assert rtrn == "2023-01-07 to 2023-01-10"

    # test only start or only end
    assert f(start=pd.Timestamp("2022-01-07")) == "since 2022-01-07"
    assert f(end=pd.Timestamp("2022-01-07")) == "to 2022-01-07"

    # test duration only pp
    assert f(minutes=15) == "15T"
    assert f(hours=3) == "3H"
    assert f(minutes=15, hours=3) == "3H 15T"
    assert f(days=3) == "3D"
    assert f(weeks=3) == "3W"
    assert f(months=3) == "3M"
    assert f(years=3) == "3Y"
    assert f(years=1, months=2, weeks=3) == "1Y 2M 3W"

    # test duration pp plus either start or end
    dt = pd.Timestamp("2023-01-10 14:45")
    assert f(start=dt, minutes=15, hours=3) == "3H 15T from 2023-01-10 14:45"
    assert f(end=dt, minutes=15, hours=3) == "3H 15T to 2023-01-10 14:45"
    date = pd.Timestamp("2023-01-10")
    assert f(start=date, weeks=2) == "2W from 2023-01-10"
    assert f(end=date, months=3) == "3M to 2023-01-10"


def test_range_string():
    f = m.range_string

    intervals = [
        pd.Interval(pd.Timestamp("2022-12-20 14:45"), pd.Timestamp("2022-12-20 15:00")),
        pd.Interval(pd.Timestamp("2023-01-06 15:00"), pd.Timestamp("2023-01-06 15:15")),
        pd.Interval(pd.Timestamp("2023-01-06 15:15"), pd.Timestamp("2023-01-06 15:30")),
    ]
    index = pd.IntervalIndex(intervals)

    # test when shand should have no effect
    for shand in [True, False]:
        assert (
            f(index, close=False, shand=shand)
            == f(index)
            == "2022-12-20 14:45 to 2023-01-06 15:30"
        )
    for shand in [True, False]:
        assert (
            f(index, close=True, shand=shand) == "2022-12-20 15:00 to 2023-01-06 15:30"
        )

    intervals = [
        pd.Interval(pd.Timestamp("2023-01-06 14:45"), pd.Timestamp("2023-01-06 15:00")),
        pd.Interval(pd.Timestamp("2023-01-06 15:00"), pd.Timestamp("2023-01-06 15:15")),
        pd.Interval(pd.Timestamp("2023-01-06 15:15"), pd.Timestamp("2023-01-06 15:30")),
    ]
    index = pd.IntervalIndex(intervals)

    # test effect of shand
    expected = "2023-01-06 14:45 to 2023-01-06 15:30"
    assert f(index, close=False) == f(index, close=False, shand=False) == expected
    expected = "2023-01-06 15:00 to 2023-01-06 15:30"
    assert f(index, close=True) == f(index, close=True, shand=False) == expected

    assert f(index, shand=True) == "2023-01-06 14:45 to 15:30"
    assert f(index, close=True, shand=True) == "2023-01-06 15:00 to 30"

    intervals = [
        pd.Interval(pd.Timestamp("2023-01-06 14:45"), pd.Timestamp("2023-01-06 15:00")),
        pd.Interval(pd.Timestamp("2023-01-08 15:15"), pd.Timestamp("2023-01-08 15:30")),
    ]
    index = pd.IntervalIndex(intervals)
    assert f(index, shand=True) == "2023-01-06 14:45 to 08 15:30"

    intervals = [
        pd.Interval(pd.Timestamp("2023-01-06 14:45"), pd.Timestamp("2023-01-06 15:00")),
        pd.Interval(pd.Timestamp("2023-02-06 15:15"), pd.Timestamp("2023-02-06 15:30")),
    ]
    index = pd.IntervalIndex(intervals)
    assert f(index, shand=True) == "2023-01-06 14:45 to 02-06 15:30"

    # test for passsing as a pd.DatetimeIndex
    index = pd.DatetimeIndex(["2023-01-05", "2023-01-06", "2023-01-09"])
    assert f(index, close=False) == "2023-01-05 to 2023-01-09"
    assert f(index, close=True) == "2023-01-06 to 2023-01-09"
    assert f(index, shand=True) == "2023-01-05 to 09"
