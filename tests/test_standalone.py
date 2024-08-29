"""Tests for the `market_analy.standalone` module."""

import pandas as pd
from pandas.testing import assert_frame_equal

from market_analy import standalone as m


def test_net_of():
    df = pd.DataFrame(
        {
            ("A", "open"): [18, 90, 50, 52, 78],
            ("A", "high"): [25, 120, 51, 52, 208],
            ("A", "low"): [15, 80, 49, 52, 52],
            ("A", "close"): [20, 100, 50, 52, 104],
            ("B", "open"): [222, 333, 444, 555, 666],
            ("B", "high"): [444, 555, 666, 777, 888],
            ("B", "low"): [111, 222, 333, 444, 555],
            ("B", "close"): [10, 40, 30, 28.5, 57],
        }
    )

    rtrn = m.net_of(df, rebase=False).round(3)

    # 2nd day, A rises 500%, B rises 400%, so net is 100% rise (i.e. doubling)
    # 3rd day, A drops 50%, B drops 25%, so net is a 25% drop
    # 4rd day, A rises 4%, B falls 5%, so net is a 9% rise
    # 5th day, A rises 100%, B rises 100%, so net is unchanged
    expected_rtrn = pd.DataFrame(
        {
            "open": [18, 36, 30, 32.7, 24.525],
            "high": [25, 48, 30.6, 32.7, 65.4],
            "low": [15, 32, 29.4, 32.7, 16.35],
            "close": [20, 40, 30, 32.7, 32.7],
        }
    )

    assert_frame_equal(rtrn, expected_rtrn)

    rtrn = m.net_of(df, rebase=True).round(3)

    # test when rebased, as is by default

    expected_rtrn = pd.DataFrame(
        {
            "open": [90, 180, 150, 163.5, 122.625],
            "high": [125, 240, 153, 163.5, 327],
            "low": [75, 160, 147, 163.5, 81.75],
            "close": [100, 200, 150, 163.5, 163.5],
        }
    )

    assert_frame_equal(rtrn, expected_rtrn)

    # as first test although includes volume column which should
    # be incuded in the result unchanged and wihtout influence from
    # other volume
    df = pd.DataFrame(
        {
            ("A", "open"): [18, 90, 50, 52, 78],
            ("A", "high"): [25, 120, 51, 52, 208],
            ("A", "low"): [15, 80, 49, 52, 52],
            ("A", "close"): [20, 100, 50, 52, 104],
            ("A", "volume"): [1, 2, 3, 4, 5],
            ("B", "open"): [222, 333, 444, 555, 666],
            ("B", "high"): [444, 555, 666, 777, 888],
            ("B", "low"): [111, 222, 333, 444, 555],
            ("B", "close"): [10, 40, 30, 28.5, 57],
            ("B", "volume"): [1111, 2222, 3333, 4444, 5555],
        }
    )

    rtrn = m.net_of(df, rebase=False).round(3)

    expected_rtrn = pd.DataFrame(
        {
            "open": [18, 36, 30, 32.7, 24.525],
            "high": [25, 48, 30.6, 32.7, 65.4],
            "low": [15, 32, 29.4, 32.7, 16.35],
            "close": [20, 40, 30, 32.7, 32.7],
            "volume": [1, 2, 3, 4, 5],
        }
    )

    assert_frame_equal(rtrn, expected_rtrn)
