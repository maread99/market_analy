"""Tests for analysis module.

For most methods of Analysis and Compare testing consists of verifying the
return from one or two specific calls against hard-coded expected returns.
"""

from collections import abc
import contextlib
import io
import pickle
import re

import bqplot as bq
import market_prices as mp
import numpy as np
import pandas as pd
from pandas import Timestamp as T
from pandas.testing import assert_series_equal, assert_frame_equal, assert_index_equal
import pytest

from market_analy import analysis, guis, trends
from market_analy.utils import bq_utils as bqu

# pylint: disable=too-many-lines
# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, unused-argument, invalid-name
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway.
#   invalid-name: names in tests not expected to strictly conform with snake_case.


def verify_app(f, cls, *args, **kwargs):
    """Verify `f` returns instance of `cls` and displays a `ipyvuetify.App`."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert isinstance(f(*args, **kwargs), cls)
    assert stdout.getvalue()[:3] == "App"


def test_pct_chg_and_abs_chg():
    f = analysis.pct_chg_top_to_bot
    srs = pd.Series(range(30, 41))
    assert round(f(srs), 5) == 0.33333
    srs2 = pd.Series(range(30, 19, -1))
    assert round(f(srs2), 5) == -0.33333
    df = pd.DataFrame({"a": srs, "b": srs2})
    assert_series_equal(f(df).round(4), pd.Series([0.3333, -0.3333], index=["a", "b"]))

    f = analysis.abs_chg_top_to_bot
    assert f(srs) == 10
    assert f(srs2) == -10
    df = pd.DataFrame({"a": srs, "b": srs2})
    assert_series_equal(f(df).round(5), pd.Series([10, -10], index=["a", "b"]))


def test_add_summary():
    f = analysis.add_summary
    index = list(range(4))
    a = pd.Series(index, index=index)
    b = pd.Series(list(range(0, 40, 10)), index=index)
    df = pd.DataFrame(dict(a=a, b=b))

    # check axis "row"
    expected = df.copy()
    expected.loc["SUMMARY"] = [1.5, 15]
    assert_frame_equal(f(df, aggmethod="mean", axis="row"), expected)
    assert_frame_equal(f(df, axis="row"), expected)  # check dflt

    # check axis "column"
    expected = df.copy()
    expected["Foo"] = [0, 5.5, 11, 16.5]
    assert_frame_equal(f(df, axis="column", label="Foo"), expected)

    # check axis "both"
    expected = df.copy()
    expected.loc["SUMMARY"] = [6, 60]
    expected["SUMMARY"] = [0, 11, 22, 33, np.NaN]
    assert_frame_equal(f(df, aggmethod="sum", axis="both"), expected)

    # check invalid axis
    with pytest.raises(ValueError, match="`axis` invalid"):
        f(df, axis="invalid")


def test_add_summary_row():
    f = analysis.add_summary_row
    index = list(range(4))
    a = pd.Series(index, index=index)
    b = pd.Series(list(range(0, 40, 10)), index=index)
    df = pd.DataFrame(dict(a=a, b=b))
    rtrn = f(df, [("mean", "a")])
    expected = df.copy()
    expected.loc["SUMMARY"] = [1.5, np.NaN]
    assert_frame_equal(rtrn, expected)

    # verify can pass in a list of a single column
    rtrn = f(df, [("sum", "b")], label="foO")
    expected = df.copy()
    expected.loc["foO"] = [0, 60]
    expected.loc[("foO", "a")] = np.NaN
    assert_frame_equal(rtrn, expected)

    rtrn = f(df, [("mean", "a"), ("sum", "b")], label="spam")
    expected = df.copy()
    expected.loc["spam"] = [0, 60]
    expected.loc[("spam", "a")] = 1.5
    assert_frame_equal(rtrn, expected)


def test__add_duration_columns():
    f = analysis._add_duration_columns
    d = {"a_key": "a_value"}
    rtrn = f(d.copy(), T("2023-01-04"), T("2023-01-07"))
    assert rtrn == {"a_key": "a_value", "days": [3]}
    rtrn = f(d.copy(), T("2023-01-04 13:22"), T("2023-01-07 13:44"))
    assert rtrn == {"a_key": "a_value", "days": [3], "minutes": [22]}
    rtrn = f(d.copy(), T("2023-01-04 13:22"), T("2023-01-07 16:44"))
    assert rtrn == {"a_key": "a_value", "days": [3], "hours": [3], "minutes": [22]}


def test_max_advance_and_max_decline():
    opens = closes = [
        100,
        105,
        110,
        105,
        100,
        95,
        100,
        105,
        110,
        115,
        110,
        105,
        100,
        95,
        91,
        100,
        110,
        119,
        110,
        100,
        90,
        86,
        90,
        100,
    ]
    highs = [v + 1 for v in opens]
    lows = [v - 1 for v in opens]

    # test max_adv
    # ...with daily index
    index = pd.date_range("2023-01-01 12:43", freq="D", periods=len(opens))
    df = pd.DataFrame(dict(open=opens, high=highs, low=lows, close=closes), index=index)
    rtrn = analysis.max_advance(df, label="max_adv")
    assert len(rtrn) == 1
    expected_cols = pd.Index(["start", "low", "end", "high", "pct_chg", "days"])
    assert_index_equal(rtrn.columns, expected_cols)
    assert rtrn.loc[("max_adv", "start")] == index[14]
    assert rtrn.loc[("max_adv", "low")] == lows[14]
    assert rtrn.loc[("max_adv", "end")] == index[17]
    assert rtrn.loc[("max_adv", "high")] == highs[17]
    assert round(rtrn.loc[("max_adv", "pct_chg")], 3) == 0.333
    assert rtrn.loc[("max_adv", "days")] == 3

    # ...with minute index
    index = pd.date_range("2023-01-01 12:43", freq="T", periods=len(opens))
    df = pd.DataFrame(dict(open=opens, high=highs, low=lows, close=closes), index=index)
    rtrn = analysis.max_advance(df, label="max_adv")
    assert len(rtrn) == 1
    expected_cols = pd.Index(["start", "low", "end", "high", "pct_chg", "minutes"])
    assert_index_equal(rtrn.columns, expected_cols)
    assert rtrn.loc[("max_adv", "start")] == index[14]
    assert rtrn.loc[("max_adv", "low")] == lows[14]
    assert rtrn.loc[("max_adv", "end")] == index[17]
    assert rtrn.loc[("max_adv", "high")] == highs[17]
    assert round(rtrn.loc[("max_adv", "pct_chg")], 3) == 0.333
    assert rtrn.loc[("max_adv", "minutes")] == 3
    round(rtrn.loc[("max_adv", "pct_chg")], 3)

    # test max_dec
    # ...with daily index
    index = pd.date_range("2023-01-01 12:43", freq="D", periods=len(opens))
    df = pd.DataFrame(dict(open=opens, high=highs, low=lows, close=closes), index=index)
    rtrn = analysis.max_decline(df, label="max_dec")
    assert len(rtrn) == 1
    expected_cols = pd.Index(["start", "high", "end", "low", "pct_chg", "days"])
    assert_index_equal(rtrn.columns, expected_cols)
    assert rtrn.loc[("max_dec", "start")] == index[17]
    assert rtrn.loc[("max_dec", "high")] == highs[17]
    assert rtrn.loc[("max_dec", "end")] == index[21]
    assert rtrn.loc[("max_dec", "low")] == lows[21]
    assert round(rtrn.loc[("max_dec", "pct_chg")], 3) == -0.292
    assert rtrn.loc[("max_dec", "days")] == 4

    # ...with minute index
    index = pd.date_range("2023-01-01 12:43", freq="T", periods=len(opens))
    df = pd.DataFrame(dict(open=opens, high=highs, low=lows, close=closes), index=index)
    rtrn = analysis.max_decline(df, label="max_dec")
    assert len(rtrn) == 1
    expected_cols = pd.Index(["start", "high", "end", "low", "pct_chg", "minutes"])
    assert_index_equal(rtrn.columns, expected_cols)
    assert rtrn.loc[("max_dec", "start")] == index[17]
    assert rtrn.loc[("max_dec", "high")] == highs[17]
    assert rtrn.loc[("max_dec", "end")] == index[21]
    assert rtrn.loc[("max_dec", "low")] == lows[21]
    assert round(rtrn.loc[("max_dec", "pct_chg")], 3) == -0.292
    assert rtrn.loc[("max_dec", "minutes")] == 4


def test_style_df():
    opens = closes = [100, 110, 105]
    highs = [v + 1 for v in opens]
    lows = [v - 1 for v in opens]
    index = pd.date_range("2023-01-01", freq="D", periods=len(opens))
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adjclose": closes,
            "volume": [1000, 2000, 3000],
            "date": index,
            "start": index - pd.Timedelta(days=1),
            "end": index + pd.Timedelta(hours=12, minutes=33),
            "days": 3,
            "hours": 5.0,
            "minutes": 33.0,
        },
        index=index,
    )
    df["pct_chg"] = (df.close - df.close.shift(1)) / df.close.shift(1)
    df["chg"] = df.close - df.close.shift(1)

    rtrn = analysis.style_df(df, na_rep="", caption="A caption")
    # expected verified as required from inspection
    expected = "\\begin{table}\n\\caption{A caption}\n\\begin{tabular}{lrrrrrrlllrrrrr}\n & open & high & low & close & adjclose & volume & date & start & end & days & hours & minutes & pct_chg & chg \\\\\n2023-01-01 00:00:00 & 100.00 & 101.00 & 99.00 & 100.00 & 100.00 & 1000 & 2023-01-01 & 2022-12-31 & 2023-01-01 12:33 & 3 & 5 & 33 & \\colorYellowGreen  & \\colorYellowGreen  \\\\\n2023-01-02 00:00:00 & 110.00 & 111.00 & 109.00 & 110.00 & 110.00 & 2000 & 2023-01-02 & 2023-01-01 & 2023-01-02 12:33 & 3 & 5 & 33 & \\colorYellowGreen 10.0% & \\colorYellowGreen 10.00 \\\\\n2023-01-03 00:00:00 & 105.00 & 106.00 & 104.00 & 105.00 & 105.00 & 3000 & 2023-01-03 & 2023-01-02 & 2023-01-03 12:33 & 3 & 5 & 33 & \\colorCrimson -4.55% & \\colorCrimson -5.00 \\\\\n\\end{tabular}\n\\end{table}\n"
    assert rtrn.to_latex() == expected


def test_style_df_grid():
    a = [-0.1, 0, 0.1, 0.11, 0.12]
    b = [-0.2, 0, 0.23, 0.24, 0.25]
    c = [-0.3, 0, 0.36, 0.37, 0.38]
    df = pd.DataFrame(dict(A=a, B=b, C=c), index=["A", "B", "C", "D", "E"])
    rtrn = analysis.style_df_grid(df, caption="A capt")
    # expected verified as required from inspection
    expected = "\\begin{table}\n\\caption{A capt}\n\\begin{tabular}{lrrr}\n & A & B & C \\\\\nA & \\background-color#f2a17f \\color#000000 -10.00% & \\background-color#c2383a \\color#f1f1f1 -20.00% & \\background-color#67001f \\color#f1f1f1 -30.00% \\\\\nB & \\background-color#fbe6da \\color#000000 0.00% & \\background-color#fbe6da \\color#000000 0.00% & \\background-color#fbe6da \\color#000000 0.00% \\\\\nC & \\background-color#d5e7f1 \\color#000000 10.00% & \\background-color#529dc8 \\color#f1f1f1 23.00% & \\background-color#0d3f76 \\color#f1f1f1 36.00% \\\\\nD & \\background-color#cfe4ef \\color#000000 11.00% & \\background-color#4695c4 \\color#f1f1f1 24.00% & \\background-color#08366a \\color#f1f1f1 37.00% \\\\\nE & \\background-color#c5dfec \\color#000000 12.00% & \\background-color#3f8ec0 \\color#f1f1f1 25.00% & \\background-color#053061 \\color#f1f1f1 38.00% \\\\\n\\end{tabular}\n\\end{table}\n"
    assert rtrn.to_latex() == expected


@pytest.fixture(scope="class")
def daily_pp():
    """Period parameters covering 2023-01-01 thorugh 2023-01-10."""
    return {
        "start": T("2023-01-06"),
        "end": T("2023-01-10"),
    }


@pytest.fixture(scope="class")
def intraday_pp():
    """Period parameters covering period defined with intraday extremes.

    Period parameters covere period from "2023-01-06 14:45" UTC to
    "2023-01-09 16:15" UTC.
    """
    return {
        "start": T("2023-01-06 14:45", tz="UTC"),
        "end": T("2023-01-10 16:15", tz="UTC"),
    }


class TestAnalysis:
    """Tests for the `analysis.Analysis` class."""

    @pytest.fixture(scope="class")
    def analy(self, prices_analysis) -> abc.Iterator[analysis.Analysis]:
        yield analysis.Analysis(prices_analysis)

    @pytest.fixture(scope="class")
    def analy_other(self, prices_analysis_other) -> abc.Iterator[analysis.Analysis]:
        yield analysis.Analysis(prices_analysis_other)

    def test_constructor_raises(self):
        msg = re.escape(
            "The Analysis class requires a `prices` instance that gets"
            "price data for a single symbol, although the past instance"
            " gets prices for 2: ['AZN.L', 'MSFT']."
        )
        with pytest.raises(ValueError, match=msg):
            analysis.Analysis(mp.PricesYahoo("AZN.L, MSFT"))

    def test_prices(self, analy, prices_analysis):
        assert analy.prices is prices_analysis

    def test_symbol(self, analy):
        assert analy.symbol == "AZN.L"

    def test_daily_prices(self, analy, daily_pp, intraday_pp):
        expected = pd.DataFrame(
            {
                "open": {
                    T("2023-01-06 00:00:00"): 11734.0,
                    T("2023-01-09 00:00:00"): 11694.0,
                    T("2023-01-10 00:00:00"): 11714.0,
                },
                "high": {
                    T("2023-01-06 00:00:00"): 11808.0,
                    T("2023-01-09 00:00:00"): 11758.0,
                    T("2023-01-10 00:00:00"): 11886.0,
                },
                "low": {
                    T("2023-01-06 00:00:00"): 11690.0,
                    T("2023-01-09 00:00:00"): 11602.0,
                    T("2023-01-10 00:00:00"): 11666.0,
                },
                "close": {
                    T("2023-01-06 00:00:00"): 11782.0,
                    T("2023-01-09 00:00:00"): 11736.0,
                    T("2023-01-10 00:00:00"): 11802.0,
                },
                "volume": {
                    T("2023-01-06 00:00:00"): 4153691.0,
                    T("2023-01-09 00:00:00"): 1989665.0,
                    T("2023-01-10 00:00:00"): 3318361.0,
                },
            }
        )

        expected.columns.name = ""
        rtrn = analy.daily_prices(**daily_pp)
        assert_frame_equal(rtrn, expected, check_freq=None)

        expected = pd.DataFrame(
            {
                "open": {T("2023-01-09 00:00:00"): 11694.0},
                "high": {T("2023-01-09 00:00:00"): 11758.0},
                "low": {T("2023-01-09 00:00:00"): 11602.0},
                "close": {T("2023-01-09 00:00:00"): 11736.0},
                "volume": {T("2023-01-09 00:00:00"): 1989665.0},
            },
        )
        expected.columns.name = ""
        rtrn = analy.daily_prices(**intraday_pp)
        assert_frame_equal(rtrn, expected, check_freq=None)

    def test_daily_close_prices(self, analy, daily_pp, intraday_pp):
        rtrn = analy.daily_close_prices(**daily_pp)
        expected = pd.DataFrame(
            {
                "AZN.L": {
                    T("2023-01-06 00:00:00"): 11782.0,
                    T("2023-01-09 00:00:00"): 11736.0,
                    T("2023-01-10 00:00:00"): 11802.0,
                },
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected, check_freq=False)

        rtrn = analy.daily_close_prices(**intraday_pp)
        expected = pd.DataFrame({"AZN.L": {T("2023-01-09 00:00:00"): 11736.0}})
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected, check_freq=False)

    def test_price_chg(self, analy, daily_pp, intraday_pp):
        f = analy.price_chg
        # test for daily_pp
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame({"chg": {"AZN.L": 92.0}})
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Price Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lr}\n & chg \\\\\nsymbol &  \\\\\nAZN.L & \\colorYellowGreen 92.00 \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**daily_pp)
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame({"chg": {"AZN.L": 72.32}})
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Price Change 2023-01-06 14:45 to 2023-01-10 16:15}\n\\begin{tabular}{lr}\n & chg \\\\\nsymbol &  \\\\\nAZN.L & \\colorYellowGreen 72.32 \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**intraday_pp)
        assert rtrn.to_latex() == expected

    def test_pct_chg(self, analy, daily_pp, intraday_pp):
        # test for daily_pp
        f = analy.pct_chg

        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame({"pct_chg": {"AZN.L": 0.007856532877882128}})
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Percentage Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lr}\n & pct_chg \\\\\nsymbol &  \\\\\nAZN.L & \\colorYellowGreen 0.79% \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**daily_pp)
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame({"pct_chg": {"AZN.L": 0.006164363493010638}})
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Percentage Change 2023-01-06 14:45 to 2023-01-10 16:15}\n\\begin{tabular}{lr}\n & pct_chg \\\\\nsymbol &  \\\\\nAZN.L & \\colorYellowGreen 0.62% \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**intraday_pp)
        assert rtrn.to_latex() == expected

    def test_summary_chg(self, analy):
        """Test `Analysis.summary_chg`.

        Notes
        -----
        Does not test default (no args) due to the mocked `prices` instance
        being unable to request 1h prices due to a limitation within the
        mocking. NB The default is tested within the `TestCompare` class.
        """
        f = analy.summary_chg
        # test specific periods
        periods = [
            {"minutes": 5},
            {"minutes": 15},
            {"days": 1},
            {"days": 2},
            {"days": 5},
            {"weeks": 2},
            {"months": 1},
            {"months": 3},
            {"months": 6},
            {"years": 1},
        ]
        rtrn = f(periods=periods, style=False)
        expected = pd.DataFrame(
            {
                "5T": {"AZN.L": 0.003910068426197455},
                "15T": {"AZN.L": 0.0027338410466706264},
                "1D": {"AZN.L": -0.008365758754863784},
                "2D": {"AZN.L": -0.037393767705382386},
                "5D": {"AZN.L": -0.04227733934611044},
                "2W": {"AZN.L": -0.10735551663747811},
                "1M": {"AZN.L": -0.09128186842574437},
                "3M": {"AZN.L": -0.03098859315589353},
                "6M": {"AZN.L": -0.05435992578849724},
                "1Y": {"AZN.L": 0.1878350034956886},
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = f(periods=periods)
        expected = "\\begin{table}\n\\caption{Percentage Change}\n\\begin{tabular}{lrrrrrrrrrr}\n & 5T & 15T & 1D & 2D & 5D & 2W & 1M & 3M & 6M & 1Y \\\\\nsymbol &  &  &  &  &  &  &  &  &  &  \\\\\nAZN.L & \\colorYellowGreen 0.39% & \\colorYellowGreen 0.27% & \\colorCrimson -0.84% & \\colorCrimson -3.74% & \\colorCrimson -4.23% & \\colorCrimson -10.7% & \\colorCrimson -9.13% & \\colorCrimson -3.10% & \\colorCrimson -5.44% & \\colorYellowGreen 18.8% \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_chg(self, analy, intraday_pp, daily_pp):
        # test for daily_pp
        rtrn = analy.chg(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {"AZN.L": 0.007856532877882128},
                "chg": {"AZN.L": 92.0},
                "close": {"AZN.L": 11802.0},
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = analy.chg(style=True, **daily_pp)
        expected = "\\begin{table}\n\\caption{Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lrrr}\n & pct_chg & chg & close \\\\\nsymbol &  &  &  \\\\\nAZN.L & \\colorYellowGreen 0.79% & \\colorYellowGreen 92.00 & 11802.00 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = analy.chg(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {"AZN.L": 0.006164363493010638},
                "chg": {"AZN.L": 72.3203125},
                "close": {"AZN.L": 11804.3203125},
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = analy.chg(style=True, **intraday_pp)
        expected = "\\begin{table}\n\\caption{Change 2023-01-06 14:45 to 2023-01-10 16:15}\n\\begin{tabular}{lrrr}\n & pct_chg & chg & close \\\\\nsymbol &  &  &  \\\\\nAZN.L & \\colorYellowGreen 0.62% & \\colorYellowGreen 72.32 & 11804.32 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_chg_every_interval(self, analy, intraday_pp):
        f = analy.chg_every_interval
        rtrn = f("90T", style=False, **intraday_pp, anchor="workback")
        expected = pd.DataFrame(
            {
                "AZN.L": {
                    pd.Interval(
                        T("2023-01-06 15:15:00", tz="Europe/London"),
                        T("2023-01-09 08:15:00", tz="Europe/London"),
                        closed="left",
                    ): -0.004094864357618154,
                    pd.Interval(
                        T("2023-01-09 08:15:00", tz="Europe/London"),
                        T("2023-01-09 09:45:00", tz="Europe/London"),
                        closed="left",
                    ): -0.0006852835360630461,
                    pd.Interval(
                        T("2023-01-09 09:45:00", tz="Europe/London"),
                        T("2023-01-09 11:15:00", tz="Europe/London"),
                        closed="left",
                    ): -0.0008571918395336877,
                    pd.Interval(
                        T("2023-01-09 11:15:00", tz="Europe/London"),
                        T("2023-01-09 12:45:00", tz="Europe/London"),
                        closed="left",
                    ): -0.0011701826044526423,
                    pd.Interval(
                        T("2023-01-09 12:45:00", tz="Europe/London"),
                        T("2023-01-09 14:15:00", tz="Europe/London"),
                        closed="left",
                    ): 0.0032329911891490672,
                    pd.Interval(
                        T("2023-01-09 14:15:00", tz="Europe/London"),
                        T("2023-01-09 15:45:00", tz="Europe/London"),
                        closed="left",
                    ): 0.00017123287671232877,
                    pd.Interval(
                        T("2023-01-09 15:45:00", tz="Europe/London"),
                        T("2023-01-10 08:45:00", tz="Europe/London"),
                        closed="left",
                    ): 0.005231907261171033,
                    pd.Interval(
                        T("2023-01-10 08:45:00", tz="Europe/London"),
                        T("2023-01-10 10:15:00", tz="Europe/London"),
                        closed="left",
                    ): 0.011315635802016374,
                    pd.Interval(
                        T("2023-01-10 10:15:00", tz="Europe/London"),
                        T("2023-01-10 11:45:00", tz="Europe/London"),
                        closed="left",
                    ): -0.005894240485011789,
                    pd.Interval(
                        T("2023-01-10 11:45:00", tz="Europe/London"),
                        T("2023-01-10 13:15:00", tz="Europe/London"),
                        closed="left",
                    ): -0.001694053870913095,
                    pd.Interval(
                        T("2023-01-10 13:15:00", tz="Europe/London"),
                        T("2023-01-10 14:45:00", tz="Europe/London"),
                        closed="left",
                    ): 0.0035635499745460715,
                    pd.Interval(
                        T("2023-01-10 14:45:00", tz="Europe/London"),
                        T("2023-01-10 16:15:00", tz="Europe/London"),
                        closed="left",
                    ): -0.002002002663172134,
                }
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = f("90T", style=True, **intraday_pp, anchor="workback")
        expected = "\\begin{table}\n\\caption{Change over prior 90T.   Period: 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC}\n\\begin{tabular}{lr}\nindex & AZN.L \\\\\n[2023-01-06 15:15:00, 2023-01-09 08:15:00) & \\colorCrimson -0.41% \\\\\n[2023-01-09 08:15:00, 2023-01-09 09:45:00) & \\colorCrimson -0.07% \\\\\n[2023-01-09 09:45:00, 2023-01-09 11:15:00) & \\colorCrimson -0.09% \\\\\n[2023-01-09 11:15:00, 2023-01-09 12:45:00) & \\colorCrimson -0.12% \\\\\n[2023-01-09 12:45:00, 2023-01-09 14:15:00) & \\colorYellowGreen 0.32% \\\\\n[2023-01-09 14:15:00, 2023-01-09 15:45:00) & \\colorYellowGreen 0.02% \\\\\n[2023-01-09 15:45:00, 2023-01-10 08:45:00) & \\colorYellowGreen 0.52% \\\\\n[2023-01-10 08:45:00, 2023-01-10 10:15:00) & \\colorYellowGreen 1.13% \\\\\n[2023-01-10 10:15:00, 2023-01-10 11:45:00) & \\colorCrimson -0.59% \\\\\n[2023-01-10 11:45:00, 2023-01-10 13:15:00) & \\colorCrimson -0.17% \\\\\n[2023-01-10 13:15:00, 2023-01-10 14:45:00) & \\colorYellowGreen 0.36% \\\\\n[2023-01-10 14:45:00, 2023-01-10 16:15:00) & \\colorCrimson -0.20% \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_chg_every_interval_chart(self, analy, intraday_pp):
        f = analy.chg_every_interval

        pp = {"start": "2022-01-01", "years": 1}
        verify_app(f, guis.PctChg, "10D", chart=True, **pp)

        gui = analy.chg_every_interval("10d", chart=True, _display=False, **pp)
        assert gui.chart.title == "Change over prior 10d.   Period: 1Y from 2022-01-01"
        assert gui.chart.axes[0].orientation == "horizontal"
        assert gui.chart.axes[1].orientation == "vertical"
        assert gui.chart.axes[1].orientation == "vertical"

        start, end = pd.Timestamp("2022-01-05"), pd.Timestamp("2023-01-04")
        assert start == gui.chart.data.index[0].left
        assert end == gui.chart.data.index[-1].right

        expected = pd.Interval(pd.Timestamp("2022-08-10"), end, "left")
        assert gui.chart.plotted_interval == expected
        expected_plottable = pd.Interval(start, end, "left")
        assert gui.chart.plottable_interval == expected_plottable
        slider_end = pd.Timestamp("2022-12-16")
        assert gui.date_slider.slider.value == (pd.Timestamp("2022-08-10"), slider_end)

        # verify max dates
        assert gui._icon_row_top.children[0].tooltip == "Max dates"
        gui._icon_row_top.children[0].click()
        assert gui.date_slider.slider.value == (start, slider_end)

        # verify can change slider
        gui.date_slider.slider.value = (
            pd.Timestamp("2022-04-13"),
            pd.Timestamp("2022-07-13"),
        )
        expected = pd.Interval(
            pd.Timestamp("2022-04-13"), pd.Timestamp("2022-07-27"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        assert gui._icon_row_top.children[-1].tooltip == "Close"
        gui._icon_row_top.children[-1].click()

        # verify for horizontal bars and intraday interval
        gui = analy.chg_every_interval(
            "21T", **intraday_pp, chart=True, direction="horizontal", _display=False
        )
        assert (
            gui.chart.title
            == "Change over prior 21T.   Period: 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC"
        )
        assert gui.chart.axes[0].orientation == "vertical"
        assert gui.chart.axes[1].orientation == "horizontal"
        assert gui.chart.axes[1].orientation == "horizontal"

        start, end = pd.Timestamp("2023-01-06 15:00:00"), pd.Timestamp(
            "2023-01-10 16:03:00"
        )
        assert start == gui.chart.data.index[0].left
        assert end == gui.chart.data.index[-1].right

        expected = pd.Interval(pd.Timestamp("2023-01-10 12:33"), end, "left")
        assert gui.chart.plotted_interval == expected
        expected_plottable = pd.Interval(start, end, "left")
        assert gui.chart.plottable_interval == expected_plottable
        slider_end = pd.Timestamp("2023-01-10 15:42")
        assert gui.date_slider.slider.value == (
            pd.Timestamp("2023-01-10 12:33"),
            slider_end,
        )

        # verify max dates
        assert gui._icon_row_top.children[0].tooltip == "Max dates"
        gui._icon_row_top.children[0].click()
        assert gui.date_slider.slider.value == (start, slider_end)

        # verify can change slider
        gui.date_slider.slider.value = (
            pd.Timestamp("2023-01-09 15:00"),
            pd.Timestamp("2023-01-10 09:24"),
        )
        expected = pd.Interval(
            pd.Timestamp("2023-01-09 15:00"), pd.Timestamp("2023-01-10 09:45"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        assert gui._icon_row_top.children[-1].tooltip == "Close"
        gui._icon_row_top.children[-1].click()

    def test_price_on(self, analy):
        """Test `Analysis.price_on`.

        Also tests:
            Passing date to `Analysis.price`.
            `Analysis.today` property.
        """
        f = analy.price_on
        # test default
        rtrn = f(style=False)
        expected = pd.DataFrame(
            {
                "pct_chg": {"AZN.L": -0.008365758754863784},
                "chg": {"AZN.L": -86.0},
                "close": {"AZN.L": 10194.0},
                "open": {"AZN.L": 10330.0},
                "high": {"AZN.L": 10342.0},
                "low": {"AZN.L": 10184.91796875},
                "volume": {"AZN.L": 764542.0},
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.price(style=False), expected)

        rtrn = f()
        expected = "\\begin{table}\n\\caption{Price 2023-02-02}\n\\begin{tabular}{lrrrrrrr}\n & pct_chg & chg & close & open & high & low & volume \\\\\nsymbol &  &  &  &  &  &  &  \\\\\nAZN.L & \\colorCrimson -0.84% & \\colorCrimson -86.00 & 10194.00 & 10330.00 & 10342.00 & 10184.92 & 764542 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected
        assert analy.today.to_latex() == expected
        assert analy.price().to_latex() == expected

        # test passing a session
        session = "2023-01-05"
        rtrn = f(session, style=False)
        expected = pd.DataFrame(
            {
                "pct_chg": {"AZN.L": 0.009308739872435856},
                "chg": {"AZN.L": 108.0},
                "close": {"AZN.L": 11710.0},
                "open": {"AZN.L": 11524.0},
                "high": {"AZN.L": 11730.0},
                "low": {"AZN.L": 11476.0},
                "volume": {"AZN.L": 2215897.0},
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.price(session, style=False), expected)

        rtrn = f(session)
        expected = "\\begin{table}\n\\caption{Price 2023-01-05}\n\\begin{tabular}{lrrrrrrr}\n & pct_chg & chg & close & open & high & low & volume \\\\\nsymbol &  &  &  &  &  &  &  \\\\\nAZN.L & \\colorYellowGreen 0.93% & \\colorYellowGreen 108.00 & 11710.00 & 11524.00 & 11730.00 & 11476.00 & 2215897 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_price_at(self, analy):
        """Test `Analysis.price_at`.

        Also tests:
            Passing time to `Analysis.price`.
            `Analysis.now` property.

        Note
        ----
        The default price_at (now) is not the price as at the time defined
        by the index as a result of the underlying `price.price_at`
        implementation returning the current close according to the daily
        data. In turn, this relates to the price as at a different time
        as a result of how the data is mocked for tests. The hard-coded
        'expected' value here is as would be expected to be returned by the
        underlying implementation, i.e. as required.
        """
        f = analy.price_at
        # test default
        rtrn = f()
        expected = pd.DataFrame(
            {"AZN.L": {T("2023-02-02 15:10:00+0000", tz="Europe/London"): 10194.0}}
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.now, expected)

        # test passing a value
        dt = "2023-01-06 09:33"
        rtrn = analy.price_at(dt)
        expected = pd.DataFrame(
            {
                "AZN.L": {
                    T("2023-01-06 09:33:00+0000", tz="Europe/London"): 11709.7802734375
                }
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)
        # test .price()
        assert_frame_equal(rtrn, analy.price(dt))

    def test_price_range(self, analy, intraday_pp):
        rtrn = analy.price_range(**intraday_pp)
        interval = pd.Interval(
            T("2023-01-06 14:45:00", tz="Europe/London"),
            T("2023-01-10 16:15:00", tz="Europe/London"),
            closed="right",
        )
        expected = pd.DataFrame(
            {
                ("AZN.L", "open"): {interval: 11732.0},
                ("AZN.L", "high"): {interval: 11886.0},
                ("AZN.L", "low"): {interval: 11602.0},
                ("AZN.L", "close"): {interval: 11804.3203125},
                ("AZN.L", "volume"): {interval: 2539545.0},
            }
        )
        expected.columns.names = ["symbol", ""]
        assert_frame_equal(rtrn, expected)

    def test_max_adv(self, analy, daily_pp):
        f = analy.max_adv
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "start": {"AZN.L": T("2023-01-09 15:00:00+0000", tz="Europe/London")},
                "low": {"AZN.L": 11602.0},
                "end": {"AZN.L": T("2023-01-10 10:00:00+0000", tz="Europe/London")},
                "high": {"AZN.L": 11886.0},
                "pct_chg": {"AZN.L": 0.024478538183071885},
                "hours": {"AZN.L": 19},
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Maximum Advance 2023-01-06 to 2023-01-10}\n\\begin{tabular}{llrlrrr}\n & start & low & end & high & pct_chg & hours \\\\\nAZN.L & 2023-01-09 15:00 & 11602.00 & 2023-01-10 10:00 & 11886.00 & \\colorYellowGreen 2.45% & 19 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_max_dec(self, analy, daily_pp):
        f = analy.max_dec
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "start": {"AZN.L": T("2023-01-06 16:00:00+0000", tz="Europe/London")},
                "high": {"AZN.L": 11808.0},
                "end": {"AZN.L": T("2023-01-09 15:00:00+0000", tz="Europe/London")},
                "low": {"AZN.L": 11602.0},
                "pct_chg": {"AZN.L": -0.017445799457994626},
                "days": {"AZN.L": 2},
                "hours": {"AZN.L": 23},
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Maximum Decline 2023-01-06 to 2023-01-10}\n\\begin{tabular}{llrlrrrr}\n & start & high & end & low & pct_chg & days & hours \\\\\nAZN.L & 2023-01-06 16:00 & 11808.00 & 2023-01-09 15:00 & 11602.00 & \\colorCrimson -1.74% & 2 & 23 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_corr(self, analy, analy_other):
        f = analy.corr
        rtrn = f(analy_other, style=False, end="2022-12-31", days=20)
        assert rtrn == 0.49620301268372446

        rtrn = f(analy_other, end="2022-12-31", days=20)
        expected = "\\begin{table}\n\\caption{Correlation 20D to 2022-12-31  (2022-12-01 to 30)}\n\\begin{tabular}{lr}\n & BARC.L \\\\\nAZN.L & \\background-color#6bacd1 \\color#f1f1f1 0.50 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_plot_line(self, analy, intraday_pp):
        """Verify can create line plot gui."""
        f = analy.plot
        verify_app(f, guis.ChartLine, **intraday_pp, chart_type="line")

    def test_plot_ohlc(self, analy, intraday_pp):
        """Verifies various aspects of gui beahviour for ohlc plots."""
        f = analy.plot
        verify_app(f, guis.ChartOHLC, **intraday_pp)

        gui = analy.plot(**intraday_pp, display=False)
        labels = [label for label, pt in gui._interval_selector.options]

        assert labels == ["1D", "4H", "1H", "30T", "15T", "5T", "2T", "1T"]

        expected_plottable = pd.Interval(
            intraday_pp["start"].tz_convert(None),
            intraday_pp["end"].tz_convert(None),
            "left",
        )
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.chart.plotted_interval == expected_plottable

        expected_interval = mp.intervals.TDInterval.T5
        assert gui._interval_selector.value == expected_interval
        expected_init = (
            intraday_pp["start"].tz_convert(None),
            intraday_pp["end"].tz_convert(None) - expected_interval,
        )
        assert gui.date_slider.slider.value == expected_init

        gui.date_slider.slider.value = (
            pd.Timestamp("2023-01-06 14:50"),
            pd.Timestamp("2023-01-09 15:35"),
        )
        expected = pd.Interval(
            pd.Timestamp("2023-01-06 14:50"), pd.Timestamp("2023-01-09 15:40"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        # verify all ranges as expected if change to longer interval
        gui._interval_selector.value = mp.intervals.TDInterval.T15
        expected = (pd.Timestamp("2023-01-06 15:00"), pd.Timestamp("2023-01-09 15:15"))
        assert gui.date_slider.slider.value == expected
        expected = pd.Interval(
            pd.Timestamp("2023-01-06 15:00"), pd.Timestamp("2023-01-09 15:30"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        # verify effect of max dates button
        assert gui._icon_row_top.children[0].tooltip == "Max dates"
        gui._icon_row_top.children[0].click()  # click max_dates
        assert gui.chart.plotted_interval == expected_plottable
        assert gui.chart.plottable_interval == expected_plottable
        expected = (
            intraday_pp["start"].tz_convert(None),
            intraday_pp["end"].tz_convert(None) - pd.Timedelta(15, "T"),
        )
        assert gui.date_slider.slider.value == expected

        # check edge of availability of 1D interval
        gui.date_slider.slider.value = (
            pd.Timestamp("2023-01-09 08:00"),
            pd.Timestamp("2023-01-09 16:15"),
        )
        expected = pd.Interval(
            pd.Timestamp("2023-01-09 08:00"), pd.Timestamp("2023-01-09 16:30"), "left"
        )
        assert gui.chart.plotted_interval == expected
        gui._interval_selector.value = mp.intervals.TDInterval.D1

        # plottable and plotted should be reduced to one day, as remainder either side is
        # less than one day and hence can't be otherwise represented by a 1D interval
        expected = pd.Interval(
            pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-10"), "left"
        )
        assert gui.chart.plottable_interval == expected
        assert gui.chart.plotted_interval == expected
        expected = (pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-09"))
        assert gui.date_slider.slider.value == expected

        # ...and if go back to 15T then the full range available will now be restricted to
        # the range that was available for 1D
        gui._interval_selector.value = mp.intervals.TDInterval.T15
        expected = pd.Interval(
            pd.Timestamp("2023-01-09 08:00"), pd.Timestamp("2023-01-09 16:30"), "left"
        )
        assert gui.chart.plottable_interval == expected
        assert gui.chart.plotted_interval == expected
        expected = (pd.Timestamp("2023-01-09 08:00"), pd.Timestamp("2023-01-09 16:15"))
        assert gui.date_slider.slider.value == expected

        # verify effect of reset button
        assert gui._icon_row_top.children[1].tooltip == "Reset"
        gui._icon_row_top.children[1].click()
        assert gui._interval_selector.value == mp.intervals.TDInterval.T5
        assert gui.chart.plotted_interval == expected_plottable
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == expected_init

        # verify raises dialog box when try to change interval to 1D for period < 1d
        slider_tup = (
            pd.Timestamp("2023-01-09 08:00"),
            pd.Timestamp("2023-01-09 16:20"),
        )
        gui.date_slider.slider.value = slider_tup
        expected = pd.Interval(
            pd.Timestamp("2023-01-09 08:00"), pd.Timestamp("2023-01-09 16:25"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert not gui._dialog.value
        gui._interval_selector.value = mp.intervals.TDInterval.D1
        assert gui._dialog.value
        assert (
            gui._dialog.text
            == "Interval '1D' unavailable: the currently plotted range is shorther than the requested interval."
        )
        gui._dialog.close_dialog()
        assert not gui._dialog.value
        # verify values unchanged
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == slider_tup

        # verify effect of clicking zoom
        gui.tabs_control.but_zoom.fire_event("click", None)
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == slider_tup

        # verify can call all intraday intervals
        gui._icon_row_top.children[1].click()  # reset chart
        gui._interval_selector.value = mp.intervals.TDInterval.T1
        gui._interval_selector.value = mp.intervals.TDInterval.T2
        gui._interval_selector.value = mp.intervals.TDInterval.T5
        gui._interval_selector.value = mp.intervals.TDInterval.T15
        gui._interval_selector.value = mp.intervals.TDInterval.T30
        gui._interval_selector.value = mp.intervals.TDInterval.H1
        gui._interval_selector.value = mp.intervals.TDInterval.H4
        gui._interval_selector.value = mp.intervals.TDInterval.D1

        # verify max_adv and crosshairs
        gui._icon_row_top.children[1].click()  # reset chart
        gui.tabs_control.v_model = 1
        assert not gui._html_output._html.value
        assert len(gui.crosshairs) == 0
        gui.tabs_control.but_arrow_up.fire_event("click", None)
        assert len(gui.crosshairs) == 2
        ch_bot = gui.crosshairs[0]
        assert ch_bot.hhair.y[0] == 11602
        assert ch_bot.vhair.x[0] == pd.Timestamp("2023-01-09 15:05")
        ch_top = gui.crosshairs[1]
        assert ch_top.hhair.y[0] == 11886
        assert ch_top.vhair.x[0] == pd.Timestamp("2023-01-10 10:15")
        assert gui._html_output._html.value
        gui._html_output.children[1].click()  # close html output
        assert not gui._html_output._html.value
        gui.tabs_control.v_model = 0
        gui.tabs_control.but_trash.fire_event("click", None)
        assert not gui.crosshairs
        # verify max_dec
        gui.tabs_control.v_model = 1
        assert len(gui.crosshairs) == 0
        gui.tabs_control.but_arrow_down.fire_event("click", None)
        assert gui._html_output._html.value
        assert len(gui.crosshairs) == 2
        ch_top = gui.crosshairs[0]
        assert ch_top.hhair.y[0] == 11808
        assert ch_top.vhair.x[0] == pd.Timestamp("2023-01-06 16:10")
        ch_bot = gui.crosshairs[1]
        assert ch_bot.hhair.y[0] == 11602
        assert ch_bot.vhair.x[0] == pd.Timestamp("2023-01-09 15:05")

        # verify effect of `log_scale` and `max_ticks` kwargs
        assert type(gui.chart.scales["y"]) == bq.scales.LogScale
        assert gui._icon_row_top.children[-1].tooltip == "Close"  # close gui
        gui._icon_row_top.children[-1].click()
        gui = analy.plot(**intraday_pp, display=False, log_scale=False, max_ticks=50)
        assert type(gui.chart.scales["y"]) == bq.scales.LinearScale
        expected = pd.Interval(
            pd.Timestamp("2023-01-10 12:05"), pd.Timestamp("2023-01-10 16:15"), "left"
        )
        assert gui.chart.plotted_interval == expected
        gui._icon_row_top.children[-1].click()  # close gui

        # instantiate new gui covering wider date range
        gui = analy.plot(start="2021-12-30", end="2023-01-05", display=False)
        labels = [label for label, pt in gui._interval_selector.options]
        expected = ["3M", "1M", "5D", "1D", "4H", "1H", "30T", "15T", "5T", "2T", "1T"]
        assert labels == expected

        gui._interval_selector.value = mp.intervals.TDInterval.D5
        gui._interval_selector.value = mp.intervals.DOInterval.M1
        gui._interval_selector.value = mp.intervals.DOInterval.M3

        expected = pd.Interval(
            pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected

        gui.date_slider.slider.value = (
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2022-10-01"),
        )

        # verify raises advices when can not display a shorter interval due to lack of data availability
        gui._icon_row_top.children[1].click()  # reset chart
        expected = pd.Interval(
            pd.Timestamp("2021-12-30"), pd.Timestamp("2023-01-06"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected
        slider_tup = (pd.Timestamp("2021-12-30"), pd.Timestamp("2023-01-05"))
        assert gui.date_slider.slider.value == slider_tup

        assert not gui._dialog.value
        gui._interval_selector.value = mp.intervals.TDInterval.T30
        assert gui._dialog.value
        assert (
            gui._dialog.text
            == "Prices for interval '30T' are only available from '2022-12-05' although the earliest date that can be plotted on the chart implies that require data from '2021-12-30'."
        )
        gui._dialog.close_dialog()
        assert not gui._dialog.value
        # verify values unchanged
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected
        assert gui.date_slider.slider.value == slider_tup

        gui._interval_selector.value = mp.intervals.TDInterval.T1
        assert gui._dialog.value
        assert (
            gui._dialog.text
            == "Prices for interval '1T' are only available from '2023-01-03' although the earliest date that can be plotted on the chart implies that require data from '2021-12-30'."
        )
        gui._dialog.close_dialog()
        assert not gui._dialog.value
        # verify values unchanged
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected
        assert gui.date_slider.slider.value == slider_tup

    @pytest.fixture(scope="class")
    def trend_kwargs(self) -> abc.Iterator[dict]:
        """Value for 'trend_kwargs' argument of trend methods."""
        yield {"prd": 60, "ext_break": 0.04, "ext_limit": 0.02}

    @pytest.fixture(scope="class")
    def trend_period_kwargs(self) -> abc.Iterator[dict]:
        """Value for **kwargs of trend methods."""
        yield {"start": "2018-01-02", "end": "2022-12-30"}

    def test_movements(self, path_res, analy, trend_kwargs, trend_period_kwargs):
        """Test `Analysis.movements`.

        Verifies return as expected, with expected verified visually.
        """
        movements = analy.movements("1D", trend_kwargs, **trend_period_kwargs)
        path = path_res / "azn_1D_prd60.dat"
        file = open(path, "rb")
        for m in movements.moves:
            try:
                loaded = pickle.load(file)
            except EOFError:
                break
            assert m == loaded
        file.close()

    def test_trends_chart(self, analy, trend_kwargs, trend_period_kwargs):
        """Verifies various aspects of gui beahviour for trends plot."""
        f = analy.trends_chart
        interval = "1D"
        kwargs = trend_period_kwargs
        verify_app(f, trends.TrendsGui, interval, trend_kwargs, **kwargs)

        gui = f(interval, trend_kwargs, **kwargs, display=False)
        movements = analy.movements("1D", trend_kwargs, **kwargs)
        for gm, m in zip(gui.movements.moves, movements.moves):
            assert gm == m

        prd = trend_kwargs["prd"]
        ext_break = trend_kwargs["ext_break"]
        ext_limit = trend_kwargs["ext_limit"]

        # verify scatter marks
        scats = gui.chart.mark_scatters
        assert len(scats) == 8
        for s in scats:
            assert isinstance(s, bq.Scatter)
            assert s.visible

        def get_scat(marker: str, cols: list[str]):
            return next((s for s in scats if (s.marker, s.colors) == (marker, cols)))

        cols_adv, cols_dec = ["lime"], ["red"]

        scat = get_scat("triangle-down", cols_dec)
        assert (scat.x == movements.starts_dec).all()
        for y, move in zip(scat.y, movements.declines):
            assert y == move.start_px
        scat = get_scat("triangle-up", cols_adv)
        assert (scat.x == movements.starts_adv).all()
        for y, move in zip(scat.y, movements.advances):
            assert y == move.start_px

        scat = get_scat("cross", cols_dec)
        assert (scat.x == movements.ends_dec_solo).all()
        for y, end in zip(scat.y, movements.ends_dec_solo):
            move = (m for m in movements.declines if m.end == end).__next__()
            assert y == move.end_px
        scat = get_scat("cross", cols_adv)
        assert (scat.x == movements.ends_adv_solo).all()
        for y, end in zip(scat.y, movements.ends_adv_solo):
            move = (m for m in movements.advances if m.end == end).__next__()
            assert y == move.end_px

        scat = get_scat("circle", cols_dec)
        assert (scat.x == movements.starts_conf_dec).all()
        assert (scat.y == movements.starts_conf_dec_px).all()
        scat = get_scat("circle", cols_adv)
        assert (scat.x == movements.starts_conf_adv).all()
        assert (scat.y == movements.starts_conf_adv_px).all()

        scat = get_scat("square", cols_dec)
        assert (scat.x == movements.ends_conf_dec).all()
        assert (scat.y == movements.ends_conf_dec_px).all()
        scat = get_scat("square", cols_adv)
        assert (scat.x == movements.ends_conf_adv).all()
        assert (scat.y == movements.ends_conf_adv_px).all()

        # verify trend line
        apply_map = {0: "yellow", 1: "lime", -1: "red"}
        expected = movements.trend.apply(lambda v: apply_map[v]).to_list()
        assert expected == gui.chart.mark_trend.colors

        # verify initial plot and slider ranges
        start = pd.Timestamp(kwargs["start"])
        end = pd.Timestamp(kwargs["end"])
        assert gui.date_slider.slider.value == (start, end)
        one_day = pd.Timedelta(1, "D")
        assert gui.chart.plotted_interval == pd.Interval(start, end + one_day, "left")

        controls = gui.trends_controls_container
        # verify initial gui config as expected
        assert controls.is_dark_single_trend
        assert controls.but_show_all.is_light
        assert not gui._html_output._html.value
        assert gui.chart.current_move is None
        gui.current_move is None

        def verify_controls_reflect_single_trend():
            assert not controls.is_dark_single_trend
            assert not controls.but_show_all.is_light

        def assert_trend_reflects_move(move: trends.Movement):
            # verify gui as requried
            verify_controls_reflect_single_trend()
            assert (
                f"Start: {move.start.strftime('%Y-%m-%d')}"
                in gui._html_output._html.value
            )
            assert not any([s.visible for s in scats])
            assert gui.chart.current_move == move == gui.current_move

            # verify trend marks
            group = gui.chart.added_marks_groups[0]
            trend_marks = gui.chart.added_marks(group)
            trend_scat_marks = [m for m in trend_marks if isinstance(m, bq.Scatter)]
            assert len(trend_scat_marks) == 4

            def get_trend_scat(marker: str):
                return (m for m in trend_scat_marks if m.marker == marker).__next__()

            marker = "triangle-up" if move.is_adv else "triangle-down"
            scat = get_trend_scat(marker)
            assert (scat.x == [move.start]).all()
            assert (scat.y == [move.start_px]).all()

            scat = get_trend_scat("cross")
            assert (scat.x == [move.end]).all()
            assert (scat.y == [move.end_px]).all()

            scat = get_trend_scat("circle")
            assert (scat.x == [move.start_conf]).all()
            assert (scat.y == [move.start_conf_px]).all()

            scat = get_trend_scat("square")
            assert (scat.x == [move.end_conf]).all()
            assert (scat.y == [move.end_conf_px]).all()

            # verify conf change rectangle
            rec = (m for m in trend_marks if m.fill == "between").__next__()
            assert isinstance(rec, bq.Lines)
            if (move.is_adv and move.start_conf_px < move.end_conf_px) or (
                not move.is_adv and move.start_conf_px > move.end_conf_px
            ):
                fill_col = cols_adv
            else:
                fill_col = cols_dec
            assert rec.fill_colors == fill_col
            expected = [move.start_conf, move.end_conf]
            assert (rec.x == [expected, expected]).all()
            expected = [
                [move.start_conf_px, move.start_conf_px],
                [move.end_conf_px, move.end_conf_px],
            ]
            assert (rec.y == expected).all()

            break_clr = ["white"] if move.by_break else ["slategray"]
            limit_clr = ["white"] if not move.by_break else ["slategray"]

            line = (m for m in trend_marks if m.colors == break_clr).__next__()
            assert isinstance(line, bq.Lines)
            assert line.line_style == "dashed"
            assert (line.x == move.line_break.index).all()
            assert (line.y == move.line_break).all()

            line = (m for m in trend_marks if m.colors == limit_clr).__next__()
            assert isinstance(line, bq.Lines)
            assert line.line_style == "dashed"
            assert (line.x == move.line_limit.index).all()
            assert (line.y == move.line_limit).all()

        # verify marks for various single trend and control buttons to progress through trends
        gui.chart._click_starts_adv(0)
        assert_trend_reflects_move(gui.movements.moves[1])
        controls.but_next.fire_event("click", None)
        assert_trend_reflects_move(gui.movements.moves[2])
        controls.but_next.fire_event("click", None)
        controls.but_next.fire_event("click", None)
        controls.but_prev.fire_event("click", None)
        move = gui.movements.moves[3]
        assert_trend_reflects_move(move)

        # verify wide zoom on trend
        controls.but_wide.fire_event("click", None)
        verify_controls_reflect_single_trend()
        index = gui.trends.data.index
        i = index.get_loc(move.start)
        left = index[i - prd * 3]
        i = index.get_loc(move.end_conf)
        right = index[i + prd * 3]
        assert gui.date_slider.slider.value == (left, right)
        assert gui.chart.plotted_interval == pd.Interval(left, right + one_day, "left")

        # verify narrow zoom on trend
        controls.but_narrow.fire_event("click", None)
        verify_controls_reflect_single_trend()
        i = index.get_loc(move.start)
        left = index[i - prd]
        i = index.get_loc(move.end_conf)
        right = index[i + prd]
        assert gui.date_slider.slider.value == (left, right)
        assert gui.chart.plotted_interval == pd.Interval(left, right + one_day, "left")

        assert not gui._rulers
        but = controls.but_ruler
        assert but.is_light
        but.fire_event("click", None)

        assert not but.is_light and not but.is_dark
        verify_controls_reflect_single_trend()
        assert len(gui._rulers) == 2

        # verify break_rule
        def verify_break_rule(rule: bqu.TrendRule, move: trends.Movement):
            for c in rule.components:
                assert c.visible
                assert c in gui.chart.figure.marks
            assert len(rule.line.x) == prd
            assert rule.line.x[0] == rule.grip_l.x == move.start
            assert rule.line.y[0] == rule.grip_l.y == move.start_px
            assert rule.line.x[-1] == rule.grip_r.x == move.start_conf
            diff = ext_break if move.is_adv else -ext_break
            assert rule.line.y[-1] == rule.grip_r.y == move.start_px * (1 + diff)
            try:
                assert (rule.label_l.text == [move.start.strftime("%b-%d")]).all()
            except AssertionError:
                assert (rule.label_l.text == [move.start.strftime("%y-%b-%d")]).all()
            try:
                assert (rule.label_r.text == [move.start_conf.strftime("%b-%d")]).all()
            except AssertionError:
                assert (
                    rule.label_r.text == [move.start_conf.strftime("%y-%b-%d")]
                ).all()

        break_rule = (r for r in gui._rulers if r.color == ["orange"]).__next__()
        verify_break_rule(break_rule, move)

        # verify limit_rule
        def verify_limit_rule(rule: bqu.TrendRule, move: trends.Movement):
            for c in rule.components:
                assert c.visible
                assert c in gui.chart.figure.marks
            assert len(rule.line.x) == prd
            assert rule.line.x[0] == rule.grip_l.x == move.end
            assert rule.line.y[0] == rule.grip_l.y == move.end_px
            i = index.get_loc(move.end)
            right = index[i + prd - 1]
            assert rule.line.x[-1] == rule.grip_r.x == right
            diff = ext_limit if move.is_adv else -ext_limit
            assert rule.line.y[-1] == rule.grip_r.y == move.end_px * (1 + diff)
            try:
                assert (rule.label_l.text == [move.end.strftime("%b-%d")]).all()
            except AssertionError:
                assert (rule.label_l.text == [move.end.strftime("%y-%b-%d")]).all()
            try:
                assert (rule.label_r.text == [right.strftime("%b-%d")]).all()
            except AssertionError:
                assert (rule.label_r.text == [right.strftime("%y-%b-%d")]).all()

        limit_rule = (r for r in gui._rulers if r.color == ["skyblue"]).__next__()
        verify_limit_rule(limit_rule, move)

        controls.but_prev.fire_event("click", None)

        # changing trend should not change rulers...
        assert limit_rule in gui._rulers and break_rule in gui._rulers
        # although but should go light
        assert but.is_light
        verify_controls_reflect_single_trend()

        # verify clicking rulers places new rulers
        but.fire_event("click", None)
        controls.but_wide.fire_event("click", None)  # make sure can see all on plot

        move = movements.moves[2]
        assert limit_rule not in gui._rulers and break_rule not in gui._rulers

        assert not but.is_light and not but.is_dark
        verify_controls_reflect_single_trend()
        assert len(gui._rulers) == 2

        # verify break_rule
        break_rule = (r for r in gui._rulers if r.color == ["orange"]).__next__()
        verify_break_rule(break_rule, move)
        # verify limit_rule
        limit_rule = (r for r in gui._rulers if r.color == ["skyblue"]).__next__()
        verify_limit_rule(limit_rule, move)

        assert not but.is_light and not but.is_dark
        but.fire_event("click", None)

        assert not gui._rulers
        for c in break_rule.components:
            assert c not in gui.chart.figure.marks
        for c in limit_rule.components:
            assert c not in gui.chart.figure.marks

        # verify show_all but shows all scatters again
        controls.but_show_all.fire_event("click", None)
        assert controls.is_dark_single_trend
        assert controls.but_show_all.is_light
        assert not gui.chart.added_marks_groups
        assert all([s.visible for s in scats])
        assert gui.chart.current_move is None
        assert gui.current_move is None


class TestCompare:
    """Tests for the `analysis.Compare` class."""

    @pytest.fixture(scope="class")
    def analy(self, prices_compare) -> abc.Iterator[analysis.Compare]:
        yield analysis.Compare(prices_compare)

    def test_constructor_raises(self):
        msg = re.escape(
            "The Compare class requires a `prices` instance that gets"
            "price data for multiple symbols, although the past instance"
            "gets prices for only one: MSFT."
        )
        with pytest.raises(ValueError, match=msg):
            analysis.Compare(mp.PricesYahoo("MSFT"))

    def test_prices(self, analy, prices_compare):
        assert analy.prices is prices_compare

    def test_symbols(self, analy):
        assert analy.symbols == ["MSFT", "AZN.L", "9988.HK"]

    def test_prices_daily(self, analy, daily_pp, intraday_pp):
        expected = pd.DataFrame(
            {
                ("MSFT", "open"): {
                    T("2023-01-06 00:00:00"): 223.0,
                    T("2023-01-09 00:00:00"): 226.4499969482422,
                    T("2023-01-10 00:00:00"): 227.75999450683594,
                },
                ("MSFT", "high"): {
                    T("2023-01-06 00:00:00"): 225.75999450683594,
                    T("2023-01-09 00:00:00"): 231.24000549316406,
                    T("2023-01-10 00:00:00"): 231.30999755859375,
                },
                ("MSFT", "low"): {
                    T("2023-01-06 00:00:00"): 219.35000610351562,
                    T("2023-01-09 00:00:00"): 226.41000366210938,
                    T("2023-01-10 00:00:00"): 227.3300018310547,
                },
                ("MSFT", "close"): {
                    T("2023-01-06 00:00:00"): 224.92999267578125,
                    T("2023-01-09 00:00:00"): 227.1199951171875,
                    T("2023-01-10 00:00:00"): 228.85000610351562,
                },
                ("MSFT", "volume"): {
                    T("2023-01-06 00:00:00"): 43597700.0,
                    T("2023-01-09 00:00:00"): 27369800.0,
                    T("2023-01-10 00:00:00"): 27033900.0,
                },
                ("AZN.L", "open"): {
                    T("2023-01-06 00:00:00"): 11734.0,
                    T("2023-01-09 00:00:00"): 11694.0,
                    T("2023-01-10 00:00:00"): 11714.0,
                },
                ("AZN.L", "high"): {
                    T("2023-01-06 00:00:00"): 11808.0,
                    T("2023-01-09 00:00:00"): 11758.0,
                    T("2023-01-10 00:00:00"): 11886.0,
                },
                ("AZN.L", "low"): {
                    T("2023-01-06 00:00:00"): 11690.0,
                    T("2023-01-09 00:00:00"): 11602.0,
                    T("2023-01-10 00:00:00"): 11666.0,
                },
                ("AZN.L", "close"): {
                    T("2023-01-06 00:00:00"): 11782.0,
                    T("2023-01-09 00:00:00"): 11736.0,
                    T("2023-01-10 00:00:00"): 11802.0,
                },
                ("AZN.L", "volume"): {
                    T("2023-01-06 00:00:00"): 4153691.0,
                    T("2023-01-09 00:00:00"): 1989665.0,
                    T("2023-01-10 00:00:00"): 3318361.0,
                },
                ("9988.HK", "open"): {
                    T("2023-01-06 00:00:00"): 101.5,
                    T("2023-01-09 00:00:00"): 105.5999984741211,
                    T("2023-01-10 00:00:00"): 109.0,
                },
                ("9988.HK", "high"): {
                    T("2023-01-06 00:00:00"): 102.69999694824219,
                    T("2023-01-09 00:00:00"): 110.9000015258789,
                    T("2023-01-10 00:00:00"): 110.0,
                },
                ("9988.HK", "low"): {
                    T("2023-01-06 00:00:00"): 99.69999694824219,
                    T("2023-01-09 00:00:00"): 105.0,
                    T("2023-01-10 00:00:00"): 107.0,
                },
                ("9988.HK", "close"): {
                    T("2023-01-06 00:00:00"): 101.5999984741211,
                    T("2023-01-09 00:00:00"): 110.4000015258789,
                    T("2023-01-10 00:00:00"): 109.5,
                },
                ("9988.HK", "volume"): {
                    T("2023-01-06 00:00:00"): 73315739.0,
                    T("2023-01-09 00:00:00"): 123720147.0,
                    T("2023-01-10 00:00:00"): 76284815.0,
                },
            }
        )
        expected.columns.names = ["symbol", ""]
        rtrn = analy.daily_prices(**daily_pp)
        assert_frame_equal(rtrn, expected, check_freq=None)

        expected = pd.DataFrame(
            {
                ("MSFT", "open"): {T("2023-01-09 00:00:00"): 226.4499969482422},
                ("MSFT", "high"): {T("2023-01-09 00:00:00"): 231.24000549316406},
                ("MSFT", "low"): {T("2023-01-09 00:00:00"): 226.41000366210938},
                ("MSFT", "close"): {T("2023-01-09 00:00:00"): 227.1199951171875},
                ("MSFT", "volume"): {T("2023-01-09 00:00:00"): 27369800.0},
                ("AZN.L", "open"): {T("2023-01-09 00:00:00"): 11694.0},
                ("AZN.L", "high"): {T("2023-01-09 00:00:00"): 11758.0},
                ("AZN.L", "low"): {T("2023-01-09 00:00:00"): 11602.0},
                ("AZN.L", "close"): {T("2023-01-09 00:00:00"): 11736.0},
                ("AZN.L", "volume"): {T("2023-01-09 00:00:00"): 1989665.0},
                ("9988.HK", "open"): {T("2023-01-09 00:00:00"): 105.5999984741211},
                ("9988.HK", "high"): {T("2023-01-09 00:00:00"): 110.9000015258789},
                ("9988.HK", "low"): {T("2023-01-09 00:00:00"): 105.0},
                ("9988.HK", "close"): {T("2023-01-09 00:00:00"): 110.4000015258789},
                ("9988.HK", "volume"): {T("2023-01-09 00:00:00"): 123720147.0},
            }
        )
        expected.columns.names = ["symbol", ""]
        rtrn = analy.daily_prices(**intraday_pp)
        assert_frame_equal(rtrn, expected, check_freq=None)

    def test_daily_close_prices(self, analy, daily_pp, intraday_pp):
        rtrn = analy.daily_close_prices(**daily_pp)
        expected = pd.DataFrame(
            {
                "MSFT": {
                    T("2023-01-06 00:00:00"): 224.92999267578125,
                    T("2023-01-09 00:00:00"): 227.1199951171875,
                    T("2023-01-10 00:00:00"): 228.85000610351562,
                },
                "AZN.L": {
                    T("2023-01-06 00:00:00"): 11782.0,
                    T("2023-01-09 00:00:00"): 11736.0,
                    T("2023-01-10 00:00:00"): 11802.0,
                },
                "9988.HK": {
                    T("2023-01-06 00:00:00"): 101.5999984741211,
                    T("2023-01-09 00:00:00"): 110.4000015258789,
                    T("2023-01-10 00:00:00"): 109.5,
                },
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected, check_freq=False)

        rtrn = analy.daily_close_prices(**intraday_pp)
        expected = pd.DataFrame(
            {
                "MSFT": {T("2023-01-09 00:00:00"): 227.1199951171875},
                "AZN.L": {T("2023-01-09 00:00:00"): 11736.0},
                "9988.HK": {T("2023-01-09 00:00:00"): 110.4000015258789},
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected, check_freq=False)

    def test_price_chg(self, analy, daily_pp, intraday_pp):
        f = analy.price_chg
        # test for daily_pp
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "chg": {
                    "MSFT": 6.540008544921875,
                    "AZN.L": 92.0,
                    "9988.HK": 9.900001525878906,
                }
            },
            index=pd.Index(["MSFT", "AZN.L", "9988.HK"]),
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Price Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lr}\n & chg \\\\\nsymbol &  \\\\\nMSFT & \\colorYellowGreen 6.54 \\\\\nAZN.L & \\colorYellowGreen 92.00 \\\\\n9988.HK & \\colorYellowGreen 9.90 \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**daily_pp)
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "chg": {
                    "MSFT": 9.218597412109375,
                    "AZN.L": 72.3203125,
                    "9988.HK": 3.8000030517578125,
                }
            },
            index=pd.Index(["MSFT", "AZN.L", "9988.HK"]),
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Price Change 2023-01-06 09:45 to 2023-01-10 11:15}\n\\begin{tabular}{lr}\n & chg \\\\\nsymbol &  \\\\\nMSFT & \\colorYellowGreen 9.22 \\\\\nAZN.L & \\colorYellowGreen 72.32 \\\\\n9988.HK & \\colorYellowGreen 3.80 \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**intraday_pp)
        assert rtrn.to_latex() == expected

    def test_pct_chg(self, analy, daily_pp, intraday_pp):
        # test for daily_pp
        f = analy.pct_chg

        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "MSFT": 0.029418418500041232,
                    "AZN.L": 0.007856532877882128,
                    "9988.HK": 0.09939760720429325,
                }
            },
            index=pd.Index(["MSFT", "AZN.L", "9988.HK"]),
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Percentage Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lr}\n & pct_chg \\\\\nsymbol &  \\\\\nMSFT & \\colorYellowGreen 2.94% \\\\\nAZN.L & \\colorYellowGreen 0.79% \\\\\n9988.HK & \\colorYellowGreen 9.94% \\\\\nAv. & \\colorYellowGreen 4.56% \\\\\n\\end{tabular}\n\\end{table}\n"

        rtrn = f(**daily_pp)
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "MSFT": 0.04191986384700219,
                    "AZN.L": 0.006164363493010638,
                    "9988.HK": 0.03595083407257382,
                }
            },
            index=pd.Index(["MSFT", "AZN.L", "9988.HK"]),
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        expected = "\\begin{table}\n\\caption{Percentage Change 2023-01-06 09:45 to 2023-01-10 11:15}\n\\begin{tabular}{lr}\n & pct_chg \\\\\nsymbol &  \\\\\nMSFT & \\colorYellowGreen 4.19% \\\\\nAZN.L & \\colorYellowGreen 0.62% \\\\\n9988.HK & \\colorYellowGreen 3.60% \\\\\nAv. & \\colorYellowGreen 2.80% \\\\\n\\end{tabular}\n\\end{table}\n"
        rtrn = f(**intraday_pp)

    def test_summary_chg(self, analy):
        # test default return
        rtrn = analy.summary_chg(style=False)
        expected = pd.DataFrame(
            {
                "5T": {
                    "MSFT": -0.0006788524670320317,
                    "AZN.L": -0.001366120218579181,
                    "9988.HK": np.nan,
                },
                "15T": {
                    "MSFT": -0.0020339140958298696,
                    "AZN.L": -0.003505355404089583,
                    "9988.HK": np.nan,
                },
                "1H": {
                    "MSFT": 0.013196175483992656,
                    "AZN.L": -0.009293320425943885,
                    "9988.HK": -0.021543999039545514,
                },
                "4H": {
                    "MSFT": 0.04200879505195765,
                    "AZN.L": -0.009293320425943885,
                    "9988.HK": -0.021543999039545514,
                },
                "1D": {
                    "MSFT": 0.029396608014960357,
                    "AZN.L": -0.008171206225680905,
                    "9988.HK": -0.009090909090909038,
                },
                "2D": {
                    "MSFT": 0.04991725611983289,
                    "AZN.L": -0.03720491029272899,
                    "9988.HK": 0.013011166781899286,
                },
                "5D": {
                    "MSFT": 0.04911287369266626,
                    "AZN.L": -0.042089440060127825,
                    "9988.HK": -0.07234042553191489,
                },
                "2W": {
                    "MSFT": 0.1218039964304709,
                    "AZN.L": -0.10718038528896667,
                    "9988.HK": -0.028520472685202858,
                },
                "1M": {
                    "MSFT": 0.08489694241413859,
                    "AZN.L": -0.0911035835264753,
                    "9988.HK": 0.2637681159420291,
                },
                "3M": {
                    "MSFT": 0.18209897983108436,
                    "AZN.L": -0.03079847908745248,
                    "9988.HK": 0.6172106458803026,
                },
                "6M": {
                    "MSFT": -0.05327128396138181,
                    "AZN.L": -0.05417439703153992,
                    "9988.HK": 0.25215389062297056,
                },
                "1Y": {
                    "MSFT": -0.16997384110160807,
                    "AZN.L": 0.18806804940573296,
                    "9988.HK": -0.09166666666666667,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = analy.summary_chg()
        expected = "\\begin{table}\n\\caption{Percentage Change}\n\\begin{tabular}{lrrrrrrrrrrrr}\n & 5T & 15T & 1H & 4H & 1D & 2D & 5D & 2W & 1M & 3M & 6M & 1Y \\\\\nMSFT & \\colorCrimson -0.07% & \\colorCrimson -0.20% & \\colorYellowGreen 1.32% & \\colorYellowGreen 4.20% & \\colorYellowGreen 2.94% & \\colorYellowGreen 4.99% & \\colorYellowGreen 4.91% & \\colorYellowGreen 12.2% & \\colorYellowGreen 8.49% & \\colorYellowGreen 18.2% & \\colorCrimson -5.33% & \\colorCrimson -17.0% \\\\\nAZN.L & \\colorCrimson -0.14% & \\colorCrimson -0.35% & \\colorCrimson -0.93% & \\colorCrimson -0.93% & \\colorCrimson -0.82% & \\colorCrimson -3.72% & \\colorCrimson -4.21% & \\colorCrimson -10.7% & \\colorCrimson -9.11% & \\colorCrimson -3.08% & \\colorCrimson -5.42% & \\colorYellowGreen 18.8% \\\\\n9988.HK & \\colorYellowGreen nan% & \\colorYellowGreen nan% & \\colorCrimson -2.15% & \\colorCrimson -2.15% & \\colorCrimson -0.91% & \\colorYellowGreen 1.30% & \\colorCrimson -7.23% & \\colorCrimson -2.85% & \\colorYellowGreen 26.4% & \\colorYellowGreen 61.7% & \\colorYellowGreen 25.2% & \\colorCrimson -9.17% \\\\\nAv. & \\colorCrimson -0.10% & \\colorCrimson -0.28% & \\colorCrimson -0.59% & \\colorYellowGreen 0.37% & \\colorYellowGreen 0.40% & \\colorYellowGreen 0.86% & \\colorCrimson -2.18% & \\colorCrimson -0.46% & \\colorYellowGreen 8.59% & \\colorYellowGreen 25.6% & \\colorYellowGreen 4.82% & \\colorCrimson -2.45% \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

        # # test specific periods
        periods = [dict(hours=4), dict(weeks=2), dict(minutes=15)]
        rtrn = analy.summary_chg(periods=periods, style=False)
        expected = pd.DataFrame(
            {
                "4H": {
                    "MSFT": 0.04200879505195765,
                    "AZN.L": -0.009293320425943885,
                    "9988.HK": -0.021543999039545514,
                },
                "2W": {
                    "MSFT": 0.1218039964304709,
                    "AZN.L": -0.10718038528896667,
                    "9988.HK": -0.028520472685202858,
                },
                "15T": {
                    "MSFT": -0.0020339140958298696,
                    "AZN.L": -0.003505355404089583,
                    "9988.HK": np.nan,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

    def test_chg(self, analy, intraday_pp, daily_pp):
        # test for daily_pp
        rtrn = analy.chg(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "MSFT": 0.029418418500041232,
                    "AZN.L": 0.007856532877882128,
                    "9988.HK": 0.09939760720429325,
                },
                "chg": {
                    "MSFT": 6.540008544921875,
                    "AZN.L": 92.0,
                    "9988.HK": 9.900001525878906,
                },
                "close": {
                    "MSFT": 228.85000610351562,
                    "AZN.L": 11802.0,
                    "9988.HK": 109.5,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = analy.chg(style=True, **daily_pp)
        expected = "\\begin{table}\n\\caption{Change 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lrrr}\n & pct_chg & chg & close \\\\\nsymbol &  &  &  \\\\\nMSFT & \\colorYellowGreen 2.94% & \\colorYellowGreen 6.54 & 228.85 \\\\\nAZN.L & \\colorYellowGreen 0.79% & \\colorYellowGreen 92.00 & 11802.00 \\\\\n9988.HK & \\colorYellowGreen 9.94% & \\colorYellowGreen 9.90 & 109.50 \\\\\nAv. & \\colorYellowGreen 4.56% & \\colorYellowGreen  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

        # test for intraday_pp
        rtrn = analy.chg(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "MSFT": 0.04191986384700219,
                    "AZN.L": 0.006164363493010638,
                    "9988.HK": 0.03595083407257382,
                },
                "chg": {
                    "MSFT": 9.218597412109375,
                    "AZN.L": 72.3203125,
                    "9988.HK": 3.8000030517578125,
                },
                "close": {
                    "MSFT": 229.12860107421875,
                    "AZN.L": 11804.3203125,
                    "9988.HK": 109.5,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = analy.chg(style=True, **intraday_pp)
        expected = "\\begin{table}\n\\caption{Change 2023-01-06 09:45 to 2023-01-10 11:15}\n\\begin{tabular}{lrrr}\n & pct_chg & chg & close \\\\\nsymbol &  &  &  \\\\\nMSFT & \\colorYellowGreen 4.19% & \\colorYellowGreen 9.22 & 229.13 \\\\\nAZN.L & \\colorYellowGreen 0.62% & \\colorYellowGreen 72.32 & 11804.32 \\\\\n9988.HK & \\colorYellowGreen 3.60% & \\colorYellowGreen 3.80 & 109.50 \\\\\nAv. & \\colorYellowGreen 2.80% & \\colorYellowGreen  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_chg_every_interval(self, analy, intraday_pp):
        f = analy.chg_every_interval
        rtrn = f("90T", style=False, **intraday_pp, anchor="workback")
        # verify against expected first three rows
        expected = pd.DataFrame(
            {
                "MSFT": {
                    pd.Interval(
                        T("2023-01-05 21:45:00", tz="America/New_York"),
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        T("2023-01-06 03:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                },
                "AZN.L": {
                    pd.Interval(
                        T("2023-01-05 21:45:00", tz="America/New_York"),
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        T("2023-01-06 03:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0018748934719618204,
                },
                "9988.HK": {
                    pd.Interval(
                        T("2023-01-05 21:45:00", tz="America/New_York"),
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        closed="left",
                    ): -0.0029586099333661934,
                    pd.Interval(
                        T("2023-01-06 00:15:00", tz="America/New_York"),
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0029673892807684393,
                    pd.Interval(
                        T("2023-01-06 01:45:00", tz="America/New_York"),
                        T("2023-01-06 03:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.001972356461860063,
                },
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn[:3], expected)

        # verify against expected last three rows
        expected = pd.DataFrame(
            {
                "MSFT": {
                    pd.Interval(
                        T("2023-01-10 06:45:00", tz="America/New_York"),
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.011094009862088488,
                    pd.Interval(
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        T("2023-01-10 11:15:00", tz="America/New_York"),
                        closed="left",
                    ): -0.0023572826187263285,
                },
                "AZN.L": {
                    pd.Interval(
                        T("2023-01-10 06:45:00", tz="America/New_York"),
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        closed="left",
                    ): -0.001694053870913095,
                    pd.Interval(
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0035635499745460715,
                    pd.Interval(
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        T("2023-01-10 11:15:00", tz="America/New_York"),
                        closed="left",
                    ): -0.002002002663172134,
                },
                "9988.HK": {
                    pd.Interval(
                        T("2023-01-10 06:45:00", tz="America/New_York"),
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-10 08:15:00", tz="America/New_York"),
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                    pd.Interval(
                        T("2023-01-10 09:45:00", tz="America/New_York"),
                        T("2023-01-10 11:15:00", tz="America/New_York"),
                        closed="left",
                    ): 0.0,
                },
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn[-3:], expected)

        rtrn = f("90T", style=True, **intraday_pp, anchor="workback")
        expected = "\\begin{table}\n\\caption{Change over prior 90T.   Period: 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC}\n\\begin{tabular}{lrrr}\nindex & MSFT & AZN.L & 9988.HK \\\\\n[2023-01-05 21:45:00, 2023-01-06 00:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorCrimson -0.30% \\\\\n[2023-01-06 00:15:00, 2023-01-06 01:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.30% \\\\\n[2023-01-06 01:45:00, 2023-01-06 03:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.19% & \\colorYellowGreen 0.20% \\\\\n[2023-01-06 03:15:00, 2023-01-06 04:45:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.26% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 04:45:00, 2023-01-06 06:15:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.06% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 06:15:00, 2023-01-06 07:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.19% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 07:45:00, 2023-01-06 09:15:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.07% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 09:15:00, 2023-01-06 10:45:00) & \\colorYellowGreen 0.04% & \\colorCrimson -0.07% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 10:45:00, 2023-01-06 12:15:00) & \\colorYellowGreen 0.22% & \\colorYellowGreen 0.60% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 12:15:00, 2023-01-06 13:45:00) & \\colorYellowGreen 0.42% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 13:45:00, 2023-01-06 15:15:00) & \\colorYellowGreen 0.79% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-06 15:15:00, 2023-01-08 21:15:00) & \\colorCrimson -0.30% & \\colorYellowGreen 0.00% & \\colorYellowGreen 7.87% \\\\\n[2023-01-08 21:15:00, 2023-01-08 22:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-08 22:45:00, 2023-01-09 01:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorCrimson -0.46% \\\\\n[2023-01-09 01:15:00, 2023-01-09 02:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.18% \\\\\n[2023-01-09 02:45:00, 2023-01-09 04:15:00) & \\colorYellowGreen 0.00% & \\colorCrimson -1.02% & \\colorYellowGreen 1.01% \\\\\n[2023-01-09 04:15:00, 2023-01-09 05:45:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.17% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 05:45:00, 2023-01-09 07:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 07:15:00, 2023-01-09 08:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.12% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 08:45:00, 2023-01-09 10:15:00) & \\colorYellowGreen 1.52% & \\colorCrimson -0.14% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 10:15:00, 2023-01-09 11:45:00) & \\colorYellowGreen 0.96% & \\colorYellowGreen 0.70% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 11:45:00, 2023-01-09 13:15:00) & \\colorYellowGreen 0.15% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 13:15:00, 2023-01-09 14:45:00) & \\colorCrimson -0.81% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% \\\\\n[2023-01-09 14:45:00, 2023-01-09 20:45:00) & \\colorCrimson -0.79% & \\colorYellowGreen 0.00% & \\colorCrimson -2.08% \\\\\n[2023-01-09 20:45:00, 2023-01-09 22:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.46% \\\\\n[2023-01-09 22:15:00, 2023-01-10 00:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorYellowGreen 1.29% \\\\\n[2023-01-10 00:45:00, 2023-01-10 02:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.00% & \\colorCrimson -0.45% \\\\\n[2023-01-10 02:15:00, 2023-01-10 03:45:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 0.06% & \\colorYellowGreen 0.00% \\\\\n[2023-01-10 03:45:00, 2023-01-10 05:15:00) & \\colorYellowGreen 0.00% & \\colorYellowGreen 1.13% & \\colorYellowGreen 0.00% \\\\\n[2023-01-10 05:15:00, 2023-01-10 06:45:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.59% & \\colorYellowGreen 0.00% \\\\\n[2023-01-10 06:45:00, 2023-01-10 08:15:00) & \\colorYellowGreen 0.00% & \\colorCrimson -0.17% & \\colorYellowGreen 0.00% \\\\\n[2023-01-10 08:15:00, 2023-01-10 09:45:00) & \\colorYellowGreen 1.11% & \\colorYellowGreen 0.36% & \\colorYellowGreen 0.00% \\\\\n[2023-01-10 09:45:00, 2023-01-10 11:15:00) & \\colorCrimson -0.24% & \\colorCrimson -0.20% & \\colorYellowGreen 0.00% \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_chg_every_interval_chart(self, analy, intraday_pp):
        f = analy.chg_every_interval

        pp = {"start": "2022-01-01", "years": 1}
        verify_app(f, guis.PctChg, "10d", chart=True, **pp)

        gui = analy.chg_every_interval("10d", chart=True, **pp, _display=False)
        assert gui.chart.title == "Change over prior 10d.   Period: 1Y from 2022-01-01"
        assert gui.chart.axes[0].orientation == "horizontal"
        assert gui.chart.axes[1].orientation == "vertical"
        assert gui.chart.axes[1].orientation == "vertical"
        assert gui.chart.mark.type == "stacked"

        start, end = pd.Timestamp("2022-01-04"), pd.Timestamp("2022-12-31")
        assert start == gui.chart.data.index[0].left
        assert end == gui.chart.data.index[-1].right

        expected = pd.Interval(pd.Timestamp("2022-08-10"), end, "left")
        assert gui.chart.plotted_interval == expected
        expected_plottable = pd.Interval(start, end, "left")
        assert gui.chart.plottable_interval == expected_plottable
        slider_end = pd.Timestamp("2022-12-16")
        assert gui.date_slider.slider.value == (pd.Timestamp("2022-08-10"), slider_end)

        # verify max dates
        assert gui._icon_row_top.children[0].tooltip == "Max dates"
        gui._icon_row_top.children[0].click()
        assert gui.date_slider.slider.value == (start, slider_end)

        # verify can change slider
        slider_dates = (pd.Timestamp("2022-04-14"), pd.Timestamp("2022-07-13"))
        gui.date_slider.slider.value = slider_dates
        expected = pd.Interval(
            pd.Timestamp("2022-04-14"), pd.Timestamp("2022-07-27"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        # verify effect of change chart type
        gui._icon_row.bar_type_tog[1].fire_event("click", None)
        assert gui.chart.mark.type == "grouped"
        assert gui.date_slider.slider.value == slider_dates
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        gui._icon_row.bar_type_tog[0].fire_event("click", None)
        assert gui.chart.mark.type == "stacked"
        assert gui.date_slider.slider.value == slider_dates
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        # verify effect of clicking legend cycle button
        assert gui.chart.figure.legend_location == "top-right"
        for _ in range(4):
            gui._icon_row.legend_cycle_but.fire_event("click", None)
        assert gui.chart.figure.legend_location == "bottom-left"

        assert gui._icon_row_top.children[-1].tooltip == "Close"
        gui._icon_row_top.children[-1].click()

        # verify for horizontal bars and intraday interval
        gui = analy.chg_every_interval(
            "21T", **intraday_pp, chart=True, direction="horizontal", _display=False
        )
        assert (
            gui.chart.title
            == "Change over prior 21T.   Period: 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC"
        )
        assert gui.chart.axes[0].orientation == "vertical"
        assert gui.chart.axes[1].orientation == "horizontal"
        assert gui.chart.axes[1].orientation == "horizontal"

        start, end = pd.Timestamp("2023-01-06 09:51:00"), pd.Timestamp(
            "2023-01-10 11:15:00"
        )
        assert start == gui.chart.data.index[0].left
        assert end == gui.chart.data.index[-1].right

        expected = pd.Interval(pd.Timestamp("2023-01-10 07:45"), end, "left")
        assert gui.chart.plotted_interval == expected
        expected_plottable = pd.Interval(start, end, "left")
        assert gui.chart.plottable_interval == expected_plottable
        slider_end = pd.Timestamp("2023-01-10 10:54")
        assert gui.date_slider.slider.value == (
            pd.Timestamp("2023-01-10 07:45"),
            slider_end,
        )

        # verify max dates
        assert gui._icon_row_top.children[0].tooltip == "Max dates"
        gui._icon_row_top.children[0].click()
        assert gui.date_slider.slider.value == (start, slider_end)

        # verify can change slider
        slider_datetimes = (
            pd.Timestamp("2023-01-09 04:57"),
            pd.Timestamp("2023-01-09 13:21"),
        )
        gui.date_slider.slider.value = slider_datetimes
        expected = pd.Interval(
            pd.Timestamp("2023-01-09 04:57"), pd.Timestamp("2023-01-09 13:42"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        # verify effect of change chart type
        gui._icon_row.bar_type_tog[1].fire_event("click", None)
        assert gui.chart.mark.type == "grouped"
        assert gui.date_slider.slider.value == slider_datetimes
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        gui._icon_row.bar_type_tog[0].fire_event("click", None)
        assert gui.chart.mark.type == "stacked"
        assert gui.date_slider.slider.value == slider_datetimes
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable

        assert gui._icon_row_top.children[-1].tooltip == "Close"
        gui._icon_row_top.children[-1].click()

    def test_price_on(self, analy):
        """Test `Compare.price_on`.

        Also tests:
            Passing date to `Compare.price`.
            `Compare.today` property.
        """
        f = analy.price_on
        # test default
        rtrn = f(style=False)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "9988.HK": -0.009090909090909038,
                    "AZN.L": -0.008171206225680905,
                    "MSFT": 0.029396608014960357,
                },
                "chg": {"9988.HK": -1.0, "AZN.L": -84.0, "MSFT": 7.42999267578125},
                "close": {
                    "9988.HK": 109.0,
                    "AZN.L": 10196.0,
                    "MSFT": 260.17999267578125,
                },
                "open": {
                    "9988.HK": 111.4000015258789,
                    "AZN.L": 10330.0,
                    "MSFT": 258.82000732421875,
                },
                "high": {
                    "9988.HK": 112.30000305175781,
                    "AZN.L": 10342.0,
                    "MSFT": 260.7900085449219,
                },
                "low": {
                    "9988.HK": 108.69999694824219,
                    "AZN.L": 10184.91796875,
                    "MSFT": 257.25,
                },
                "volume": {
                    "9988.HK": 39931077.0,
                    "AZN.L": 764650.0,
                    "MSFT": 12214074.0,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.price(style=False), expected)

        rtrn = f()
        expected = "\\begin{table}\n\\caption{Price 2023-02-02}\n\\begin{tabular}{lrrrrrrr}\n & pct_chg & chg & close & open & high & low & volume \\\\\nsymbol &  &  &  &  &  &  &  \\\\\n9988.HK & \\colorCrimson -0.91% & \\colorCrimson -1.00 & 109.00 & 111.40 & 112.30 & 108.70 & 39931077 \\\\\nAZN.L & \\colorCrimson -0.82% & \\colorCrimson -84.00 & 10196.00 & 10330.00 & 10342.00 & 10184.92 & 764650 \\\\\nMSFT & \\colorYellowGreen 2.94% & \\colorYellowGreen 7.43 & 260.18 & 258.82 & 260.79 & 257.25 & 12214074 \\\\\nAv. & \\colorYellowGreen 0.40% & \\colorYellowGreen  &  &  &  &  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected
        assert analy.today.to_latex() == expected
        assert analy.price().to_latex() == expected

        # test passing session
        session = "2023-01-05"
        rtrn = f(session, style=False)
        expected = pd.DataFrame(
            {
                "pct_chg": {
                    "9988.HK": 0.03319498856421843,
                    "AZN.L": 0.009308739872435856,
                    "MSFT": -0.02963774929736973,
                },
                "chg": {
                    "9988.HK": 3.1999969482421875,
                    "AZN.L": 108.0,
                    "MSFT": -6.790008544921875,
                },
                "close": {
                    "9988.HK": 99.5999984741211,
                    "AZN.L": 11710.0,
                    "MSFT": 222.30999755859375,
                },
                "open": {
                    "9988.HK": 102.0999984741211,
                    "AZN.L": 11524.0,
                    "MSFT": 227.1999969482422,
                },
                "high": {
                    "9988.HK": 103.80000305175781,
                    "AZN.L": 11730.0,
                    "MSFT": 227.5500030517578,
                },
                "low": {
                    "9988.HK": 98.19999694824219,
                    "AZN.L": 11476.0,
                    "MSFT": 221.75999450683594,
                },
                "volume": {
                    "9988.HK": 104685426.0,
                    "AZN.L": 2215897.0,
                    "MSFT": 39585600.0,
                },
            }
        )
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.price(session, style=False), expected)

        rtrn = f(session)
        expected = "\\begin{table}\n\\caption{Price 2023-01-05}\n\\begin{tabular}{lrrrrrrr}\n & pct_chg & chg & close & open & high & low & volume \\\\\nsymbol &  &  &  &  &  &  &  \\\\\n9988.HK & \\colorYellowGreen 3.32% & \\colorYellowGreen 3.20 & 99.60 & 102.10 & 103.80 & 98.20 & 104685426 \\\\\nAZN.L & \\colorYellowGreen 0.93% & \\colorYellowGreen 108.00 & 11710.00 & 11524.00 & 11730.00 & 11476.00 & 2215897 \\\\\nMSFT & \\colorCrimson -2.96% & \\colorCrimson -6.79 & 222.31 & 227.20 & 227.55 & 221.76 & 39585600 \\\\\nAv. & \\colorYellowGreen 0.43% & \\colorYellowGreen  &  &  &  &  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_price_at(self, analy):
        """Test `Compare.price_at`.

        Also tests:
            Passing time to `Compare.price`.
            `Compare.now` property.
        """
        # test default
        rtrn = analy.price_at()
        ts = T("2023-02-02 10:10:00-0500", tz="America/New_York")
        expected = pd.DataFrame(
            {
                "MSFT": {ts: 260.17999267578125},
                "AZN.L": {ts: 10196.0},
                "9988.HK": {ts: 109.0},
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)
        assert_frame_equal(analy.now, expected)

        # test passing a value
        dt = "2023-01-06 09:33"
        rtrn = analy.price_at(dt)
        ts = T("2023-01-06 09:33:00-0500", tz="America/New_York")
        expected = pd.DataFrame(
            {
                "MSFT": {ts: 222.07000732421875},
                "AZN.L": {ts: 11750.0},
                "9988.HK": {ts: 101.5999984741211},
            }
        )
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)

        # test .price()
        assert_frame_equal(rtrn, analy.price(dt))

    def test_price_range(self, analy, intraday_pp):
        rtrn = analy.price_range(**intraday_pp)
        interval = pd.Interval(
            T("2023-01-06 09:45:00", tz="America/New_York"),
            T("2023-01-10 11:15:00", tz="America/New_York"),
            closed="right",
        )
        expected = pd.DataFrame(
            {
                ("MSFT", "open"): {interval: 219.91000366210938},
                ("MSFT", "high"): {interval: 231.30999755859375},
                ("MSFT", "low"): {interval: 219.35000610351562},
                ("MSFT", "close"): {interval: 229.12860107421875},
                ("MSFT", "volume"): {interval: 64867671.0},
                ("AZN.L", "open"): {interval: 11732.0},
                ("AZN.L", "high"): {interval: 11886.0},
                ("AZN.L", "low"): {interval: 11602.0},
                ("AZN.L", "close"): {interval: 11804.3203125},
                ("AZN.L", "volume"): {interval: 2539545.0},
                ("9988.HK", "open"): {interval: 105.69999694824219},
                ("9988.HK", "high"): {interval: 112.0},
                ("9988.HK", "low"): {interval: 105.0},
                ("9988.HK", "close"): {interval: 109.5},
                ("9988.HK", "volume"): {interval: 175324614.0},
            }
        )
        expected.columns.names = ["symbol", ""]
        assert_frame_equal(rtrn, expected)

    def test_analysis(self, analy, prices_analysis):
        """Test `Compare.analysis`.

        Limited test to make sure returns object of expected class, with
        expected symbol and returns expected data for a single call to a
        single method.
        """
        expected = analysis.Analysis(prices_analysis)  # more or less expected
        symbol = "AZN.L"
        rtrn = analy.analysis(symbol)
        assert isinstance(rtrn, analysis.Analysis)
        assert rtrn.symbol == symbol
        kwargs = {"start": "2022-01-01", "days": 15}
        assert rtrn.chg(**kwargs).to_latex() == expected.chg(**kwargs).to_latex()

    def test_relative_strength(self, analy, daily_pp):
        f = analy.relative_strength
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "MSFT": {
                    "MSFT": 0.0,
                    "AZN.L": -0.025136597124664206,
                    "9988.HK": 0.05014303218052518,
                },
                "AZN.L": {
                    "MSFT": 0.025136597124664206,
                    "AZN.L": 0.0,
                    "9988.HK": 0.07527962930518939,
                },
                "9988.HK": {
                    "MSFT": -0.05014303218052518,
                    "AZN.L": -0.07527962930518939,
                    "9988.HK": 0.0,
                },
            }
        )
        expected.columns.name = "symbol"
        expected.index.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Relative Strength 2023-01-06 to 2023-01-10 (2023-01-05 16:00 to 10 16:00)}\n\\begin{tabular}{lrrrr}\n & MSFT & AZN.L & 9988.HK & Av. \\\\\nsymbol &  &  &  &  \\\\\nMSFT & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#a7d0e4 \\color#000000 2.51% & \\background-color#c94741 \\color#f1f1f1 -5.01% & \\background-color#fae7dc \\color#000000 -0.83% \\\\\nAZN.L & \\background-color#f7b799 \\color#000000 -2.51% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#67001f \\color#f1f1f1 -7.53% & \\background-color#ee9677 \\color#000000 -3.35% \\\\\n9988.HK & \\background-color#3783bb \\color#f1f1f1 5.01% & \\background-color#053061 \\color#f1f1f1 7.53% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#529dc8 \\color#f1f1f1 4.18% \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_max_adv(self, analy, daily_pp):
        f = analy.max_adv
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "start": {
                    "9988.HK": T("2023-01-08 20:30:00-0500", tz="America/New_York"),
                    "AZN.L": T("2023-01-09 10:05:00-0500", tz="America/New_York"),
                    "MSFT": T("2023-01-06 09:50:00-0500", tz="America/New_York"),
                },
                "low": {"9988.HK": 105.0, "AZN.L": 11602.0, "MSFT": 219.35000610351562},
                "end": {
                    "9988.HK": T("2023-01-09 02:55:00-0500", tz="America/New_York"),
                    "AZN.L": T("2023-01-10 05:15:00-0500", tz="America/New_York"),
                    "MSFT": T("2023-01-10 10:15:00-0500", tz="America/New_York"),
                },
                "high": {
                    "9988.HK": 112.0,
                    "AZN.L": 11886.0,
                    "MSFT": 231.30999755859375,
                },
                "pct_chg": {
                    "9988.HK": 0.06666666666666665,
                    "AZN.L": 0.024478538183071885,
                    "MSFT": 0.05452469169038432,
                },
                "days": {"9988.HK": 0, "AZN.L": 0, "MSFT": 4},
                "hours": {"9988.HK": 6.0, "AZN.L": 19.0, "MSFT": 0.0},
                "minutes": {"9988.HK": 25, "AZN.L": 10, "MSFT": 25},
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Maximum Advance 2023-01-06 to 2023-01-10}\n\\begin{tabular}{llrlrrrrr}\n & start & low & end & high & pct_chg & days & hours & minutes \\\\\n9988.HK & 2023-01-08 20:30 & 105.00 & 2023-01-09 02:55 & 112.00 & \\colorYellowGreen 6.67% & 0 & 6 & 25 \\\\\nAZN.L & 2023-01-09 10:05 & 11602.00 & 2023-01-10 05:15 & 11886.00 & \\colorYellowGreen 2.45% & 0 & 19 & 10 \\\\\nMSFT & 2023-01-06 09:50 & 219.35 & 2023-01-10 10:15 & 231.31 & \\colorYellowGreen 5.45% & 4 & 0 & 25 \\\\\nAv. &  &  &  &  & \\colorYellowGreen 4.86% & 1 &  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_max_dec(self, analy, intraday_pp):
        f = analy.max_dec
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "start": {
                    "9988.HK": T("2023-01-09 02:55:00-0500", tz="America/New_York"),
                    "AZN.L": T("2023-01-06 11:10:00-0500", tz="America/New_York"),
                    "MSFT": T("2023-01-09 11:15:00-0500", tz="America/New_York"),
                },
                "high": {
                    "9988.HK": 112.0,
                    "AZN.L": 11808.0,
                    "MSFT": 231.23660278320312,
                },
                "end": {
                    "9988.HK": T("2023-01-09 20:40:00-0500", tz="America/New_York"),
                    "AZN.L": T("2023-01-09 10:05:00-0500", tz="America/New_York"),
                    "MSFT": T("2023-01-09 15:55:00-0500", tz="America/New_York"),
                },
                "low": {"9988.HK": 107.0, "AZN.L": 11602.0, "MSFT": 226.77000427246094},
                "pct_chg": {
                    "9988.HK": -0.044642857142857095,
                    "AZN.L": -0.017445799457994626,
                    "MSFT": -0.019316139646498254,
                },
                "days": {"9988.HK": 0, "AZN.L": 2, "MSFT": 0},
                "hours": {"9988.HK": 17, "AZN.L": 22, "MSFT": 4},
                "minutes": {"9988.HK": 45, "AZN.L": 55, "MSFT": 40},
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**intraday_pp)
        expected = "\\begin{table}\n\\caption{Maximum Decline 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC}\n\\begin{tabular}{llrlrrrrr}\n & start & high & end & low & pct_chg & days & hours & minutes \\\\\n9988.HK & 2023-01-09 02:55 & 112.00 & 2023-01-09 20:40 & 107.00 & \\colorCrimson -4.46% & 0 & 17 & 45 \\\\\nAZN.L & 2023-01-06 11:10 & 11808.00 & 2023-01-09 10:05 & 11602.00 & \\colorCrimson -1.74% & 2 & 22 & 55 \\\\\nMSFT & 2023-01-09 11:15 & 231.24 & 2023-01-09 15:55 & 226.77 & \\colorCrimson -1.93% & 0 & 4 & 40 \\\\\nAv. &  &  &  &  & \\colorCrimson -2.71% & 0 &  &  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_relative_strength_max_adv(self, analy, intraday_pp):
        f = analy.relative_strength_max_adv
        rtrn = f(style=False, **intraday_pp)
        expected = pd.DataFrame(
            {
                "9988.HK": {
                    "9988.HK": 0.0,
                    "AZN.L": -0.04218812848359477,
                    "MSFT": -0.012141974976282333,
                },
                "AZN.L": {
                    "9988.HK": 0.04218812848359477,
                    "AZN.L": 0.0,
                    "MSFT": 0.030046153507312434,
                },
                "MSFT": {
                    "9988.HK": 0.012141974976282333,
                    "AZN.L": -0.030046153507312434,
                    "MSFT": 0.0,
                },
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**intraday_pp)
        expected = "\\begin{table}\n\\caption{Relative Strength Maximum Advance 2023-01-06 14:45 UTC to 2023-01-10 16:15 UTC}\n\\begin{tabular}{lrrrr}\n & 9988.HK & AZN.L & MSFT & Av. \\\\\n9988.HK & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#053061 \\color#f1f1f1 4.22% & \\background-color#b6d7e8 \\color#000000 1.21% & \\background-color#87beda \\color#000000 1.81% \\\\\nAZN.L & \\background-color#67001f \\color#f1f1f1 -4.22% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#c13639 \\color#f1f1f1 -3.00% & \\background-color#da6853 \\color#f1f1f1 -2.41% \\\\\nMSFT & \\background-color#f9c4a9 \\color#000000 -1.21% & \\background-color#2f79b5 \\color#f1f1f1 3.00% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#dbeaf2 \\color#000000 0.60% \\\\\nAv. & \\background-color#f09c7b \\color#000000 -1.81% & \\background-color#4c99c6 \\color#f1f1f1 2.41% & \\background-color#fbe3d4 \\color#000000 -0.60% & \\background-color#000000 \\color#f1f1f1  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_relative_strength_max_dec(self, analy, daily_pp):
        f = analy.relative_strength_max_dec
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "9988.HK": {
                    "9988.HK": 0.0,
                    "AZN.L": 0.02719705768486247,
                    "MSFT": 0.02532671749635884,
                },
                "AZN.L": {
                    "9988.HK": -0.02719705768486247,
                    "AZN.L": 0.0,
                    "MSFT": -0.0018703401885036275,
                },
                "MSFT": {
                    "9988.HK": -0.02532671749635884,
                    "AZN.L": 0.0018703401885036275,
                    "MSFT": 0.0,
                },
            }
        )
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Relative Strength Maximum Decline 2023-01-06 to 2023-01-10}\n\\begin{tabular}{lrrrr}\n & 9988.HK & AZN.L & MSFT & Av. \\\\\n9988.HK & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#67001f \\color#f1f1f1 -2.72% & \\background-color#7f0823 \\color#f1f1f1 -2.53% & \\background-color#ce4f45 \\color#f1f1f1 -1.75% \\\\\nAZN.L & \\background-color#053061 \\color#f1f1f1 2.72% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#eaf1f5 \\color#000000 0.19% & \\background-color#a0cce2 \\color#000000 0.97% \\\\\nMSFT & \\background-color#0e4179 \\color#f1f1f1 2.53% & \\background-color#f9eee7 \\color#000000 -0.19% & \\background-color#f6f7f7 \\color#000000 0.00% & \\background-color#b6d7e8 \\color#000000 0.78% \\\\\nAv. & \\background-color#3b88be \\color#f1f1f1 1.75% & \\background-color#f6b191 \\color#000000 -0.97% & \\background-color#f9c4a9 \\color#000000 -0.78% & \\background-color#000000 \\color#f1f1f1  \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_corr(self, analy, daily_pp):
        f = analy.corr
        rtrn = f(style=False, **daily_pp)
        expected = pd.DataFrame(
            {
                "MSFT": {
                    "MSFT": 1.0,
                    "AZN.L": 0.06537557557924834,
                    "9988.HK": 0.5854013116725509,
                },
                "AZN.L": {
                    "MSFT": 0.06537557557924834,
                    "AZN.L": 1.0,
                    "9988.HK": -0.37270084882795307,
                },
                "9988.HK": {
                    "MSFT": 0.5854013116725509,
                    "AZN.L": -0.37270084882795307,
                    "9988.HK": 1.0,
                },
            }
        )
        expected.index.name = "symbol"
        expected.columns.name = "symbol"
        assert_frame_equal(rtrn, expected)

        rtrn = f(**daily_pp)
        expected = "\\begin{table}\n\\caption{Correlation 2023-01-06 to 2023-01-10 (2023-01-06 09:30 to 2023-01-10 16:00)}\n\\begin{tabular}{lrrrr}\n & MSFT & AZN.L & 9988.HK & Av. \\\\\nsymbol &  &  &  &  \\\\\nMSFT & \\background-color#053061 \\color#f1f1f1 1.00 & \\background-color#eaf1f5 \\color#000000 0.07 & \\background-color#4997c5 \\color#f1f1f1 0.59 & \\background-color#a9d1e5 \\color#000000 0.33 \\\\\nAZN.L & \\background-color#eaf1f5 \\color#000000 0.07 & \\background-color#053061 \\color#f1f1f1 1.00 & \\background-color#f5ac8b \\color#000000 -0.37 & \\background-color#fce2d2 \\color#000000 -0.15 \\\\\n9988.HK & \\background-color#4997c5 \\color#f1f1f1 0.59 & \\background-color#f5ac8b \\color#000000 -0.37 & \\background-color#053061 \\color#f1f1f1 1.00 & \\background-color#e3edf3 \\color#000000 0.11 \\\\\n\\end{tabular}\n\\end{table}\n"
        assert rtrn.to_latex() == expected

    def test_plot_mult(self, analy, intraday_pp):
        """Verifies various aspects of gui beahviour for multiple line plots."""
        f = analy.plot
        verify_app(f, guis.ChartMultLine, **intraday_pp)

        gui = analy.plot(**intraday_pp, display=False)

        # verify initial range
        assert gui._interval_selector.value == mp.intervals.TDInterval.T5
        interval = gui._interval_selector.value
        start, end = intraday_pp["start"], intraday_pp["end"]
        start = start.astimezone("America/New_York").tz_localize(None)
        end = end.astimezone("America/New_York").tz_localize(None)
        expected = expected_plottable = pd.Interval(start, end, "left")
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == (start, end - interval)

        # verify rebasing to left of visible range
        test_rebase_date = pd.Timestamp("2023-01-09 10:25")
        gui.date_slider.slider.value = (test_rebase_date, end - interval)
        assert (gui.chart.prices.iloc[0] == 100).all()
        gui._but_rebase.fire_event("click", None)
        gui.date_slider.slider.value = (start, end - interval)
        assert not (gui.chart.prices.iloc[0] == 100).any()
        assert (gui.chart.prices.loc[test_rebase_date] == 100).all()
        assert gui._icon_row_top.children[0].tooltip == "Max dates"

        gui._icon_row_top.children[0].click()  # click max_dates
        assert gui.date_slider.slider.value == (start, end - interval)
        assert (gui.chart.prices.iloc[0] == 100).all()
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == (start, end - interval)

        # verify zooms and rebases to zoom
        slider_start = pd.Timestamp("2023-01-09 09:55")
        slider_tup = (slider_start, pd.Timestamp("2023-01-09 13:10"))
        gui.date_slider.slider.value = slider_tup
        gui.tabs_control.v_model = 1
        gui.tabs_control.but_zoom.fire_event("click", None)
        expected = pd.Interval(slider_start, pd.Timestamp("2023-01-09 13:15"), "left")
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == slider_tup
        assert (gui.chart.data.loc[slider_start] == 100).all()

        # verify effect of reset button
        gui._interval_selector.value == mp.intervals.TDInterval.T15
        assert gui._icon_row_top.children[1].tooltip == "Reset"
        gui._icon_row_top.children[1].click()
        assert gui._interval_selector.value == mp.intervals.TDInterval.T5
        assert gui.chart.plotted_interval == expected_plottable
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == (start, end - interval)
        assert (gui.chart.prices.iloc[0] == 100).all()

        # verify effect of clicking legend cycle button
        assert gui.chart.figure.legend_location == "top-left"
        for _ in range(3):
            gui._but_legend.fire_event("click", None)
        assert gui.chart.figure.legend_location == "right"

        # verify can call all intraday intervals
        gui._interval_selector.value = mp.intervals.TDInterval.T1
        gui._interval_selector.value = mp.intervals.TDInterval.T2
        gui._interval_selector.value = mp.intervals.TDInterval.T5
        gui._interval_selector.value = mp.intervals.TDInterval.T15
        gui._interval_selector.value = mp.intervals.TDInterval.T30
        gui._interval_selector.value = mp.intervals.TDInterval.H1
        gui._interval_selector.value = mp.intervals.TDInterval.H4

        # verify raises advices when can't change interval due to prices being unavailable
        # for a symbol
        gui._interval_selector.value = (
            mp.intervals.TDInterval.D1
        )  # reduce plottable to single day
        expected = pd.Interval(
            pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-10"), "left"
        )
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected
        assert not gui._dialog.value
        gui._interval_selector.value = mp.intervals.TDInterval.T15
        assert gui._dialog.value
        assert (
            gui._dialog.text
            == "Prices for interval '15T' are not available over the current plottable dates as no price is available over this peroid for the following symbols: ['9988.HK']."
        )
        gui._dialog.close_dialog()
        assert not gui._dialog.value

        # verify max_adv, max_dec and crosshairs
        gui._icon_row_top.children[1].click()  # reset chart
        gui.tabs_control.v_model = 1
        assert not gui._html_output._html.value
        assert len(gui.crosshairs) == 0
        gui.tabs_control.but_arrow_up.fire_event("click", None)
        assert len(gui.crosshairs) == 6
        ch_bot = gui.crosshairs[0]
        assert ch_bot.hhair.y[0] == 99.7975422238282
        assert ch_bot.vhair.x[0] == pd.Timestamp("2023-01-06 09:50")
        ch_top = gui.crosshairs[1]
        assert ch_top.hhair.y[0] == 105.23897244504055
        assert ch_top.vhair.x[0] == pd.Timestamp("2023-01-10 10:15")

        assert gui._html_output._html.value
        gui._html_output.children[1].click()  # close html output
        assert not gui._html_output._html.value
        gui.tabs_control.v_model = 0
        gui.tabs_control.but_trash.fire_event("click", None)
        assert not gui.crosshairs

        # verify max_dec
        gui.tabs_control.v_model = 1
        assert len(gui.crosshairs) == 0
        gui.tabs_control.but_arrow_down.fire_event("click", None)
        assert gui._html_output._html.value
        assert len(gui.crosshairs) == 6
        ch_top = gui.crosshairs[0]
        assert ch_top.hhair.y[0] == 105.205580067597
        assert ch_top.vhair.x[0] == pd.Timestamp("2023-01-09 11:15")
        ch_bot = gui.crosshairs[1]
        assert ch_bot.hhair.y[0] == 103.17341439142045
        assert ch_bot.vhair.x[0] == pd.Timestamp("2023-01-09 15:55")

        # verify effect of `log_scale`, `max_ticks` and 'rebase_on_zoom' kwargs
        assert type(gui.chart.scales["y"]) == bq.scales.LogScale
        assert gui._icon_row_top.children[-1].tooltip == "Close"
        gui._icon_row_top.children[-1].click()  # close gui
        gui = analy.plot(
            **intraday_pp,
            display=False,
            log_scale=False,
            max_ticks=50,
            rebase_on_zoom=False,
        )
        assert type(gui.chart.scales["y"]) == bq.scales.LinearScale
        expected = pd.Interval(
            pd.Timestamp("2023-01-10 07:05"), pd.Timestamp("2023-01-10 11:15"), "left"
        )
        assert gui.chart.plotted_interval == expected
        gui.date_slider.slider.value = slider_tup
        gui.tabs_control.v_model = 1
        gui.tabs_control.but_zoom.fire_event("click", None)
        expected = pd.Interval(slider_start, pd.Timestamp("2023-01-09 13:15"), "left")
        assert gui.chart.plotted_interval == expected
        assert gui.chart.plottable_interval == expected_plottable
        assert gui.date_slider.slider.value == slider_tup
        assert not (gui.chart.data.loc[slider_start] == 100).any()

        gui._icon_row_top.children[-1].click()  # close gui

        # verify raises advices where unable to get price data for requested interval
        gui = analy.plot(start="2022-01-04", display=False)
        plotted_interval = gui.chart.plotted_interval
        slider = gui.date_slider.slider.value
        assert not gui._dialog.value
        gui._interval_selector.value = mp.intervals.TDInterval.H4
        assert gui._dialog.value
        assert gui._dialog.text.startswith(
            "Data is unavailable at a sufficiently low base interval to evaluate prices at interval"
        )
        gui._dialog.close_dialog()
        assert not gui._dialog.value
        # verify date range unchanged by error messages having been raised
        assert gui.chart.plotted_interval == plotted_interval
        assert gui.chart.plottable_interval == plotted_interval
        assert gui.date_slider.slider.value == slider
