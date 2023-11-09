"""Tests for trends_alt.py module"""

from collections import abc
import pickle

import pandas as pd
import pytest

from market_analy import trends_alt


# NOTE: testing of `trends_akt.Trends` is LIMITED. See Notes section of
# `trends_alt.Trends` regarding the need for a more comprehensive test suite.


@pytest.fixture
def data_dji_1D_alt(path_res, xnys) -> abc.Iterator[pd.DataFrame]:
    """Daily data for Dow Jones Index.

    Data from call to:
        prices = PricesYahoo("^DJI")
        data = prices.get(
            interval="1D", start="2018", end="2023-04-18", lose_single_symbol=True
        )
    """
    path = path_res / "dji_1D_alt.csv"
    data = pd.read_csv(path, index_col=0)
    data.columns.name = ""
    data.index = pd.DatetimeIndex(data.index, freq=xnys.day)
    yield data


def test_dji_1D(path_res, data_dji_1D_alt):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2018", end "2023-04-18"
        prd 10, ext 0.005, grad None
        rvr [0.8, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4]

    Stored movements confirmed as required by inspection.
    """
    moves = trends_alt.Trends(
        data=data_dji_1D_alt,
        interval="1D",
        prd=10,
        ext=0.005,
        rvr=[0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4],
        grad=None,
        rvr_init=(0.99, 0.8),
        min_bars=3,
    ).get_movements()

    path = path_res / "dji_1D.dat"
    file = open(path, "rb")
    for move in moves.cases:
        try:
            loaded = pickle.load(file)
        except EOFError:
            break
        assert move == loaded
    file.close()
