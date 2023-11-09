"""Tests for trends.analy.py module"""

from collections import abc
import pathlib
import pickle

import pandas as pd
import pytest

from market_analy.trends import analy as analy_trends


# NOTE: testing of `analy.Trends` is LIMITED. See Notes section of
# `analy.Trends` regarding the need for a more comprehensive test suite.


@pytest.fixture
def data_dji_1D(path_res, xnys) -> abc.Iterator[pd.DataFrame]:
    """Daily data for Dow Jones Index.

    Data from call to:
        prices = PricesYahoo("^DJI")
        data = prices.get(
            interval="1D", start="2018", end="2023-05-24", lose_single_symbol=True
        )
    """
    path = path_res / "dji_1D.csv"
    data = pd.read_csv(path, index_col=0)
    data.columns.name = ""
    data.index = pd.DatetimeIndex(data.index, freq=xnys.day)
    yield data


@pytest.fixture
def data_dji_15T(path_res, xnys) -> abc.Iterator[pd.DataFrame]:
    """Daily data for Dow Jones Index.

    Data from call to:
        prices = PricesYahoo("^DJI")
        data = prices.get(
            interval="15T",
            start="2023-05-01",
            end="2023-05-30",
            lose_single_symbol=True
        )
    """
    path = path_res / "dji_15T.csv"
    data = pd.read_csv(path, index_col=0)
    data.columns.name = ""
    data.index = pd.DatetimeIndex(data.index, name="left", tz=xnys.tz)
    yield data


def assert_moves_as_saved(path: pathlib.Path, moves: list[analy_trends.Movement]):
    file = open(path, "rb")
    for m in moves:
        try:
            loaded = pickle.load(file)
        except EOFError:
            break
        assert m == loaded
    file.close()


def test_dji_1D_prd60(path_res, data_dji_1D):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2018", end "2023-05-24"
        prd 60, ext_break 0.04, ext_limit 0.02

    Stored movements confirmed as required by inspection.
    """
    moves = analy_trends.Trends(
        data=data_dji_1D,
        interval="1D",
        prd=60,
        ext_break=0.04,
        ext_limit=0.02,
    ).get_movements()
    path = path_res / "dji_1D_prd60.dat"
    assert_moves_as_saved(path, moves.cases)


def test_dji_1D_prd15(path_res, data_dji_1D):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2018", end "2023-05-24"
        prd 15, ext_break 0.02, ext_limit 0.01

    Stored movements confirmed as required by inspection.
    """
    moves = analy_trends.Trends(
        data=data_dji_1D,
        interval="1D",
        prd=15,
        ext_break=0.02,
        ext_limit=0.01,
    ).get_movements()
    path = path_res / "dji_1D_prd15.dat"
    assert_moves_as_saved(path, moves.cases)


def test_dji_1D_prd15_minbars10(path_res, data_dji_1D):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2018", end "2023-05-24"
        prd 15, ext_break 0.02, ext_limit 0.01, min_bars 10

    Stored movements confirmed as required by inspection.
    """
    moves = analy_trends.Trends(
        data=data_dji_1D,
        interval="1D",
        prd=15,
        ext_break=0.02,
        ext_limit=0.01,
        min_bars=10,
    ).get_movements()
    path = path_res / "dji_1D_prd15_minbars10.dat"
    assert_moves_as_saved(path, moves.cases)


def test_dji_15T_prd15_minbars10(path_res, data_dji_15T):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2023-05-01", end "2023-05-30"
        prd 15, ext_break 0.002, ext_limit 0.001, min_bars 10

    Stored movements confirmed as required by inspection.
    """
    moves = analy_trends.Trends(
        data=data_dji_15T,
        interval="15T",
        prd=15,
        ext_break=0.002,
        ext_limit=0.001,
        min_bars=10,
    ).get_movements()
    path = path_res / "dji_15T_prd15_minbars10.dat"
    assert_moves_as_saved(path, moves.cases)
