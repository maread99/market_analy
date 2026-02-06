"""Tests for trends_alt.py module."""

import pickle
from collections import abc

import pandas as pd
import pytest

from market_analy.trends.analy import TrendsAlt

# NOTE: testing of `trends_akt.Trends` is LIMITED. See Notes section of
# `trends_alt.Trends` regarding the need for a more comprehensive test suite.


@pytest.fixture
def data_dji_1D_alt(path_res) -> abc.Iterator[pd.DataFrame]:
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
    data.index = pd.DatetimeIndex(data.index)
    yield data


def test_dji_1D(path_res, data_dji_1D_alt):
    """Tests movements returns as expected.

    For:
        DJI, 1D, start "2018", end "2023-04-18"
        prd 10, ext 0.005, grad None
        rvr [0.8, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4]

    Stored movements confirmed as required by inspection.
    """
    filename_dat = "dji_1D.dat"
    path = path_res / filename_dat
    with path.open("rb") as file:
        move_saved = pickle.load(file)
    # Cannot simply use CustomDay of 'current' XNYS calendar as may have
    # changed since movements saved
    data_dji_1D_alt.index.freq = move_saved.sel.index.freq

    moves = TrendsAlt(
        data=data_dji_1D_alt,
        interval="1D",
        prd=10,
        ext=0.005,
        rvr=[0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4],
        grad=None,
        rvr_init=(0.99, 0.8),
        min_bars=3,
    ).get_movements()

    with path.open("rb") as file:
        for move in moves.cases:
            try:
                loaded = pickle.load(file)
            except EOFError:
                break

            if pd.__version__ >= "3.0.0":
                loaded.sel.index = loaded.sel.index.as_unit("us")
                loaded.start_conf_line.index = loaded.start_conf_line.index.as_unit(
                    "us"
                )
                loaded.end_line_consol.index = loaded.end_line_consol.index.as_unit(
                    "us"
                )
                loaded.end_line_rvr.index = loaded.end_line_rvr.index.as_unit("us")
                if loaded.end_line_rvr_opp is not None:
                    loaded.end_line_rvr_opp.index = (
                        loaded.end_line_rvr_opp.index.as_unit("us")
                    )
                if loaded.eel is not None:
                    loaded.eel.index = loaded.eel.index.as_unit("us")

            assert move == loaded
