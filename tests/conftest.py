"""Common pytest fixtures and test utility functions."""

from __future__ import annotations

from collections import abc
import pathlib
import pickle

import exchange_calendars as xcals
import pandas as pd
import pytest
import market_prices as mp

_RESOURCES_DIR = pathlib.Path(__file__).parent.joinpath("resources")
_PATH_ANALYSIS = _RESOURCES_DIR / "prices_analysis"
_PATH_ANALYSIS_OTHER = _RESOURCES_DIR / "prices_analysis_other"
_PATH_COMPARE = _RESOURCES_DIR / "prices_compare"


def _pickle_new_prices(symbols: str, path: pathlib.Path):
    """Pickle a new instance of `mp.PricesYahoo` with all available price data.

    This function can be used to add a new resource or override an existing
    pickled resource. NOTE overriding an existing resource may result in tests
    that rely on that resource to fail.

    For example:
        `_pickle_new_prices("AMZN", _PATH_ANALYSIS)`
        `_pickle_new_prices("MSFT, AMZN", _PATH_ANALYSIS)`

    Use `_unpickle_prices` to recreate prices instance.

    Notes
    -----
    Not able to just pickle the `mp.PricesBase`, or even the `mp.data.Data`
    instances, due to including BaseInterval enumerations that cannot be
    pickled.
    """
    prices = mp.PricesYahoo(symbols)
    # time adjusted to ensure prices instance doesn't request prices
    now = pd.Timestamp.now().floor("T")
    margin = pd.Timedelta(hours=1, minutes=1)
    now_required = now - margin

    prices.request_all_prices()

    file = open(path, "wb")
    pickle.dump(now_required, file)  # dump time now
    pickle.dump(prices.symbols, file)  # dump symbols

    # dump Data info
    for bi in prices.bis:
        data = prices._pdata[bi]
        # override rl so that it doesn't try to always call the 'live index',
        # rather just assumes it's got all the available data.
        if data._table is None:  # if an interval not synchorised for all symbols
            rl = None
        else:
            index = data._table.index
            rl = index[-1] if bi == pd.Timedelta("1D") else index[-1].right
        ranges = data._ranges
        if bi == pd.Timedelta("1D"):  # again to prevent trying to get the live indice
            rng = ranges[0]
            ranges = [pd.Interval(rng.left, rng.right + pd.Timedelta("1D"), "both")]
        pickle.dump(
            [
                ranges,
                data._ll,
                rl,
                data._table,
            ],
            file,
        )

    file.close()


def _unpickle_prices(path: pathlib.Path, class_mocker) -> mp.PricesYahoo:
    """Reconstruct an instance of `PricesYahoo` to use in testing"""
    file = open(path, "rb")
    now_required = pickle.load(file)

    def _mock_now(*args, **kwargs) -> pd.Timestamp:
        return pd.Timestamp(now_required, *args, **kwargs)

    class_mocker.patch("pandas.Timestamp.now", _mock_now)

    # reconstruct each of the data instances
    symbols = pickle.load(file)
    prices = mp.PricesYahoo(symbols)

    for bi in prices.bis:
        data = prices._pdata[bi]
        data._ranges, data._ll, data._rl, data._table = pickle.load(file)

    file.close()
    return prices


@pytest.fixture(scope="class")
def prices_analysis(class_mocker) -> abc.Iterator[mp.PricesYahoo]:
    """`PricesYahoo` instance to use for testing `Analysis` class."""
    yield _unpickle_prices(_PATH_ANALYSIS, class_mocker)


@pytest.fixture(scope="class")
def prices_compare(class_mocker) -> abc.Iterator[mp.PricesYahoo]:
    """`PricesYahoo` instance to use for testing `Compare` class."""
    yield _unpickle_prices(_PATH_COMPARE, class_mocker)


@pytest.fixture(scope="class")
def prices_analysis_other(class_mocker) -> abc.Iterator[mp.PricesYahoo]:
    """'Another' `PricesYahoo` instance to use for testing `Analysis` class."""
    yield _unpickle_prices(_PATH_ANALYSIS_OTHER, class_mocker)


@pytest.fixture
def path_res() -> abc.Iterator[pathlib.Path]:
    yield _RESOURCES_DIR


@pytest.fixture
def xnys() -> abc.Iterator[xcals.ExchangeCalendar]:
    yield xcals.get_calendar(
        "XNYS", start=pd.Timestamp("2015"), end=pd.Timestamp("2024")
    )
