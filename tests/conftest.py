"""Common pytest fixtures and test utility functions."""

import pathlib
import pickle
from collections import abc

import exchange_calendars as xcals
import market_prices as mp
import pandas as pd
import pytest

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
    now = pd.Timestamp.now().floor("min")
    margin = pd.Timedelta(hours=1, minutes=1)
    now_required = now - margin

    prices.request_all_prices()

    with path.open("wb") as file:
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
            if bi == pd.Timedelta(
                "1D"
            ):  # again to prevent trying to get the live indice
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


CAL_START = "1986-03-13"
XLON = xcals.get_calendar("XLON", start=CAL_START)
CAL_MAPPING = {
    "AZN.L": XLON,
    "BARC.L": XLON,
    "MSFT": xcals.get_calendar("XNYS", start=CAL_START),
    "9988.HK": xcals.get_calendar("XHKG", start=CAL_START),
}
DELAY_MAPPING = {"AZN.L": 15, "BARC.L": 15, "MSFT": 0, "9988.HK": 15}
# Date and time (UTC) of each symbol's first trade, in the form that
# `mp.PricesYahoo` reads from `yahooquery.Ticker.quote_type`.
FIRST_TRADE_MAPPING = {
    "AZN.L": "1993-05-21 07:00:00",
    "BARC.L": "1988-07-01 07:00:00",
    "MSFT": "1986-03-13 14:30:00",
    "9988.HK": "2019-11-26 01:30:00",
}


class _TickerStub:
    """Stand-in for `yahooquery.Ticker` that makes no request to the Yahoo API.

    The constructor of `mp.PricesYahoo` creates a `yahooquery.Ticker`
    instance. Creating a `yahooquery.Ticker` instance makes requests to set up a
    session, to get a crumb and to validate the symbols. If `calendars` and
    `delays` are passed for every symbol, the constructor of `mp.PricesYahoo` then
    makes one further request, via the `quote_type` property, for the date of each
    symbol's first trade.

    This stub provides only the attributes that the constructor of `mp.PricesYahoo`
    reads if `calendars` and `delays` are passed for every symbol. Every symbol is
    taken as valid. The `quote_type` property returns the date of each symbol's
    first trade from `FIRST_TRADE_MAPPING`.
    """

    def __init__(self, symbols: list[str], **_kwargs):
        self.symbols = symbols
        self.invalid_symbols: list[str] = []

    @property
    def quote_type(self) -> dict[str, dict[str, str]]:
        return {
            s: {"firstTradeDateEpochUtc": FIRST_TRADE_MAPPING[s]} for s in self.symbols
        }


@pytest.fixture
def mock_ticker(mocker) -> abc.Iterator[None]:
    """Patch `yahooquery.Ticker` with a stub that makes no request.

    See `_TickerStub`.
    """
    mocker.patch("yahooquery.Ticker", _TickerStub)
    yield


def _unpickle_prices(path: pathlib.Path, class_mocker) -> mp.PricesYahoo:
    """Reconstruct an instance of `PricesYahoo` to use in testing.

    The instance is reconstructed without network access (see `_TickerStub`).
    """
    class_mocker.patch("yahooquery.Ticker", _TickerStub)
    with path.open("rb") as file:
        now_required = pickle.load(file)

        def _mock_now(*args, **kwargs) -> pd.Timestamp:
            return pd.Timestamp(now_required, *args, **kwargs)

        class_mocker.patch("pandas.Timestamp.now", _mock_now)

        # reconstruct each of the data instances
        symbols = pickle.load(file)
        calendars = {s: CAL_MAPPING[s] for s in symbols}
        delays = {s: DELAY_MAPPING[s] for s in symbols}
        prices = mp.PricesYahoo(symbols, calendars=calendars, delays=delays)

        for bi in prices.bis:
            data = prices._pdata[bi]
            data._ranges, data._ll, data._rl, data._table = pickle.load(file)
            if data._table is not None:
                # Requried as pickled with previous version of pandas when Index was
                # inferred as having 'object' dtype. NB if test suit run with pandas
                # 2.3.3 then columns will be inferred as 'object' regardless of
                # astype("str") here
                df = data._table
                arrays = [
                    df.columns.get_level_values(i).astype("str")
                    for i in range(df.columns.nlevels)
                ]
                data._table.columns = pd.MultiIndex.from_arrays(arrays)

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
