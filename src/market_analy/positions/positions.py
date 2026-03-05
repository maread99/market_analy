"""Base classes to represent positions.

`PositionBase` offers a base class to represent a single position
taken in a financial instrument.

`PositionsBase` offers a base class to represent all positions
taken in a single financial instrument over an analysis period.

`ConsolidatedBase` provides a base class to consolidate the
results of positions taken across multiple instruments.
"""

import typing
from collections.abc import Sequence
from dataclasses import dataclass

import bqplot as bq
import pandas as pd

from market_analy.cases import (
    CaseBase,
    CasesBase,
    CasesSupportsChartAnaly,
    CaseSupportsChartAnaly,
)
from market_analy.charts import tooltip_html_style
from market_analy.formatters import (
    formatter_datetime,
    formatter_float,
    formatter_percent,
)


@dataclass(frozen=True, eq=False)
class PositionBase(CaseBase, CaseSupportsChartAnaly):
    """Base for classes defining a position.

    Attributes
    ----------
    data
        OHLC prices.

        Index should represent all bars over which the position
        was open, inclusive of both the bar during which it was
        opened and the bar during which it was closed.

        The OHLC columns should represent:

            'open' - price as at bar open, except for bar during
            which position opened, for which should be price at
            which position was opened.

            'high'/'low' - price as at bar high/low except on
            bars when position opened or closed, in which case
            should not include any information outside of the
            period during which the position was held. For the
            bar when position was opened the 'high'/'low' should
            be bar high/low following the position's open. For
            the bar when position was closed the 'high'/'low'
            should be bar high/low as at the time the position
            was closed. If it is not possible to evaluate either
            value then the value should be defined as 'np.NaN'.
            NOTE: 'high' for the bar when position is closed is
            ASSUMED as that bar's high. Without this assumption
            the `max_bar` and `max_px` properties would be
            otherwise inaccurate whenever the maximum price was
            registered prior to a reversal on the bar the
            position was closed. With this assumption the
            `max_bar` and `max_px` properties will be inaccurate
            whenever following a position being closed the price
            moves back to and extends the high before the end of
            the bar.

            'close' - price as at bar close, except of bar
            during which position closed, in which should should
            be price at which position was closed.

    closed
        True - position closed as at end of available data.
        False - position remained open as at end of available
        data.

    spread
        Difference between mid and bid or offer. As float
        representing percentage, for example 0.002 for 0.2%.
    """

    data: pd.DataFrame
    closed: bool
    spread: float

    @property
    def bars(self) -> pd.DatetimeIndex:
        """Bars over which position was held.

        Inclusive of bar during which position opened and bar
        during which position closed.
        """
        index = self.data.index
        index = typing.cast("pd.DatetimeIndex", index)
        return index  # noqa: RET504

    @property
    def open_bar(self) -> pd.Timestamp:
        """Bar when position opened."""
        return self.bars[0]

    @property
    def close_bar(self) -> pd.Timestamp:
        """Bar when position closed.

        If position not closed then most recent bar of analysis
        period.
        """
        return self.bars[-1]

    @property
    def open_px(self) -> float:
        """Mid-price when position opened."""
        return self.data.iloc[0].open

    @property
    def close_px(self) -> float:
        """Mid-price when position closed.

        If position not closed then most recent price of
        analysis period.
        """
        return self.data.iloc[-1].close

    @property
    def chg_abs(self) -> float:
        """Absolute difference between open and close price.

        If position not closed then absolute difference between
        open price and most recent close price from available
        data.

        NOTE Does NOT account for spread.
        """
        return self.close_px - self.open_px

    @property
    def rtrn_gross(self) -> float:
        """Return gross of any costs.

        Does not account for spread.
        """
        return self.chg_abs / self.open_px

    @property
    def rtrn_net(self) -> float:
        """Return net of spread.

        Does not account for any costs other than spread.
        """
        return self.rtrn_gross - (2 * self.spread)

    @property
    def profitable(self) -> bool:
        """Query if profitable on a net basis.

        NOTE: Only spread is considered within costs.
        """
        return self.rtrn_net > 0

    @property
    def max_bar(self) -> pd.Timestamp:
        """Bar when position reached highest registered price."""
        return self.bars[self.data.high == self.max_px][0]

    @property
    def max_px(self) -> float:
        """Highest mid-price registered whilst position held."""
        return self.data.high.max()

    @property
    def chg_abs_max(self) -> float:
        """Absolute difference between open and maximum price.

        NOTE Does NOT account for spread.
        """
        return self.max_px - self.open_px

    @property
    def rtrn_gross_max(self) -> float:
        """Maximum gross return registered during position."""
        return self.chg_abs_max / self.open_px

    @property
    def rtrn_net_max(self) -> float:
        """Maximum return registered, net of spread."""
        return self.rtrn_gross_max - (2 * self.spread)

    @property
    def duration(self) -> int:
        """Position duration, in bars, part or full."""
        return len(self.data)

    @property
    def line(self) -> pd.Series:
        """Line representing position over time.

        Returns
        -------
        pd.Series
            Index represents bars. Value (dtype float)
            represents: For bar during which position was opened,
            the price the position was opened at and the price as
            at the end of the bar. The two prices are provided by
            way of consecutive rows with the same index value,
            with the first representing the price at which the
            position was opened. For bar during which position
            was closed, the price the position was closed at. For
            all other bars, the price as at the end of that bar.
        """
        srs = self.data["close"].copy()
        srs.name = "price"
        first_row = pd.Series([self.open_px], index=srs.index[:1], name="price")
        return pd.concat((first_row, srs))

    @property
    def _start(self) -> pd.Timestamp:
        """Bar when case considered to start."""
        return self.open_bar

    @property
    def _end(self) -> pd.Timestamp | None:
        """Bar when case considered to have concluded.

        None if case had not concluded as at end of available
        data.
        """
        return self.close_bar


@dataclass(frozen=True)
class PositionsBase(CasesBase, CasesSupportsChartAnaly):
    """All positions over an analysis period.

    Attributes
    ----------
    cases
        Ordered sequence of all positions that would have been
        taken during analysis period defined by `data`.

    data
        OHLC data in which `cases` identified.

    ticker
        Ticker with which the analysis corresponds, for example
        "MSFT".
    """

    cases: Sequence[PositionBase]
    data: pd.DataFrame
    ticker: str

    @property
    def lines(self) -> list[pd.Series]:
        """Lines representing positions over time."""
        return [pos.line for pos in self.cases]

    @property
    def open_bars(self) -> list[pd.Timestamp]:
        """Open bars of all positions."""
        return [pos.open_bar for pos in self.cases]

    @property
    def open_pxs(self) -> list[float]:
        """Open prices for all positions."""
        return [pos.open_px for pos in self.cases]

    @property
    def close_bars(self) -> list[pd.Timestamp]:
        """Close bars of all positions."""
        return [pos.close_bar for pos in self.cases]

    @property
    def close_pxs(self) -> list[float]:
        """Close prices for all positions."""
        return [pos.close_px for pos in self.cases]

    @property
    def max_bars(self) -> list[pd.Timestamp]:
        """Bars when positions registered highest prices."""
        return [pos.max_bar for pos in self.cases]

    @property
    def max_pxs(self) -> list[float]:
        """Highest prices registered by positions."""
        return [pos.max_px for pos in self.cases]

    @property
    def rtrns_gross(self) -> list[float]:
        """Positions gross returns."""
        return [pos.rtrn_gross for pos in self.cases]

    @property
    def rtrns_net(self) -> list[float]:
        """Positions net returns."""
        return [pos.rtrn_net for pos in self.cases]

    @property
    def profitables(self) -> list[bool]:
        """Query if positions were profitable."""
        return [pos.profitable for pos in self.cases]

    @property
    def rtrn_gross(self) -> float:
        """Aggregated gross return across all positions."""
        return sum(self.rtrns_gross)

    @property
    def rtrn_net(self) -> float:
        """Aggregated net return across all positions."""
        return sum(self.rtrns_net)

    @property
    def profitable(self) -> bool:
        """Query if positions profitable in aggregate (net).

        NOTE: Only spread is considered within costs.
        """
        return self.rtrn_net > 0

    @staticmethod
    def get_case_html(case: CaseSupportsChartAnaly) -> str:
        """Return html to describe a position.

        Parameters
        ----------
        case
            Case for which to get html.
        """
        case = typing.cast("PositionBase", case)
        color = "limegreen" if case.profitable else "crimson"
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>Open: " + formatter_datetime(case.open_bar)
        s += f"<br>Open px: {formatter_float(case.open_px)}"
        close = "None" if not case.closed else formatter_datetime(case.close_bar)
        s += f"<br>Close: {close}"
        close_px = "None" if not case.closed else formatter_float(case.close_px)
        s += f"<br>Close px: {close_px}"
        s += f"<br>Chg: {formatter_float(case.chg_abs)}"
        s += f"<br>Rtrn (net): {formatter_percent(case.rtrn_net)}"
        s += f"<br>Duration: {case.duration}"
        s += f"<br>Max bar: {formatter_datetime(case.max_bar)}"
        s += f"<br>Max px: {formatter_float(case.max_px)}"
        s += f"<br>Max Rtrn (net): {formatter_percent(case.rtrn_net_max)}"
        return s

    def __repr__(self) -> str:
        return (
            f"<{self.__class__} {self.ticker}"
            f" {round(self.rtrn_net * 100, 1)}"
            f" {round(self.rtrn_gross * 100, 1)}"
        )


class ConsolidatedBase:
    """Summary of positions across instruments over an analysis period.

    Attributes
    ----------
    results
        `PositionsBase` for each instrument over the analysis
        period.
    """

    def __init__(self, results: Sequence[PositionsBase]):
        self.results = {res.ticker: res for res in results}

        tickers = [res.ticker for res in results]
        net = [res.rtrn_net for res in results]
        gross = [res.rtrn_gross for res in results]

        df = pd.DataFrame({"net": net, "gross": gross}, index=tickers)
        df = df.sort_values(by="net", ascending=False)

        df["net_pct"] = (df["net"] * 100).round(1)
        df["gross_pct"] = (df["gross"] * 100).round(1)

        self._rtrns = df

    @property
    def num_symbols(self) -> int:
        """Number of instruments in consolidated analysis."""
        return len(self.results)

    @property
    def rtrns(self) -> pd.DataFrame:
        """Returns."""
        return self._rtrns[["net", "gross"]]

    @property
    def rtrns_pct(self) -> pd.DataFrame:
        """Returns as percent change."""
        df = self._rtrns[["net_pct", "gross_pct"]]
        df.columns = pd.Index(["net", "gross"])
        return df

    @property
    def profitables_pct(self) -> float:
        """Percentage of instruments that were net profitable."""
        return round(
            100 * (self._rtrns["net"] > 0).sum() / self.num_symbols,
            1,
        )

    @property
    def losses_pct(self) -> float:
        """Percentage of instruments resulting in net losses."""
        return round(
            100 * (self._rtrns["net"] < 0).sum() / self.num_symbols,
            1,
        )

    @property
    def rtrn_net(self) -> float:
        """Aggregated net return."""
        return self.rtrns["net"].sum()

    @property
    def rtrn_gross(self) -> float:
        """Aggregated gross return."""
        return self.rtrns["gross"].sum()

    @property
    def rtrn_gross_pct(self) -> float:
        """Aggregated gross return, as percentage."""
        return round(self.rtrn_gross * 100, 1)

    @property
    def rtrn_net_pct(self) -> float:
        """Aggregated net return, as percentage."""
        return round(self.rtrn_net * 100, 1)

    @property
    def profitable(self) -> bool:
        """Query if profitable in aggregate (on net basis).

        NOTE: Only spread is considered within costs.
        """
        return self.rtrn_net > 0

    def distributed_rtrns(self) -> bq.Figure:  # type: ignore[return-value]
        """Plot distributed returns."""
        from .charts import (  # noqa: PLC0415
            distributed_rtrns as chart_distributed_rtrns,
        )

        return chart_distributed_rtrns(self.rtrns_pct.net)
