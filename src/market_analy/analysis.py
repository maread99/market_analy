"""Analysis of market instruments.

Analysis is provides for by the methods of the analysis classes `Analysis`
and `Compare`. The `Analysis` class provides for analysis of a single
instrument whilst the `Compare` class provides for analysis of multiple
symbols.

The symbols to be analysed are passed to the analysis class together with,
optionally, a `market_prices` class to use to get price data. By default
this class will be `market_prices.PricesYahoo`.

** Parameters defining analyis period **
The period to be analysed is passed to the methods of the analysis class
as period parameters that will be passed on to the 'get' method of the
`market_prices` class, i.e. by default to `PricesYahoo.get`. Call help on
this underlying method for doc on the available prices parameters, for
example with `help(PricesYahoo.get)`, or directly from an analysis instance
with `help(<analysis_instance>.prices.get)`. The analysis method will also
pass on all other parameters save for those that might be defined by the
method itself (as noted in the method doc).

If period parameters are not defined then the analysis will usually be
undertaken over the longest period for which data is available.

** 'style' argument **
Some analysis methods offer a 'style' which if passed as True (default)
will format the ouput and return as Styler. To return as the underlying
data pass 'style' as False.

Functions
---------
pct_chg_top_to_bot(): -> pd.Series | float:
    Percentage change between first and last value of pandas object.

abs_chg_top_to_bot(): -> pd.Series | float:
    Absolute change between first and last value of pandas object.

add_summary(): -> pd.DataFrame
    Add summary row and/or column (same agg func applied to all rows/cols).

add_summary_row(): -> pd.DataFrame
    Add summary row commposed of different agg funcs.

max_advance() -> pd.DataFrame:
    Maximum percentage advance.

max_decline() -> pd.DataFrame:
    Maximum percentage decline.

formatter_datetime():
    Format value to format suitable for timestamp accuracy.

formatter_percent(x: float):
    Format value as percentage.

formatter_float(x: float):
    Format value as float.

style_df(): -> Styler
    Styler with columns styled according to `FORMATTERS`

style_df_grid(): -> Styler
    Styler in grid style.

Classes
-------
Base(metaclass=ABCMeta):
    Base functionality for analysis of one or more instruments.

Analysis(Base):
    Analyse single instrument.

Compare(Base):
    Analyse multiple insturments.

TestAnalysis(_TestBase):
    Test for Analysis class.

TestCompare(_TestBase):
    Test for Compare class.
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Union

import market_prices as mp
import matplotlib as mpl
import pandas as pd
from exchange_calendars import ExchangeCalendar
from pandas.io.formats.style import Styler

from market_analy import guis
from market_analy.formatters import FORMATTERS, formatter_percent
from market_analy.utils.mkt_prices_utils import (
    period_string,
    range_string,
    request_daily_prices,
)
from market_analy.utils.pandas_utils import (
    rebase_to_row,
    interval_index_new_tz,
    index_dates_to_str,
)

if TYPE_CHECKING:
    from .trends.movements import MovementsSupportChartAnaly
    from .trends import TrendsProto

from .trends.guis import TrendsGui, TrendsGuiBase
from .trends.analy import Trends

Date = Union[str, datetime, None]
Calendar = Union[str, ExchangeCalendar]
Symbols = Union[str, list[str]]
ChartEngine = Literal["bqplot", "matplotlib"]


def pct_chg_top_to_bot(pd_obj: pd.Series | pd.DataFrame) -> pd.Series | float:
    """Return pct change between first and last value of `pd_obj`.

    Returns
    -------
    pd.Series | float
        If `pd_obj` a `pd.Series` then returns float, if `pd.DataFrame`
        then returns `pd.Series` of float values.
    """
    return pd_obj.iloc[-1] / pd_obj.iloc[0] - 1


def abs_chg_top_to_bot(pd_obj: pd.Series | pd.DataFrame) -> pd.Series | float:
    """Return abs change between first and last value of `pd_obj`.

    Returns
    -------
    pd.Series | float
        If `pd_obj` a pd.Series then returns float, if pd.DataFrame returns
        Series of float values.
    """
    return pd_obj.iloc[-1] - pd_obj.iloc[0]


def add_summary(
    df: pd.DataFrame,
    aggmethod: str = "mean",
    axis: Literal["row", "column", "both"] = "row",
    label: str = "SUMMARY",
) -> pd.DataFrame:
    """Add summary row and/or column.

    Parameters
    ----------
    aggmethod
        `pd.DataFrame` method to be used to calcuale summary. Will be
        applied to all rows/columns. NB use `add_summary_row` to create
        a summary row composed of different aggregate methods for different
        columns.

    axis
        Direction summary being applied.

    label
        Label for summary row/colum.
    """
    if axis not in (opts := ["row", "column", "both"]):
        msg = f"`axis` invalid, received as {axis} although must be in {opts}."
        raise ValueError(msg)
    if axis != "column":
        srs = getattr(df, aggmethod)(axis=0)
        srs.name = label
        summary_row = pd.DataFrame(srs).transpose()
    if axis != "row":
        summary_column = getattr(df, aggmethod)(axis=1)
        summary_column.name = label

    if axis == "row":
        return pd.concat((df, summary_row), axis=0)
    elif axis == "column":
        return pd.concat([df, summary_column], axis=1)
    rtrn = pd.concat((df, summary_row), axis=0)
    return pd.concat([rtrn, summary_column], axis=1)


SummaryRowFuncDef = tuple[str, Union[str, list[str]]]
SummaryRowDef = Union[SummaryRowFuncDef, list[SummaryRowFuncDef]]


def add_summary_row(
    df: pd.DataFrame,
    summary: SummaryRowDef = ("mean", "pct_chg"),
    label: str = "SUMMARY",
) -> pd.DataFrame:
    """Add summary row commposed of multiple aggregating functions.

    NB use `add_summary` to add a summary row composed of a single
    aggregating function.

    Parameters
    ----------
    summary
        2-tuple or list of 2-tuples where tuple takes:
            [0] string of the name of an aggregating method that is to
            contribute to populating the summary row.

            [1] string of column name, or list of column names, to which
            aggregating method to be applied.

    label:
        Label for summary row.
    """
    kwargs = {}
    summary = [summary] if isinstance(summary, tuple) else summary
    for s in summary:
        cols = s[1] if isinstance(s[1], list) else [s[1]]
        for col in cols:
            kwargs.update({col: pd.NamedAgg(column=col, aggfunc=s[0])})
    summary_row = df.groupby(by=lambda _: label).agg(**kwargs)
    return pd.concat((df, summary_row))


DURATION_COLUMNS = ["days", "hours", "minutes"]


def _add_duration_columns(d: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    diff = end - start
    for attr in DURATION_COLUMNS:
        v = getattr(diff.components, attr)
        if v > 0:
            d[attr] = [v]
    return d


def max_advance(prices: pd.DataFrame, label: Any = None) -> pd.DataFrame:
    """Return maximum percentage advance.

    Parameters
    ----------
    prices
        Price data covering period within which to return maximum advance.

    label
        Value to assign to index label. Default 0.
    """
    df = prices.loc[:, ["high", "low"]]

    df["max_adv"] = df["high"] / df["low"].cummin() - 1
    max_adv = df["max_adv"].max()

    end = df[df["max_adv"] == max_adv]
    end = end.index[-1]  # if >1 max adv of same value, the most recent

    df = df.loc[:end]
    start = df[df["low"] == df["low"].min()]
    start = start.index[-1]  # if >1 lows of same value, the most recent

    d = {
        "start": [start],
        "low": [df.loc[start, "low"]],
        "end": [end],
        "high": [df.loc[end, "high"]],
        "pct_chg": [max_adv],
    }
    d = _add_duration_columns(d, start, end)
    return pd.DataFrame(d, index=[label])


def max_decline(prices: pd.DataFrame, label: Any = None) -> pd.DataFrame:
    """Return maximum percentage decline.

    Parameters
    ----------
    prices
        Price data covering period within which to return maximum decline.

    label
        Value to assign to index label. Default 0.
    """
    px = prices.loc[:, ["high", "low"]]

    px["max_dec"] = px["low"] / px["high"].cummax() - 1
    max_dec = px["max_dec"].min()

    end = px[px["max_dec"] == max_dec]
    end = end.index[-1]  # if >1 max dec of same value, the most recent

    px = px.loc[:end]
    start = px[px["high"] == px["high"].max()]
    start = start.index[-1]  # if >1 highs of same value, the most recent

    d = {
        "start": [start],
        "high": [px.loc[start, "high"]],
        "end": [end],
        "low": [px.loc[end, "low"]],
        "pct_chg": [max_dec],
    }
    d = _add_duration_columns(d, start, end)
    return pd.DataFrame(d, index=[label])


def _color_chg(value: float | int) -> str:
    """Return colour to represent sign of a value.

    Returns
    -------
        Css property:
            "color: crimson" for negative values.
            "color: yellowgreen" for negative values.
    """
    color = "Crimson" if value < 0 else "YellowGreen"
    return f"color: {color}"


def style_df(
    df: pd.DataFrame,
    na_rep: Any = "",
    chg_cols: Sequence[str] = ("chg", "pct_chg"),
    caption: str | None = None,
) -> Styler:
    """Style `pd.DataFrame` with columns included to `FORMATTERS`."""
    formatter = {k: v for k, v in FORMATTERS.items() if k in df.columns}
    styler = df.style.format(formatter, na_rep=na_rep)

    chg_subset = [c for c in df.columns if c in chg_cols]
    if chg_subset:
        styler.map(_color_chg, subset=chg_subset)

    if caption:
        styler.set_caption(caption)

    return styler


def style_df_grid(
    df: pd.DataFrame,
    format: str | None = "{:.2%}",
    na_rep: Any = "",
    gradient_kwargs: dict | None = {"cmap": "RdBu"},
    caption: str | None = None,
) -> Styler:
    """Style `df` in a grid style."""
    styler = df.style
    if format is not None:
        styler.format(format, na_rep=na_rep)
    if gradient_kwargs is not None:
        styler.background_gradient(axis=None, **gradient_kwargs)
    if caption is not None:
        styler.set_caption(caption)
    return styler


# Defaults
CHG_SUMMARY_PERIODS = [
    {"minutes": 5},
    {"minutes": 15},
    {"hours": 1},
    {"hours": 4},
    {"days": 1},
    {"days": 2},
    {"days": 5},
    {"weeks": 2},
    {"months": 1},
    {"months": 3},
    {"months": 6},
    {"years": 1},
]


# Classes
class Base(metaclass=ABCMeta):
    """Base functionality for analysis of one or more instruments.

    Properties
    ----------
    prices : PricesBase, default: PricesYahoo
        Prices accessor.

    today -> Styler:
        Price row for most recent day.

    now -> pd.DataFrame:
        Most recently registered price.

    Methods
    -------
    daily_prices() -> pd.DataFrame
        Daily open, high, low, close, volume.

    daily_close_prices() -> pd.DataFrame
        Daily close prices.

    pct_chg() -> pd.DataFrame | Styler:
        Percentage change over specific period.

    summary_chg() -> pd.DataFrame:
        Percentage change over multiple periods.

    price_chg() -> pd.Series | float:
        Absolute price change over specified period.

    chg() -> pd.DataFrame | Styler:
        Absolute and percentage price change over specified period.

    chg_every_interval() -> pd.DataFrame | Styler:
        Percent change each interval over specified period.

    price_on() -> pd.DataFrame | Styler:
        Price row for a day. Most recent by default.

    price_at() -> pd.DataFrame:
        Price at a specific datetime.

    price() -> pd.DataFrame:
        Price on a day or at a timestamp.

    price_range() -> pd.DataFrame:
        OHLCV over a period.
    """

    PctChgBarCls = guis.GuiPctChg

    def __init__(self, prices: mp.PricesBase):
        """`analysis.Base` constructor.

        Parameters
        ----------
        prices
            Instance of a subclass of `market_prices.PricesBase` from
            which to request price data for the symbols to be analysed.
            For example:
                prices = market_prices.PricesYahoo("MSFT")
                prices = market_prices.PricesYahoo("MSFT, AZN.L")
        """
        self._prices = prices

    @property
    def prices(self) -> mp.PricesBase:
        """Associated prices object."""
        return self._prices

    def _daily_prices(self, close_only: bool, **kwargs) -> pd.DataFrame:
        kwargs["close_only"] = close_only
        kwargs["interval"] = "1d"
        kwargs.setdefault("lose_single_symbol", True)
        return self.prices.get(**kwargs)

    def daily_prices(self, **kwargs) -> pd.DataFrame:
        """Daily open, high, low, close, volume.

        If multiple symbols, columns indexed with MultiIndex with symbol
        in level 0.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'close_only' or 'interval'.
        """
        return self._daily_prices(close_only=False, **kwargs)

    def daily_close_prices(self, **kwargs) -> pd.DataFrame:
        """Daily close prices.

        Columns indexed with symbol only. Not multiindexed.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'close_only' or 'interval'.
        """
        return self._daily_prices(close_only=True, **kwargs)

    def price_chg(self, style: bool = True, **kwargs) -> pd.Series | float:
        """Absolute price change over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'interval', 'add_a_row', 'close_only' or 'lose_single_symbol'
        """
        interval = "1D" if request_daily_prices(**kwargs) else None
        df = self.prices.get(
            interval=interval,
            add_a_row=True,
            fill="both",
            close_only=True,
            lose_single_symbol=False,
            **kwargs,
        )
        abs_chg = abs_chg_top_to_bot(df)
        res = pd.DataFrame({"chg": abs_chg})
        if style:
            caption = "Price Change " + range_string(df.index, close=True)
            return style_df(res, caption=caption)
        else:
            return res

    def pct_chg(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Percentage change over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'interval', 'add_a_row', 'close_only' or 'lose_single_symbol'.
        """
        interval = "1D" if request_daily_prices(**kwargs) else None
        df = self.prices.get(
            interval=interval,
            add_a_row=True,
            fill="both",
            close_only=True,
            lose_single_symbol=False,
            **kwargs,
        )
        pct_chg = pct_chg_top_to_bot(df)
        res = pd.DataFrame({"pct_chg": pct_chg})
        if style:
            if len(res.index) > 1:
                res = add_summary_row(res, ("mean", "pct_chg"), label="Av.")
            range_str = range_string(df.index, close=True)
            caption = "Percentage Change " + range_str
            return style_df(res, caption=caption)
        else:
            return res

    def summary_chg(
        self, periods: list[dict] | None = None, style: bool = True
    ) -> pd.DataFrame:
        """Percentage change over multiple periods.

        Parameters
        ----------
        periods
            List of dictionaries with each dictionary defining a period
            over which require pct change. Period defined with period
            parameters. See method doc with `help(analysis.__doc__)`.
            Cannot include 'interval', 'add_a_row', 'close_only' or
            'lose_single_symbol'.
        """
        periods = CHG_SUMMARY_PERIODS if periods is None else periods
        dfs = [self.pct_chg(style=False, **period) for period in periods]
        df = pd.concat(dfs, axis=1)
        df.columns = [period_string(**period) for period in periods]
        if style:
            if len(df.index) > 1:
                df = add_summary(df, "mean", axis="row", label="Av.")
            styler = style_df(
                df, chg_cols=list(df.columns), caption="Percentage Change"
            )
            styler.format(formatter_percent)
            return styler
        else:
            return df

    def chg(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Absolute and percentage price change over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'interval', 'add_a_row', 'close_only' or 'lose_single_symbol'.
        """
        interval = "1D" if request_daily_prices(**kwargs) else None
        df = self.prices.get(
            interval=interval,
            add_a_row=True,
            fill="both",
            close_only=True,
            lose_single_symbol=False,
            **kwargs,
        )
        abs_chg = abs_chg_top_to_bot(df)
        pct_chg = pct_chg_top_to_bot(df)
        res = pd.DataFrame({"pct_chg": pct_chg, "chg": abs_chg, "close": df.iloc[-1]})
        if style:
            if len(res.index) > 1:
                res = add_summary_row(res, ("mean", "pct_chg"), label="Av.")
            caption = "Change " + range_string(df.index, close=True)
            return style_df(res, caption=caption)
        return res

    def chg_every_interval(
        self,
        interval: str = "1d",
        style: bool = True,
        chart: bool = False,
        direction: Literal["horizontal", "vertical"] = "vertical",
        _display: bool = True,
        **kwargs,
    ) -> pd.DataFrame | Styler:
        """Percent change every `interval` over specified period.

        Parameters
        ----------
        interval
            Interval over which to evaluate each percentage changes. Passed
            as interval parameter of `self.prices.get`, examples:
                '5T', '15T', '1h', '1d', '5d', '1M'

        style
            True to format ouput and return as Styler.
            Ignored if chart True.

        chart
            True to display as Bar chart.

        direction
            (only relevant is `chart` True)
            Direction of bars.

        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'add_a_row' or 'close_only'.

        _display
            (for testing only)
            Do not display chart when chart is True.
        """
        df = self.prices.get(
            interval,
            close_only=True,
            add_a_row=True,
            fill="both",
            **kwargs,
        )
        shifted = df.shift(1)
        chgs = (df - shifted) / shifted
        chgs = chgs[1:]
        if chart or style:
            caption = (
                "Change over prior "
                + interval
                + ".   Period: "
                + period_string(**kwargs)
            )

        if chart:
            return self.PctChgBarCls(
                data=chgs, title=caption, direction=direction, display=_display
            )

        elif style:
            if isinstance(chgs.index, pd.DatetimeIndex):
                chgs.index = index_dates_to_str(chgs.index)
            else:
                chgs.index = interval_index_new_tz(chgs.index, None)
            symbol_cols = chgs.columns
            styler = style_df(chgs.reset_index(), chg_cols=symbol_cols, caption=caption)
            styler.format({c: formatter_percent for c in symbol_cols})
            styler.hide(axis="index")
            return styler
        return chgs

    def price_on(
        self, session: pd.Timestamp | str | None = None, style: bool = True
    ) -> pd.DataFrame | Styler:
        """Price row for a session.

        Parameters
        ----------
        session: pd.Timestamp, default: Most recent for `.prices` lead symbol.
            Session for which to return price row, for example
            "2023-01-05".

        style: bool, default: True
            True: Format output (returns `Styler`).
            False: Return underlying `pd.DataFrame`.
        """
        if session is None:
            cal = self.prices.calendar_default
            symbol = self.prices.lead_symbol_default
            minute = pd.Timestamp.now() - self.prices.delays[symbol]
            session = cal.minute_to_session(minute, "previous")
        df = self.prices.session_prices(session)
        date = df.index[0]
        df = df.pt.stacked.droplevel(0)  # one row for each symbol
        chg_df = self.chg(end=date, days=1, style=False)
        chg_df.pop("close")  # so as not to replicate column
        df = pd.concat([df, chg_df], axis=1)
        df = df[["pct_chg", "chg", "close", "open", "high", "low", "volume"]]
        if not style:
            return df
        if len(df.index) > 1:
            df = add_summary_row(df, ("mean", "pct_chg"), label="Av.")
        caption = f"Price {date.strftime('%Y-%m-%d')}"
        return style_df(df, caption=caption)

    def price_at(self, ts: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """Price at a specific datetime.

        Parameters
        ----------
        ts
            Datetime at which to return most recently registered price.

            Default: most recently registered price at lowest available
            interval.
        """
        return self.prices.price_at(ts)

    def price(
        self, ts: pd.Timestamp | str | None = None, style: bool = True
    ) -> pd.DataFrame:
        """Price on a day or at a timestamp.

        Parameters
        ----------
        ts: pd.Timestamp | str | None, default: most recent day
            If require price for a day then `ts` should be passed with no
            time component and with tz as None.

            Otherwise, timestamp for which require most recently registered
            price.

        style
            True to format output and return a `Styler`.
            Only relevant if label represents a day.
        """
        if ts is None or mp.helpers.is_date(pd.Timestamp(ts)):
            return self.price_on(ts, style)
        else:
            return self.price_at(ts)

    def price_range(self, **kwargs) -> pd.DataFrame:
        """OHLCV over period.

        Exposes `self.prices.price_range`.

        See `help(self.prices.price_range)` for parameters.
        """
        return self.prices.price_range(**kwargs)

    @property
    def today(self) -> Styler:
        """Price row for most recent day."""
        return self.price_on(style=True)

    @property
    def now(self) -> pd.DataFrame:
        """Most recently registered price."""
        return self.price_at()


class Analysis(Base):
    """Single instrument analysis.

    Properties
    ----------

    symbol -> str:
        Instrument symbol.

    Methods
    -------
    plot():
        Chart prices over a specified period.

    max_adv() -> pd.Dataframe | Styler:
        Maximum percentage advance over specified period.

    max_dec() -> pd.Dataframe | Styler:
        Maximum percentage decline over specified period.

    corr() -> pd.DataFrame | Styler:
        Correlation with other Analysis object over specified period.
    """

    def __init__(self, prices: mp.PricesBase):
        """Construct `Analysis` instance.

        Parameters
        ----------
        prices:
            Instance of a subclass of `market_prices.PricesBase` from
            which to request price data for the single symbol to be
            analysed. For example:
                prices = market_prices.PricesYahoo("MSFT")

        Raises
        ------
        ValueError
            If `prices` gets prices for multiple symbols.
        """
        if len(prices.symbols) > 1:
            msg = (
                "The Analysis class requires a `prices` instance that gets"
                "price data for a single symbol, although the past instance"
                f" gets prices for {len(prices.symbols)}: {prices.symbols}."
            )
            raise ValueError(msg)
        super().__init__(prices)

    @property
    def symbol(self) -> str:
        """Instrument symbol."""
        return self.prices.symbols[0]

    def plot(
        self,
        interval: mp.intervals.RowInterval | None = None,
        engine: ChartEngine = "bqplot",
        chart_type: Literal["line", "candle"] = "candle",
        max_ticks: int | None = None,
        log_scale: bool = True,
        **kwargs,
    ) -> guis.GuiOHLC | guis.GuiLine | mpl.artist.Artist:
        """Chart prices over specified period.

        Parameters
        ----------
        interval
            Price data interval to use, as 'interval` parameter described
            by `help(self.prices.get)`.

            Default: None if `engine` "bqplot" otherwise '1d'.

        engine : Literal["bqplot", "matplotlib"]
            Chart backend.

        chart_type
            Chart type. 'candle' option only implemented for bqplot engine.

        log_scale
            True for price axis to be to log scale. Only implemented if
            `engine` is "bqplot".

        max_ticks
            Maximum number of x-axis ticks that will shown by default (more
            can be shown via slider). None for no limit. Only implemented
            if `engine` is "bqplot".

        **kwargs:
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'close_only' if engine is "matplotlib".
        """
        if engine == "bqplot":
            kwargs["interval"] = interval
            Cls = guis.GuiOHLC if chart_type == "candle" else guis.GuiLine
            return Cls(self, log_scale=log_scale, max_ticks=max_ticks, **kwargs)
        else:
            interval = "1d" if interval is None else interval
            subset = self.prices.get(close_only=True, **kwargs)
            return subset.plot()

    def max_adv(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Maximum percentage advance over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'lose_single_symbol'.

            By default return will be evaluted from data of the highest
            interval that can most accurately reflect the period start and
            end (either as passed or evaluated). This will be a composite
            table if the 'end' could be represented more accurately by a
            lower interval for which data is not otherwise available to
            cover the full requested period. Alternatively, 'interval' can
            be passed to evaluate the movement based on data for a specific
            interval.
        """
        if "interval" not in kwargs:
            kwargs["composite"] = True
        prices = self.prices.get(lose_single_symbol=True, fill="both", **kwargs)
        if isinstance(prices.index, pd.IntervalIndex):
            prices = prices.pt.indexed_left
        df = max_advance(prices, label=self.symbol)
        if not style:
            return df
        caption = f"Maximum Advance {period_string(**kwargs)}"
        return style_df(df, caption=caption)

    def max_dec(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Maximum percentage decline over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'lose_single_symbol'.

            By default return will be evaluted from data of the highest
            interval that can most accurately reflect the period start and
            end (either as passed or evaluated). This will be a composite
            table if the 'end' could be represented more accurately by a
            lower interval for which data is not otherwise available to
            cover the full requested period. Alternatively, 'interval' can
            be passed to evaluate the movement based on data for a specific
            interval.
        """
        if "interval" not in kwargs:
            kwargs["composite"] = True
        prices = self.prices.get(lose_single_symbol=True, fill="both", **kwargs)
        if isinstance(prices.index, pd.IntervalIndex):
            prices = prices.pt.indexed_left
        df = max_decline(prices, label=self.symbol)
        if style:
            caption = f"Maximum Decline {period_string(**kwargs)}"
            return style_df(df, caption=caption)
        else:
            return df

    def corr(
        self, other: "Analysis", method: str = "pearson", style: bool = True, **kwargs
    ) -> float | pd.Styler:
        """Correlation with another Analysis object.

        Correlation will be based on most granular constant interval
        avaiable for period. Pass 'interval' within kwargs to force
        corrleation with data for a specific interval.

        Parameters
        ----------
        method
            As pd.DataFrame.corr() 'method' option.

        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'composite' or 'lose_single_symbol'.
        """
        # corr acts on Series and takes a Series as other
        subset = self.prices.get(close_only=True, **kwargs).iloc[:, 0]
        other_subset = other.prices.get(close_only=True, **kwargs).iloc[:, 0]
        corr = subset.corr(other_subset, method=method)
        if not style:
            return corr
        df = pd.DataFrame([corr], index=[self.symbol], columns=[other.symbol])
        caption = (
            f"Correlation {period_string(**kwargs)} "
            + f" ({range_string(subset.index, shand=True)})"
        )
        return style_df_grid(
            df,
            format="{:.2f}",
            caption=caption,
            gradient_kwargs={"cmap": "RdBu", "vmin": -1, "vmax": 1},
        )

    def movements(
        self,
        interval: mp.intervals.RowInterval,
        trend_kwargs: dict,
        trend_cls: type[TrendsProto] = Trends,
        **kwargs,
    ) -> MovementsSupportChartAnaly:
        """Evaluate trends over a given period.

        Parameters
        ----------
        interval
            Price data interval to use for analysis, as 'interval`
            parameter described by `help(self.prices.get)`.

        trend_kwargs
            Kwargs to pass to the `trend_cls`. Do not include 'data' or
            'interval'.

            For the default `trend_cls` (`trends.Trends`) the kwargs are
            'prd', 'ext_break', 'ext_limit' and 'min_bars'. See
            documentation for trends class with
            `help(trends.Trends.__doc__)`.

        trend_cls
            Class to use to evaluate trends. Must fulfil protocol described
            by `trends_base.TrendsProto`.

            Default is `trends.Trends`.

        **kwargs:
            Parameters to define period / price data to be analysed. See
            module doc with `help(analysis.__doc__)`.
        """
        data = self.prices.get(interval, lose_single_symbol=True, **kwargs)
        if isinstance(data.index, pd.IntervalIndex):
            data.index = data.index.left
        return trend_cls(data, interval, **trend_kwargs).get_movements()

    def trends_chart(
        self,
        interval: mp.intervals.RowInterval,
        trend_kwargs: dict,
        gui_cls: type[TrendsGuiBase] = TrendsGui,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        **kwargs,
    ) -> TrendsGuiBase:
        """Visualise trends on an OHLC chart.

        Underlying trends data can be accessed via the following attributes
        of the returned object:
            cases: Movements

            trends: Instance of trends class responsible for evaluating
            movement.

        Parameters
        ----------
        interval
            Price data interval to use for analysis, as 'interval`
            parameter described by `help(self.prices.get)`.

        trend_kwargs
            Kwargs to pass to the `gui_cls`. Do not include 'analysis' or
            'interval'.

            For the default `gui_cls` (`TrendsGui`) the kwargs are
            'prd', 'ext_break', 'ext_limit' and 'min_bars'. See
            documentation for associated trends class with
            `help(trends.Trends.__doc__)`.

        gui_cls
            Class to use to create gui to visualise evaluated trends. Must
            be a subclass of `TrendsGuiBase`.

            Default is `TrendsGui`.

        max_ticks
            Maximum number of bars (x-axis ticks) that will shown by
            default (more can be shown via slider). None for no limit.

        log_scale
            True to plot prices against a log scale. False to plot prices
            against a linear scale.

        display
            False to not display the GUI.

        **kwargs:
            Parameters to define period / price data to be analysed. See
            module doc with `help(analysis.__doc__)`.
        """
        return gui_cls(
            self,
            interval,
            **trend_kwargs,
            max_ticks=max_ticks,
            log_scale=log_scale,
            display=display,
            **kwargs,
        )


class Compare(Base):
    """Analyse and compare multiple instruments.

    Properties
    ----------
    symbols -> list[str]:
        Symbols of all instruments being analysed.

    Methods
    -------
    analysis() -> Analysis:
        Analysis object for specific symbol.

    plot():
        Chart rebased prices over a specified period.

    max_adv() -> pd.DataFrame | Styler:
        Maximum percentage advance over specified period.

    max_dec() -> pd.DataFrame | Styler:
        Maximum percentage decline over specified period.

    relative_strength() -> pd.DataFrame | Styler:
        Relative strength between specified instruments over specified period.

    relative_strength_max_adv() -> pd.DataFrame | Styler:
        Relative strength of symbols maximum advances over specified period.

    relative_strength_max_dec() -> pd.DataFrame | Styler:
        Relative strength of symbols maximum declines over specified period.

    corr() -> pd.DataFrame | Styler:
        Correlation between specified instruments over specified period.

    Static Methods
    --------------
    max_chg_compare() -> pd.DataFrame | Styler:
        Maximum change, over specific price data, for multiple instruments.
    """

    PctChgBarCls = guis.GuiPctChgMult

    @staticmethod
    def max_chg_compare(
        direction: Literal["advance", "decline"],
        prices: pd.DataFrame,
        style: bool = True,
        **kwargs,
    ) -> pd.DataFrame | Styler:
        """Maximum change of multiple instruments.

        Parameters
        ----------
        direction
            'advance' for maximum advance, 'decline' for maximum decline.

        prices
            Price table from which to evaluate maximum change.

        style
            True to format ouput and return as Styler.

        **kwargs
            Parameters that were passed to define `prices`.
        """
        func = max_advance if direction == "advance" else max_decline
        dfs = []
        for symbol, s_prices in prices.T.groupby(level="symbol"):
            s_prices = s_prices.T
            s_prices.columns = s_prices.columns.droplevel(level=0)
            if isinstance(s_prices.index, pd.IntervalIndex):
                s_prices = s_prices.pt.indexed_left
            dfs.append(func(s_prices, label=symbol))
        df = pd.concat(dfs)

        dur_cols = [c for c in DURATION_COLUMNS if c in df]
        for col in dur_cols:
            if col in df:
                df.fillna({col: 0}, inplace=True)
            if col == "days":
                df[col] = df[col].astype("int64")

        if direction == "advance":
            df = df[["start", "low", "end", "high", "pct_chg"] + dur_cols]
        else:
            df = df[["start", "high", "end", "low", "pct_chg"] + dur_cols]
        if not style:
            return df
        if len(df.index) > 1:
            av_c = None if not dur_cols else dur_cols[0]
            sum_cols = ["pct_chg"] if av_c is None else ["pct_chg", av_c]
            df = add_summary_row(df, ("mean", sum_cols), label="Av.")
        direction_ = "Advance" if direction == "advance" else "Decline"
        caption = f"Maximum {direction_} {period_string(**kwargs)}"
        return style_df(df, caption=caption)

    def __init__(self, prices: mp.PricesBase):
        """Construct `Compare` instance.

        Parameters
        ----------
        prices:
            Instance of a subclass of `market_prices.PricesBase` from
            which to request price data for the multiple symbols to be
            compared. For example:
                prices = market_prices.PricesYahoo("MSFT, AZN.L")

        Raises
        ------
        ValueError
            If `prices` gets prices for only a single symbol.
        """
        if len(prices.symbols) == 1:
            msg = (
                "The Compare class requires a `prices` instance that gets"
                "price data for multiple symbols, although the past instance"
                f"gets prices for only one: {prices.symbols[0]}."
            )
            raise ValueError(msg)
        super().__init__(prices)

    @property
    def symbols(self) -> list[str]:
        return self.prices.symbols

    def analysis(self, symbol: str) -> Analysis:
        """Analysis object for a given symbol."""
        prices = self.prices.prices_for_symbols(symbol)
        return Analysis(prices)

    def relative_strength(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Relative strength over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'composite' or 'lose_single_symbol'.
        """
        subset = self.prices.get(fill="both", add_a_row=True, close_only=True, **kwargs)
        pct_chg = pct_chg_top_to_bot(subset)
        df = pd.DataFrame({s: pct_chg - v for s, v in pct_chg.items()})
        df.columns.name = "symbol"
        if not style:
            return df
        if len(df.index) > 1:
            df = add_summary(df, "mean", axis="column", label="Av.")
        caption = (
            f"Relative Strength {period_string(**kwargs)} "
            + f"({range_string(subset.index, close=True, shand=True)})"
        )
        return style_df_grid(df, caption=caption)

    def plot(
        self,
        interval: mp.intervals.RowInterval | None = None,
        engine: ChartEngine = "bqplot",
        rebase_on_zoom: bool = True,
        max_ticks: int | None = None,
        log_scale: bool = True,
        **kwargs,
    ) -> guis.GuiMultLine | mpl.artist.Artist:
        """Chart rebased close prices over specified period.

        Parameters
        ----------
        interval
            Price data interval to use, as 'interval` parameter described
            by `help(self.prices.get)`.

            Default: None if `engine` "bqplot" otherwise '1d'.

        engine: Literal["bqplot", "matplotlib"]
            Chart backend

        rebase_on_zoom
            True to rebase prices following zoom. Only implemented if
            `engine` is "bqplot".

        max_ticks
            Maximum number of x-axis ticks that will shown by default (more
            can be shown via slider). None for no limit. Only implemented
            if `engine` is "bqplot".

        log_scale
            True for price axis to be to log scale. Only implemented for
            if `engine` is "bqplot".

        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'composite' or 'lose_single_symbol'.
        """
        if engine == "bqplot":
            return guis.GuiMultLine(
                self, interval, rebase_on_zoom, max_ticks, log_scale, **kwargs
            )
        else:
            interval = "1d" if interval is None else interval
            subset = self.prices.get(close_only=True, **kwargs)
            return rebase_to_row(subset).plot()

    def _max_chg(self, direction: Literal["advance", "decline"], style=True, **kwargs):
        """Maximum change over specified period."""
        if "interval" not in kwargs:
            kwargs["composite"] = True
        prices = self.prices.get(fill="both", **kwargs)
        return self.max_chg_compare(direction, prices, style, **kwargs)

    def max_adv(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Maximum percentage advance over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`.

            By default return will be evaluted from data of the highest
            interval that can most accurately reflect the period start and
            end (either as passed or evaluated). This will be a composite
            table if the 'end' could be represented more accurately by a
            lower interval for which data is not otherwise available to
            cover the full requested period. Alternatively, 'interval' can
            be passed to evaluate the movement based on data for a specific
            interval.
        """
        return self._max_chg("advance", style, **kwargs)

    def max_dec(self, style: bool = True, **kwargs) -> pd.DataFrame | Styler:
        """Maximum percentage decline over specified period.

        Parameters
        ----------
        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`.

            By default return will be evaluted from data of the highest
            interval that can most accurately reflect the period start and
            end (either as passed or evaluated). This will be a composite
            table if the 'end' could be represented more accurately by a
            lower interval for which data is not otherwise available to
            cover the full requested period. Alternatively, 'interval' can
            be passed to evaluate the movement based on data for a specific
            interval.
        """
        return self._max_chg("decline", style, **kwargs)

    def _relative_strength_max_chg(
        self, func: Callable, style: bool = True, **kwargs
    ) -> pd.DataFrame | Styler:
        """Relative strength of maximum change over specified period."""
        df = func(style=False, **kwargs)
        srs = df["pct_chg"]
        rel_str_df = pd.DataFrame({s: srs - v for s, v in srs.items()})
        if not style:
            return rel_str_df
        if len(rel_str_df.index) > 1:
            rel_str_df = add_summary(rel_str_df, "mean", axis="both", label="Av.")
        styler = func(style=True, **kwargs)
        caption = f"Relative Strength {styler.caption}"
        return style_df_grid(rel_str_df, caption=caption)

    def relative_strength_max_adv(
        self, style: bool = True, **kwargs
    ) -> pd.DataFrame | Styler:
        """Relative strength of maximum advance over specified period.

        Parameters as for `max_adv`.
        """
        return self._relative_strength_max_chg(self.max_adv, style=style, **kwargs)

    def relative_strength_max_dec(
        self, style: bool = True, **kwargs
    ) -> pd.DataFrame | Styler:
        """Relative strength of maximum decline over specified period.

        Parameters as for `max_dec`.
        """
        return self._relative_strength_max_chg(self.max_dec, style=style, **kwargs)

    def corr(
        self, method: str = "pearson", style: bool = True, **kwargs
    ) -> pd.DataFrame | Styler:
        """Correlation over specified period.

        Correlation will be based on LEAST granular constant interval
        that can most accurately represent the period start and end. Pass
        `interval` within kwargs to force corrleation to be against a
        specific interval.

        Parameters
        ----------
        method
            As pd.DataFrame.corr() 'method' option.

        **kwargs
            Parameters to define period / price data to be analysed. See
            method doc with `help(analysis.__doc__)`. Cannot include
            'close_only'.
        """
        subset = self.prices.get(fill="both", close_only=True, **kwargs)
        df = subset.corr(method=method)
        if not style:
            return df
        if len(df.index) > 1:
            df = add_summary(df, "sum", axis="column", label="Av.")
        df["Av."] = (df["Av."] - 1) / (len(df.columns) - 2)
        caption = (
            f"Correlation {period_string(**kwargs)}" f" ({range_string(subset.index)})"
        )
        return style_df_grid(
            df,
            format="{:.2f}",
            gradient_kwargs={"cmap": "RdBu", "vmin": -1},
            caption=caption,
        )
