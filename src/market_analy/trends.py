"""Trend Analysis.

Classes to evaluate and store trends:

    Movement
        Data class to represent an advance or decline identified by the
        `Trends` class.

    Movements
        Class that stores all `Movement` identified over an analysis
        period.

    Trends
        Evaluates trends from OHLC data. Broadly, a trend is considered to
        exist whenever a movement (advance or decline) is continually
        extended, by way of exceeding a limit line, whilst at the same time
        NOT exceeding a break line that underpins the trend. See
        `Trends.__doc__` for more specific documentation.

    TrendsGui
        Offers a GUI to visualise movements identified by `Trends`.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import operator
import typing

import bqplot as bq
import ipywidgets as w
import market_analy
import market_prices as mp
import numpy as np
import pandas as pd

from market_analy.charts import OHLCTrends, tooltip_html_style, TOOLTIP_STYLE
from market_analy.config import COL_ADV, COL_DEC
from market_analy.formatters import (
    formatter_percent,
    formatter_datetime,
    formatter_float,
)
from market_analy.guis import TrendsGuiBase
from market_analy.movements_base import (
    MovementProto,
    MovementsBase,
    MovementsChartProto,
)
from market_analy.utils import bq_utils as ubq

if typing.TYPE_CHECKING:
    from market_analy.analysis import Analysis


class TrendParams(typing.TypedDict):
    """Trend Parameters.

    Trend parameters for a movement identified with `Trends`.
    """

    interval: mp.intervals.RowInterval
    prd: int
    ext_break: float
    ext_limit: float
    min_bars: int


@dataclass(frozen=True)
class Movement(MovementProto):
    """Advance or decline identified by `Trends`.

    Attributes (in addition to those provided via `MovementProto`)
    ----------
    params
        Parameters passed to `Trends`

    line_break
        Line from movement start to first occassion that movement would end
        by way of breaking Break Line (or to end of data if movement would
        not have otherwise ended by breaking the Break Line).

        Line should 'reset' on each occasion that conditions are met for
        confirming the start of a movement.

    line_limit
        Line from movement start to first occassion that movement would end
        by way of not having extended the Limit Line (or to end of data if
        movement extended Limit Line within `prd` of end of data).

        Line should 'reset' on each occasion that the prior Limit Line was
        extended.

    by_break
        True: Movement ended by way exceeding break line.
        False: Movement ended by way of not exceeding limit line.
        None: Movemment had not ended as at the end of the data set.
    """

    params: TrendParams
    line_break: pd.Series
    line_limit: pd.Series
    by_break: bool | None

    # NOTE no idea why this is necessary, but it is. Perhaps to do with
    # inheriting from Protocol? Subclassing a Dataclass?
    def __eq__(self, other):
        return super().__eq__(other)

    def __lt__(self, other):
        return super().__lt__(other)

    def __le__(self, other):
        return super().__le__(other)

    def __gt__(self, other):
        return super().__gt__(other)

    def __ge__(self, other):
        return super().__ge__(other)

    @property
    def open(self) -> bool:
        """Movement has not ended."""
        return self.by_break is None

    @property
    def chg(self) -> float:
        """Absolute change over movement."""
        return self.end_px - self.start_px

    @property
    def chg_pct(self) -> float:
        """Percentage change over movement."""
        return self.chg / self.start_px

    @property
    def conf_chg(self) -> float | None:
        """Absolute change between start and end confirmations."""
        return None if self.open else self.end_conf_px - self.start_conf_px  # type: ignore

    @property
    def conf_chg_pct(self) -> float | None:
        """Percentage change between start and end confirmations."""
        return None if self.open else self.conf_chg / self.start_conf_px  # type: ignore


@dataclass
class Movements(MovementsBase, MovementsChartProto):
    """Movements identified over an analysis period by `Trends`."""

    moves: list[Movement]

    MARKER_MAP: typing.ClassVar = {
        "cross": "End",
        "circle": "Start conf",
        "square": "End conf",
    }

    def mark_to_move(self, mark: bq.Scatter, event: dict) -> Movement:
        # Only for type checker to know move is a Movement as defined on this module.
        move = super().mark_to_move(mark, event)
        assert isinstance(move, Movement)
        return move

    @staticmethod
    def get_move_html(move: Movement) -> str:
        """Return html to describe a movement."""
        color = "crimson" if move.is_dec else "limegreen"
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>Start: " + formatter_datetime(move.start)
        s += f"<br>Start px: {formatter_float(move.start_px)}"
        s += f"<br>End: {'None' if move.open else formatter_datetime(move.end)}"
        s += f"<br>Chg: {formatter_percent(move.chg_pct)}"
        by = "None" if move.open else ("break" if move.by_break else "limit")
        s += f"<br>By: {by}"
        s += f"<br>Duration: {move.duration}"
        s += f"<br>Start conf: {formatter_datetime(move.start_conf)}"
        end_conf = "None" if move.open else formatter_datetime(move.end_conf)
        s += f"<br>End conf: {end_conf}"

        s += "<br>Conf chg: "
        if move.open:
            s += "None"
        else:
            chg_color = "crimson" if move.conf_chg_pct < 0 else "limegreen"  # type: ignore
            chg_style = tooltip_html_style(color=chg_color, line_height=1.3)
            s += f"<span {chg_style}>{formatter_percent(move.conf_chg_pct)}</span>"

        s += "</p"
        return s

    def handler_hover_start(self, mark: bq.marks.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""
        move = self.mark_to_move(mark, event)
        mark.tooltip.value = self.get_move_html(move)

    def _handler_not_start(self, mark: bq.Scatter, event: dict):
        """Handler to hover over any scatter mark not representing a movement start.

        Displays tooltip describing what scatter represents and corresponding timestamp.
        """
        style = tooltip_html_style(color=mark.colors[0], line_height=1.3)
        date = ubq.discontinuous_date_to_timestamp(event["data"]["x"])
        name = self.MARKER_MAP[mark.marker]
        s = f"<p {style}>{name}: {formatter_datetime(date)}"
        s += f"<br>{name + ' px'}: {formatter_float(event['data']['y'])}</p"
        mark.tooltip.value = s

    def handler_hover_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering on mark representing a movement end."""
        self._handler_not_start(mark, event)

    def handler_hover_conf_start(self, mark: bq.Scatter, event: dict):
        """Handler for hovering on mark representing when movement start confirmed."""
        self._handler_not_start(mark, event)

    def handler_hover_conf_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering on mark representing when movement end confirmed."""
        self._handler_not_start(mark, event)

    def handler_click_trend(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking mark representing a movement start.

        Removes all existing scatters from figure.

        Adds scatters and lines for the single trend represented by the
        clicked scatter point.
        """
        move = self.mark_to_move(mark, event)
        color = mark.colors[0]

        marks = []

        def f(*args):
            marks.append(chart.create_scatter(*args))

        def fl(data: pd.Series, color_: str, line_style: str, desc: str) -> bq.Lines:
            tooltip_str = f"<p {tooltip_html_style(color=color_)}>{desc}</p>"
            line = bq.Lines(
                x=data.index,
                y=data,
                scales=chart.scales,
                colors=[color_],
                line_style=line_style,
                tooltip=w.HTML(value=tooltip_str),
                tooltip_style=TOOLTIP_STYLE,
            )
            marks.append(line)
            return line

        group = "one_trend"
        # remove any existing marks for a selected movement.
        if group in chart.added_marks_groups:
            chart.remove_added_marks(group)

        if move.closed:
            style = tooltip_html_style(color=color)
            s = f"<p {style}>Conf chg: "
            chg_color = COL_DEC if move.conf_chg_pct < 0 else COL_ADV  # type: ignore
            chg_style = tooltip_html_style(color=chg_color)
            s += f"<span {chg_style}>{formatter_percent(move.conf_chg_pct)}</span></p>"

            if move.is_adv:
                color_area = (
                    COL_ADV if move.start_conf_px < move.end_conf_px else COL_DEC
                )
            else:
                color_area = (
                    COL_ADV if move.start_conf_px > move.end_conf_px else COL_DEC
                )

            mark_chg = bq.Lines(
                x=[[move.start_conf, move.end_conf]] * 2,
                y=[[move.start_conf_px] * 2, [move.end_conf_px] * 2],
                scales=chart.scales,
                opacities=[0],
                fill="between",
                fill_colors=[color_area],
                fill_opacities=[0.2],
                tooltip=w.HTML(value=s),
                tooltip_style=TOOLTIP_STYLE,
            )
            chart.add_marks([mark_chg], group, under=True)

        def handler_start(mark: bq.Scatter, event: dict):
            mark.tooltip.value = self.get_move_html(move)

        f([move.start], [move.start_px], color, mark.marker, handler_start)

        handler = self._handler_not_start
        f([move.start_conf], [move.start_conf_px], color, "circle", handler)
        if move.closed:
            f([move.end], [move.end_px], color, "cross", handler)
            f([move.end_conf], [move.end_conf_px], color, "square", handler)

        color_break, color_limit = "white", "slategray"
        if not move.by_break:
            color_break, color_limit = color_limit, color_break
        fl(move.line_break, color_break, "dashed", "Break line")
        fl(move.line_limit, color_limit, "dashed", "Limit line")

        chart.hide_scatters()
        chart.add_marks(marks, group)


class Trends:
    """Evaluate trends from OHLC data.

    Broadly, a trend is considered to exist whenever a movement (advance or
    decline) is continually extended, by way of exceeding a 'limit line'
    whilst at the same time NOT exceeding a 'break line' that underpins the
    trend. More specifically:

        A Movement is confirmed as having started whenever:
            Break Line over a given period (`prd`) has NOT been broken.
            For an advance/decline, the Break Line is defined as a
            rising/falling line with a given gradient (`prd` / `ext_break`)
            and which passes through the low/high of the bar on which the
            movement is considered to have started.

            Limit Line has been extended during a given period (`prd`).
            For an advance/decline, the Limit Line is defined as a
            rising/falling line with a given gradient (`prd` / `ext_limit`)
            and which passes through the low/high of the bar on which the
            advance/decline movement is considered to have started.

        Movement confirmed as having ended by way of:
            break, whenever the Break Line is exceeded. In this case the
            Break Line is defined as passing through the low/high of the
            most recent bar when the advance/decline fulfilled the
            conditions to be considered as having started (this may be the
            first time the movement was confirmed or a subsequent
            occasion).

            limit, whenever the Limit Line is not exceeded over the prior
            `prd`, where that Limit Line is defined as passing through the
            most recent high/low of the advance/decline.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns that include "high", "low" and "close". Any
        other columns will be ignored.

        DataFrame must be indexed with a `pd.DatetimeIndex`.

    interval : pd.Timedelta | str
        Interval that each row of `data` represents. Examples:

            pd.Timedelta(15, "T") (or "15T"): each row of data represents
            15 minutes

            pd.Timedelta(1, "D") (or "1D"): for daily data.

    prd : int
        Period, in number of bars, over which the Break Line should not
        have been broken in order to declare a movement as having started.

        Also, the period over which the Limit Line must be extended to
        consider a movement as having started or to be continuing.

        See class documentation for definition of Break Line and
        Limit Line.

        Example, if each row of `data` (i.e. each bar) represents one
        session (i.e. if `interval` is "1D") then passing `prd` as 10 will
        set the period as 10 sessions, whilst if each bar represents an
        interval of 30 minutes (i.e. `interval` is "30T") then passing 10
        will set the period as 5 trading hours.

    ext_break
        The percentage change in the Break Line over `prd`. Accordingly,
        defines the gradient of the Break Line.

        Pass as a float, for example 0.005 for 0.5%.

        See class documentation for definition of 'Break Line'.

    ext_limit
        The percentage change in the Limit Line over `prd`. Accordingly,
        defines the gradient of the Limit Line.

        Pass as a float, for example 0.0025 for 0.25%.

        See class documentation for definition of 'Limit Line'.

    min_bars : int, default: 0
        The minimum number of bars that any movement should be comprised
        of.

    Notes
    -----
    NOTE: testing of `Trends` class is LIMITED.
    Tests are limited to verifying return as expected for certain
    combinations of parameters. In turn, 'what's expected' is as accepted
    from detailed visual inspection of the resulting trends. Class lacks,
    and requires, a test suite that verifies all possible returns of each
    of it's constituent methods, i.e. with parameters defined either side
    of boudaries to verify differences in return. Those tests should use
    dummy price data designed to the the boundaries.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        interval: mp.intervals.RowInterval,
        prd: int,
        ext_break: float,
        ext_limit: float,
        min_bars: int = 0,
    ):
        self.data = data.copy()
        self.interval = mp.intervals.to_ptinterval(interval)
        self.prd = prd

        self.ext_break = ext_break
        self.ext_limit = ext_limit
        self.min_bars = min_bars

        self.grad_break = ext_break / (prd - 1)
        self.grad_limit = ext_limit / (prd - 1)

        self.fctrs_pos_break = np.linspace(1, 1 + ext_break, prd)
        self.fctrs_neg_break = np.linspace(1, 1 - ext_break, prd)
        self.fctrs_pos_limit = np.linspace(1, 1 + ext_limit, prd)
        self.fctrs_neg_limit = np.linspace(1, 1 - ext_limit, prd)

    @property
    def params(self) -> TrendParams:
        """Parameters received by constructor."""
        return {
            "interval": self.interval,
            "prd": self.prd,
            "ext_break": self.ext_break,
            "ext_limit": self.ext_limit,
            "min_bars": self.min_bars,
        }

    @functools.lru_cache
    def base_intact(self, is_adv: bool) -> pd.Series:
        """Query if base was broken over any `self.prd` bars.

        Returns
        -------
        pd.Series
            index: as `self.data.index`
            dtype: bool

            Bool indicates if the base line was broken during the
            `self.prd` bars prior to the label. NB first `self.prd` bars
            are False.
        """

        def func(arr: np.ndarray) -> bool:
            """Query if base broken over an array shorter than `self.prd`."""
            fctrs = self.fctrs_pos_break if is_adv else self.fctrs_neg_break
            line = arr.iloc[0] * fctrs[: len(arr)]
            bv = (arr >= line) if is_adv else (arr <= line)
            return bv.all()

        col = self.data.low if is_adv else self.data.high
        bv = col.rolling(self.prd, min_periods=self.prd).apply(func)
        bv[bv.isna()] = 0
        return bv.astype("bool")

    @functools.lru_cache
    def limit_extended(self, is_adv: bool) -> pd.Series:
        """Query if limit is extended over any `self.prd` bars.

        Returns
        -------
        pd.Series
            index: as `self.data.index`
            dtype: bool

            Bool indicates if the limit line was extended during the
            `self.prd` bars prior to the label. NB first `self.prd` bars
            are False.
        """

        def func(arr: np.ndarray) -> bool:
            """Query if limit extended over array shorter than `self.prd`."""
            fctrs = self.fctrs_pos_limit if is_adv else self.fctrs_neg_limit
            line = arr.iloc[0] * fctrs[: len(arr)]
            bv = (arr > line) if is_adv else (arr < line)
            return bv.any()

        col = self.data.high if is_adv else self.data.low
        bv = col.rolling(self.prd, min_periods=self.prd).apply(func)
        bv[bv.isna()] = 0
        return bv.astype("bool")

    @functools.lru_cache
    def start_confs_all(self, is_adv: bool) -> pd.DatetimeIndex:
        """All bars on which a movement could be confirmed as having started."""
        bv = self.base_intact(is_adv) & self.limit_extended(is_adv)
        return bv[bv].index

    def start_from_start_conf(self, start_conf: pd.Timestamp) -> pd.Timestamp:
        """Get start bar from start confirmed bar."""
        idx_start_conf = self.data.index.get_loc(start_conf)
        idx_start = idx_start_conf - self.prd + 1
        return self.data.index[idx_start]

    def start_conf_from_start(self, start: pd.Timestamp) -> pd.Timestamp | None:
        """Get start confrirmed bar corresponding with a start bar.

        Returns None if `conf_start` would be beyond the end of available
        data.
        """
        idx_start = self.data.index.get_loc(start)
        idx_start_conf = idx_start + self.prd - 1
        if idx_start_conf > len(self.data):
            return None
        return self.data.index[idx_start_conf]

    def get_line(self, subset: pd.Series, is_adv: bool, limit=True) -> pd.Series:
        """Get a base or limit line for a subset of the data.

        Parameters
        ----------
        subset
            Continuous (over rows) subset of `self.data`, either 'high' or
            'low' column.

        is_adv
            True: if movement is an advance.
            False: if movement is a decline.

        limit
            True: return a limit line.
            False: retrun a break line.

        Returns
        -------
        line : pd.Series
            Series with index as subset and values delineating a line with
            gradient `self.grad_break` or `self.grad_limit` passing through
            `subset`[0].
        """
        op = operator.add if is_adv else operator.sub
        grad = self.grad_limit if limit else self.grad_break
        fctrs = np.array([op(1, (i * grad)) for i in range(len(subset))])
        return pd.Series(fctrs * subset.iloc[0], index=subset.index)

    def get_limit_line(
        self, start: pd.Timestamp, end: pd.Timestamp, is_adv: bool
    ) -> pd.Series:
        col = self.data.high if is_adv else self.data.low
        subset = col[start:end]

        srss = []
        while True:
            line = self.get_line(subset, is_adv)
            bv = subset > line if is_adv else subset < line
            idx = bv.argmax() if bv.any() else None
            srss.append(line[:idx])
            if idx is None or idx + 1 == len(bv):
                break
            subset = subset[idx:]

        return pd.concat(srss)

    def get_limit(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series]:
        """Get bar and price when price has failed to extend limit.

        Parameters
        ----------
        start_conf
            Bar when movement was confirmed.

        is_adv
            True: Movement represents an advance.
            False: Movemment represents a decline.

        Returns
        -------
        bar : pd.Timestamp | None
            First bar, following `start_conf`, when the price failed to
            extend the limit line over the prior `self.prd`.

            None if price did not fail to extend the limit line as at the
            end of the available data.

        break_px : float | None
            Close price of `bar` (as bar is included within `self.prd`, all
            of day should be included as it had this period available to
            try and extend the limit).

            None if price did not fail to extend the limit line as at the
            end of the available data.

        limit_line : pd.Series
            Limit line from movement start through `bar`.
        """
        bv = self.limit_extended(is_adv).loc[start_conf:]
        if bv.all():
            bar, px = None, None
            bar_ = self.data.index[-1]
        else:
            bar = bar_ = bv.index[bv.argmin()]
            px = self.data.loc[bar].close

        start = self.start_from_start_conf(start_conf)
        line = self.get_limit_line(start, bar_, is_adv)
        return bar, px, line

    def get_break(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series]:
        """Get bar and price when price first breaks break line.

        Gets bar when price breaks base line for the first time
        following a movement's confirmation.

        Parameters
        ----------
        start_conf
            Bar on which start of movement confirmed.

        is_adv
            True: movement for which require break is an advance.
            False: movement for which require break is a decline.

        Returns
        -------
        break_ : pd.Timestamp | None
            Bar that movement, with confirmed start as `start_conf`,
            subsequently broke the break line for the first time.

            None if price did not break the break line by the end of the
            available data.

        break_px : float | None
            Price when break line broken. This will be the value of the
            break line or the bar open if the open exceeded the break line.

            None if price did not exceed break line by the end of the
            available data.

        break_line : pd.Series
            Break line from movement start through `break_`.
        """
        start_confs_all = self.start_confs_all(is_adv)
        idx = start_confs_all.get_loc(start_conf)
        start_confs = start_confs_all[idx:]

        col = self.data.low if is_adv else self.data.high
        frms = pd.Series(start_confs).apply(self.start_from_start_conf)
        tos = frms.shift(-1)
        srss = []
        for frm, to in zip(frms, tos):
            if pd.isna(to):
                to = None
            subset = col[frm:to]
            line = self.get_line(subset, is_adv, limit=False)
            bv = subset < line if is_adv else subset > line
            if not bv.any():
                srss.append(line[1:])  # miss first so they don't overlap
                continue

            idx = bv.argmax()
            srss.append(line[1 : idx + 1])
            line_break = pd.concat(srss)
            break_ = line_break.index[-1]
            f = min if is_adv else max
            break_px = f(line_break.iloc[-1], self.data.loc[break_].open)
            return break_, break_px, line_break

        return None, None, pd.concat(srss)

    def get_end_conf(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp | None, float | None, bool | None, pd.Series, pd.Series]:
        """Get bar and price when a movement is confirmed as having ended.

        Returns
        -------
        tuple(None, None, None, pd.Series, pd.Series) if movement end had
        not been confirmed as at end of available data, otherwise:

        end_conf : pd.Timestamp
            Bar on which movement end confirmed.

        end_conf_px : float
            Price when movement end was confirmed.

        by_break : bool
            True if movement ended as a result of breaking the base line
            on `end`.

            False if movement ended as a result of failing to extend the
            limit line over the `self.prd` bars through to `end`.

        line_break : pd.Series
            Break line from movement start through to bar when price broke
            the break line, or to last bar if break line was not broken as
            at the end of available data.

        line_limit : pd.Series
            Limit line from movement start through to first bar when price
            has failed to extend limit line over the prior `self.prd` bars,
            or through to last bar if as at the end of the available data
            the price had not failed to exceed the limit line.
        """
        end_break, end_px_break, line_break = self.get_break(start_conf, is_adv)
        end_limit, end_px_limit, line_limit = self.get_limit(start_conf, is_adv)
        if end_break is None and end_limit is None:
            return None, None, None, line_break, line_limit
        if end_break is None:
            return end_limit, end_px_limit, False, line_break, line_limit
        if end_limit is None:
            return end_break, end_px_break, True, line_break, line_limit
        # if same bar then break will occur first as limit happens at bar close
        by_break = end_break <= end_limit
        px = end_px_break if by_break else end_px_limit
        return min(end_break, end_limit), px, by_break, line_break, line_limit

    def get_end(
        self, start: pd.Timestamp, end_conf: pd.Timestamp | None, is_adv: bool
    ) -> tuple[pd.Timestamp, float]:
        """Get movement end bar and price."""
        col = self.data.high if is_adv else self.data.low
        subset = col.loc[start:end_conf]
        _end_idx = subset.argmax() if is_adv else subset.argmin()
        end = subset.index[_end_idx]
        end_px = subset.max() if is_adv else subset.min()
        return end, end_px

    def get_start(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp, float]:
        """Get movement start bar and price."""
        start = self.start_from_start_conf(start_conf)
        row = self.data.loc[start]
        return start, row.low if is_adv else row.high

    def get_duration(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        idx_start = self.data.index.get_loc(start)
        idx_end = self.data.index.get_loc(end)
        return idx_end - idx_start + 1

    def get_movement(self, start_conf: pd.Timestamp, is_adv: bool) -> Movement:
        """Get movement corresponding with a given confirmed start bar."""
        end_conf, end_conf_px, by_break, line_break, line_limit = self.get_end_conf(
            start_conf, is_adv
        )
        start, start_px = self.get_start(start_conf, is_adv)
        end, end_px = self.get_end(start, end_conf, is_adv)
        start_conf_px = self.data.loc[start_conf].close

        return Movement(
            is_adv,
            start,
            start_px,
            start_conf,
            start_conf_px,
            end,
            end_px,
            end_conf,
            end_conf_px,
            self.get_duration(start, end),
            self.params,
            line_break,
            line_limit,
            by_break,
        )

    def _get_movements(self, is_adv: bool) -> list[Movement]:
        """Return list of movements of a given direction."""
        moves = []
        start_confs_all = self.start_confs_all(is_adv)
        while True:
            move = self.get_movement(start_confs_all[0], is_adv)
            if move.duration < self.min_bars:
                start_confs_all = start_confs_all[1:]
                if start_confs_all.empty:
                    break
                continue
            moves.append(move)
            if move.end_conf is None:
                break
            minconfstart = self.start_conf_from_start(move.end)
            if minconfstart is None:
                break
            idx = start_confs_all.get_slice_bound(minconfstart, side="right")
            start_confs_all = start_confs_all[idx:]
            if start_confs_all.empty:
                break
        return moves

    def get_movements(self) -> Movements:
        """Return all movements."""
        advs = self._get_movements(True)
        decs = self._get_movements(False)
        moves = sorted(advs + decs)

        # sanity check that moves do not overlap except end/start
        prev_move: Movement | None = None
        for move in moves:
            if prev_move is not None:
                assert prev_move.end <= move.start
            prev_move = move

        return Movements(moves, self.data, self.interval)


class TrendsGui(TrendsGuiBase):
    """GUI to visualise movements evaluated by `Trends` class.

    Parameters
    ----------
    `prd`, `ext_break`, `ext_limit` and `min_bars` all as `Trends` class.

    All other parameters as base class `TrendsGuiBase`.
    """

    def __init__(
        self,
        analysis: Analysis,
        interval: mp.intervals.RowInterval,
        prd: int,
        ext_break: float,
        ext_limit: float,
        min_bars: int = 0,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        **kwargs,
    ):
        trend_kwargs = {
            "prd": prd,
            "ext_break": ext_break,
            "ext_limit": ext_limit,
            "min_bars": min_bars,
        }
        super().__init__(
            analysis=analysis,
            interval=interval,
            trend_cls=Trends,
            trend_kwargs=trend_kwargs,
            max_ticks=max_ticks,
            log_scale=log_scale,
            display=display,
            narrow_view=prd,
            wide_view=prd * 3,
            **kwargs,
        )
        self.movements: Movements
        self.trends: Trends

    def _gui_click_trend_handler(self, mark: bq.Scatter, event: dict):
        """Gui level handler for clicking a mark representing a trend start.

        Lightens 'show all scatters' button to indicate option available.
        Displays tooltip to html output.
        """
        self.trends_controls_container.lighten_single_trend()
        self.trends_controls_container.but_show_all.darken()
        move = self.movements.mark_to_move(mark, event)
        html = self.movements.get_move_html(move)
        self.html_output.display(html)

    def _add_rulers(self):
        self._close_rulers()  # close any existing
        move = self.current_move
        is_adv = move.is_adv

        ohlc_mark = next(m for m in self.chart.figure.marks if isinstance(m, bq.OHLC))

        fctrs = self.trends.fctrs_pos_break if is_adv else self.trends.fctrs_neg_break
        self._rulers.append(
            market_analy.utils.bq_utils.TrendRule(
                x=move.start.asm8,
                y=move.start_px,
                length=move.params["prd"],
                factors=fctrs,
                scales=self.chart.scales,
                ordinal_values=list(ohlc_mark.x),
                figure=self.chart.figure,
                color="orange",
                draggable=True,
                stroke_width=3,
            )
        )

        fctrs = self.trends.fctrs_pos_limit if is_adv else self.trends.fctrs_neg_limit
        idx = -move.params["prd"]
        self._rulers.append(
            market_analy.utils.bq_utils.TrendRule(
                x=move.line_limit.index[idx].asm8,
                y=move.line_limit.iloc[idx],
                length=move.params["prd"],
                factors=fctrs,
                scales=self.chart.scales,
                ordinal_values=list(ohlc_mark.x),
                figure=self.chart.figure,
                color="skyblue",
                draggable=True,
                stroke_width=3,
            )
        )
