"""Classes and protocols to represent trend movements."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import bqplot as bq
import ipywidgets as w
import numpy as np
import pandas as pd

from ..cases import CaseBase, CasesBase, CasesSupportsChartAnaly, CaseSupportsChartAnaly
from ..charts import TOOLTIP_STYLE, Groups, tooltip_html_style
from ..config import COL_ADV, COL_DEC
from ..formatters import formatter_datetime, formatter_float, formatter_percent
from ..utils import bq_utils as ubq

if typing.TYPE_CHECKING:
    from .analy import TrendParams
    from .charts import OHLCTrends


@dataclass(frozen=True, eq=False)
class MovementBase(CaseBase, ABC, CaseSupportsChartAnaly):
    """Protocol of minimum requirements to define a trend Movement.

    Attributes
    ----------
    is_adv
        True: movement represents an advance
        False: movement represents a decline.

    start
        Bar when movement started.

    start_px
        Price when movement started.

    start_conf
        Bar when movement was confirmed as having started.

    start_px
        Price when movement was confirmed as having started.

    end
        Bar when movement ended.

    end_px
        Price when movement ended.

    end_conf
        Bar when movement was confirmed as having ended.

    end_conf_px
        Price when movement was confirmed as having ended.

    duration
        Duration of trend as number of bars
    """

    is_adv: bool
    start: pd.Timestamp
    start_px: float
    start_conf: pd.Timestamp
    start_conf_px: float
    end: pd.Timestamp | None
    end_px: float
    end_conf: pd.Timestamp | None
    end_conf_px: float | None
    duration: int

    @property
    @abstractmethod
    def open(self) -> bool:
        """Movement has not ended."""
        ...

    @property
    def closed(self) -> bool:
        """Movement has ended."""
        return not self.open

    @property
    def is_dec(self) -> bool:
        """Movement is a decline."""
        return not self.is_adv

    @property
    def trend(self) -> int:
        """Trend direction. 1 for advance, -1 for decline."""
        return 1 if self.is_adv else -1

    @property
    def _start(self) -> pd.Timestamp:
        """Bar when case considered to start."""
        return self.start

    @property
    def _end(self) -> pd.Timestamp | None:
        """Bar when case considered to have concluded."""
        return self.end_conf


@dataclass(frozen=True, eq=False)
class Movement(MovementBase, CaseSupportsChartAnaly):
    """Advance or decline identified by `Trends`.

    Attributes (in addition to those provided via `MovementBase`)
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
        return (
            None if self.end_conf_px is None else self.end_conf_px - self.start_conf_px
        )

    @property
    def conf_chg_pct(self) -> float | None:
        """Percentage change between start and end confirmations."""
        return None if self.conf_chg is None else self.conf_chg / self.start_conf_px


@dataclass(frozen=True, eq=False)
class MovementAlt(MovementBase, CaseSupportsChartAnaly):
    """Advance or decline identified by `Trends`.

    Attributes (in addition to those provided via MovementProto)
    ----------
    params
        Parameters passed to `Trends`

    sel
        Start Establishment Line. Line that tracks the limit for reversals
        within the period before start_conf. Line will be comprised of
        segments, each relating to an assessed start bar. The part relating
        to the actual start bar lies to the 'right' of this bar.

    start_conf_line
        Line on which the movement was confirmed as having started. See
        `Trends` class doc.

    eel
        End Establishment Line. Line that determines if an end bar, earlier
        than that which would otherwise represent the end, is justified as
        a result of the price having failed to sufficiently extend (as
        determined by the Trends 'grad' parameter) an earlier low/high
        which would therefore be considered to better represent the
        movement's end.

        None if the movement did not end by way of consolidation or had not
        ended as at the end of the available data.

    end_line_consol
        If the price failed to cross this line within 'prd' bars then the
        movement will have been considered to have ended by way of
        consolidation.

    end_line_rvr
        If the price crosses this line then the movement will be considered
        to have ended by way of reversal as a result of having exceeded
        the reversal percentage limit.

    end_line_rvr_opp
        If the price were to cross this line then the movement would be
        considered over by way of reversal as a result of a movement in the
        opposing direction being confirmed.

        None if no opposing movement was identified before the movement
        would have otherwise ended by way of consolidation or reversal as
        a result of crossing the reversal percentage limit.

    by_consol
        True: Movement ended by way of consolidation
        False: Movement ended by way of reversal.
        None: Movemment had not ended as at the end of the available data.

    by_rvr_by_pct
        True: Movement ended by way of reversal as a result of exceeding
        the reversal percentage limit.

        False: Movement ended by other means.

    rvr_max
        Maximum reversal percentage from the bar the movement start was
        confirmed, to the bar when the movement would have been considered
        to have ended by way of consolidation (or to end of available data
        if this is earlier).

    rvr_arr
        Array of percentage reversal limits, with first item representing
        the reversal limit on the bar the movement start was confirmed.
    """

    params: TrendParams
    sel: pd.Series
    start_conf_line: pd.Series
    eel: pd.Series | None
    end_line_consol: pd.Series
    end_line_rvr: pd.Series
    end_line_rvr_opp: pd.Series | None
    by_consol: bool | None
    by_rvr_by_pct: bool
    rvr_max: float
    rvr_arr: np.ndarray

    @property
    def open(self) -> bool:
        """Movement has not ended."""
        return self.by_consol is None

    @property
    def by_rvr(self) -> bool | None:
        """Movement ended by way of reversal.

        None if movement open.
        """
        return None if self.open else not self.by_consol

    @property
    def by_rvr_and_by_pct(self) -> bool | None:
        """Movement ended by way of reversal exceeding reversal limit."""
        return None if self.open else self.by_rvr and self.by_rvr_by_pct

    @property
    def by_rvr_and_by_opp(self) -> bool | None:
        """Movement ended by giving way to movement in opposite direction."""
        return None if self.open else self.by_rvr and not self.by_rvr_by_pct

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
        """Absolute change between start and end confirmation."""
        return None if self.open else self.end_conf_px - self.start_conf_px  # type: ignore

    @property
    def conf_chg_pct(self) -> float | None:
        """Percentage change between start and end confirmation."""
        return None if self.open else self.conf_chg / self.start_conf_px  # type: ignore

    @property
    def rvr(self) -> float | None:
        """Reversal limit percentage on bar movement confirmed as ended.

        None if movement did not end by reversal on exceeding reversal
        limit.
        """
        if not self.by_rvr_and_by_pct:
            return None
        arr = self.rvr_arr
        if len(arr) == 1:
            return arr[0]
        line = self.end_line_rvr
        if len(arr) < len(line):
            return arr[-1]
        i = line.index.get_loc(self.end_conf)
        return arr[i]


@dataclass(frozen=True)
class MovementsSupportChartAnaly(CasesSupportsChartAnaly, typing.Protocol):
    """Subprotocol for movements classes to display on a chart.

    Attributes
    ----------
    cases
        Ordered list of all movements identified in `data`.

    data
        OHLC data in which `moves` identified.

    interval
        Period represented by each row of `data`.
    """

    cases: Sequence[MovementBase]
    data: pd.DataFrame
    interval: pd.Timedelta | pd.DateOffset

    @property
    def advances(self) -> tuple[MovementBase, ...]:
        """Movements that represent advances."""
        ...

    @property
    def declines(self) -> tuple[MovementBase, ...]:
        """Movements that represent declines."""
        ...

    @property
    def starts_adv(self) -> pd.DatetimeIndex:
        """Start bars of advances."""
        ...

    @property
    def starts_dec(self) -> pd.DatetimeIndex:
        """Start bars of declines."""
        ...

    @property
    def ends_adv_solo(self) -> pd.DatetimeIndex:
        """End bars of advances that do not coincide with Starts.

        Bars of end of advances that do not coincide with the start bar
        of a subsequent movement (advance or decline).
        """
        ...

    @property
    def ends_dec_solo(self) -> pd.DatetimeIndex:
        """End bars of declines that do not coincide with Starts.

        Bars of end of declines that do not coincide with the start bar
        of a subsequent movement (advance or decline).
        """
        ...

    @property
    def starts_conf_adv(self) -> pd.DatetimeIndex:
        """Bars of advances when movement start confirmed."""
        ...

    @property
    def starts_conf_dec(self) -> pd.DatetimeIndex:
        """Bars of declines when movement start confirmed."""
        ...

    @property
    def starts_conf_adv_px(self) -> list[float]:
        """Prices when advance movements confirmed."""
        ...

    @property
    def starts_conf_dec_px(self) -> list[float]:
        """Prices when decline movements confirmed."""
        ...

    @property
    def ends_conf_adv(self) -> pd.DatetimeIndex:
        """Bars of advances when movement end confirmed."""
        ...

    @property
    def ends_conf_dec(self) -> pd.DatetimeIndex:
        """Bars of declines when movement end confirmed."""
        ...

    @property
    def ends_conf_adv_px(self) -> list[float]:
        """Prices when end of advance movements confirmed."""
        ...

    @property
    def ends_conf_dec_px(self) -> list[float]:
        """Prices when end of decline movements confirmed."""
        ...

    def handler_hover_start(self, mark: bq.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""
        ...

    def handler_hover_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing a movement end."""
        ...

    def handler_hover_conf_start(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing when movement start confirmed."""
        ...

    def handler_hover_conf_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing when movement end confirmed."""
        ...

    @property
    def trend(self) -> pd.Series:
        """Trend.

        `pd.Series` with index as bars (`pd.DatetimeIndex`) and values as
        corresponding trend value:
            1 advance
            0 consolidation
            -1 decline
        """
        ...

    def get_index_for_direction(self, case: MovementBase) -> int:
        """Get index position of a case in `advances` or `declines`"""
        ...


@dataclass(frozen=True)
class MovementsBase(CasesBase):
    """Movements identified over an analysis period.

    Fulfills `MovementsSupportChartAnaly` with exception of handlers which
    should be added by the subclass if it is intended that analysis will
    be charted.

    Attributes
    ----------
    cases
        Ordered list of all movements identified in `data`.

    data
        OHLC data in which `cases` identified.

    interval
        Period represented by each row of `data`.
    """

    cases: Sequence[MovementBase]
    data: pd.DataFrame
    interval: pd.Timedelta | pd.DateOffset

    def moves(self) -> Sequence[MovementBase]:
        """Alias for `cases` to maintain backwards compatibility."""
        return self.cases

    @cached_property
    def advances(self) -> tuple[MovementBase, ...]:
        """Movements that represent advances."""
        lst = [move for move in self.cases if move.is_adv]
        return tuple(lst)

    @cached_property
    def declines(self) -> tuple[MovementBase, ...]:
        """Movements that represent declines."""
        lst = [move for move in self.cases if move.is_dec]
        return tuple(lst)

    @property
    def starts_adv(self) -> pd.DatetimeIndex:
        """Start bars of advances."""
        return pd.DatetimeIndex([move.start for move in self.advances])

    @property
    def starts_dec(self) -> pd.DatetimeIndex:
        """Start bars of declines."""
        return pd.DatetimeIndex([move.start for move in self.declines])

    @property
    def starts_conf_adv(self) -> pd.DatetimeIndex:
        """Bar of advances when movement start confirmed."""
        return pd.DatetimeIndex([move.start_conf for move in self.advances])

    @property
    def starts_conf_dec(self) -> pd.DatetimeIndex:
        """Bar of declines when movement start confirmed."""
        return pd.DatetimeIndex([move.start_conf for move in self.declines])

    @property
    def starts_conf_adv_px(self) -> list[float]:
        """Prices when advance movements confirmed."""
        return [move.start_conf_px for move in self.advances]

    @property
    def starts_conf_dec_px(self) -> list[float]:
        """Prices when decline movements confirmed."""
        return [move.start_conf_px for move in self.declines]

    @property
    def ends_adv(self) -> pd.DatetimeIndex:
        """End bar of advances."""
        return pd.DatetimeIndex([move.end for move in self.advances if move.closed])

    @property
    def ends_dec(self) -> pd.DatetimeIndex:
        """End bar of declines."""
        return pd.DatetimeIndex([move.end for move in self.declines if move.closed])

    @property
    def ends_conf_adv(self) -> pd.DatetimeIndex:
        """Bar of advances when movement end confirmed."""
        return pd.DatetimeIndex(
            [move.end_conf for move in self.advances if move.closed]
        )

    @property
    def ends_conf_dec(self) -> pd.DatetimeIndex:
        """Bar of declines when movement end confirmed."""
        return pd.DatetimeIndex(
            [move.end_conf for move in self.declines if move.closed]
        )

    @property
    def ends_conf_adv_px(self) -> list[float]:
        """Prices when end of advance movements confirmed."""
        return [
            move.end_conf_px for move in self.advances if move.end_conf_px is not None
        ]

    @property
    def ends_conf_dec_px(self) -> list[float]:
        """Prices when end of decline movements confirmed."""
        return [
            move.end_conf_px for move in self.declines if move.end_conf_px is not None
        ]

    @property
    def starts(self) -> pd.DatetimeIndex:
        """Start bar of all movements."""
        return pd.DatetimeIndex([move.start for move in self.cases])

    @property
    def ends(self) -> pd.DatetimeIndex:
        """End bar of all movements."""
        return pd.DatetimeIndex([move.end for move in self.cases if move.closed])

    @property
    def ends_adv_solo(self) -> pd.DatetimeIndex:
        """End bar of advances that do not coincide with `starts`.

        Bar of end of advances that do not coincide with the start bar of a
        subsequent movement (advance or decline).
        """
        return self.ends_adv.difference(self.starts)

    @property
    def ends_dec_solo(self) -> pd.DatetimeIndex:
        """End bar of declines that do not coincide with `starts`.

        Bar of end of declines that do not coincide with the start bar of a
        subsequent movement (advance or decline).
        """
        return self.ends_dec.difference(self.starts)

    @property
    def trend(self) -> pd.Series:
        """Trend

        `pd.Series` with index as bars (`pd.DatetimeIndex`) and values as
        corresponding trend value:
            1 advance
            0 consolidation
            -1 decline
        """
        srs = pd.Series(0, index=self.data.index)
        for move in self.cases:
            stop = None if move.end is None else move.end - self.interval
            srs[move.start : stop] += move.trend
        return srs

    def get_index_for_direction(self, case: MovementBase) -> int:
        """Get index position of a case in `advances` or `declines`"""
        seq = self.advances if case.is_adv else self.declines
        return seq.index(case)

    def event_to_case(self, mark: bq.Scatter, event: dict) -> MovementBase:
        """Get case corresonding to an event for mark representing a case.

        Parameters as those passed to an event handler.
        """
        i = event["data"]["index"]
        case = self.advances[i] if mark.colors[0] == COL_ADV else self.declines[i]
        return case


@dataclass(frozen=True)
class Movements(MovementsBase, MovementsSupportChartAnaly):
    """Movements identified over an analysis period by `Trends`."""

    cases: Sequence[Movement]

    MARKER_MAP: typing.ClassVar = {
        "cross": "End",
        "circle": "Start conf",
        "square": "End conf",
    }

    def event_to_case(self, mark: bq.Scatter, event: dict) -> Movement:
        # Only for type checker to know move is a Movement as defined on this module.
        case = super().event_to_case(mark, event)
        if typing.TYPE_CHECKING:
            assert isinstance(case, Movement)
        return case

    @staticmethod
    def get_move_html(case: Movement) -> str:
        """Return html to describe a movement."""
        color = "crimson" if case.is_dec else "limegreen"
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>Start: " + formatter_datetime(case.start)
        s += f"<br>Start px: {formatter_float(case.start_px)}"
        s += f"<br>End: {'None' if case.end is None else formatter_datetime(case.end)}"
        s += f"<br>Chg: {formatter_percent(case.chg_pct)}"
        by = "None" if case.open else ("break" if case.by_break else "limit")
        s += f"<br>By: {by}"
        s += f"<br>Duration: {case.duration}"
        s += f"<br>Start conf: {formatter_datetime(case.start_conf)}"
        end_conf = (
            "None" if case.end_conf is None else formatter_datetime(case.end_conf)
        )
        s += f"<br>End conf: {end_conf}"

        s += "<br>Conf chg: "
        if case.open:
            s += "None"
        else:
            chg_color = "crimson" if case.conf_chg_pct < 0 else "limegreen"  # type: ignore
            chg_style = tooltip_html_style(color=chg_color, line_height=1.3)
            if typing.TYPE_CHECKING:
                assert case.conf_chg_pct is not None
            s += f"<span {chg_style}>{formatter_percent(case.conf_chg_pct)}</span>"

        s += "</p"
        return s

    def handler_hover_start(self, mark: bq.marks.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""
        case = self.event_to_case(mark, event)
        mark.tooltip.value = self.get_move_html(case)

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

    def handler_click_case(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking mark representing a movement start.

        Removes all existing scatters from figure.

        Adds scatters and lines for the single trend represented by the
        clicked scatter point.
        """
        case = self.event_to_case(mark, event)
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

        group = Groups.CASE
        # remove any existing marks for a selected movement.
        if group in chart.added_marks_groups:
            chart.remove_added_marks(group)

        if case.closed:
            if typing.TYPE_CHECKING:
                assert case.conf_chg_pct is not None
                assert case.end_conf_px is not None
            style = tooltip_html_style(color=color)
            s = f"<p {style}>Conf chg: "
            chg_color = COL_DEC if case.conf_chg_pct < 0 else COL_ADV  # type: ignore
            chg_style = tooltip_html_style(color=chg_color)
            s += f"<span {chg_style}>{formatter_percent(case.conf_chg_pct)}</span></p>"

            if case.is_adv:
                color_area = (
                    COL_ADV if case.start_conf_px < case.end_conf_px else COL_DEC
                )
            else:
                color_area = (
                    COL_ADV if case.start_conf_px > case.end_conf_px else COL_DEC
                )

            mark_chg = bq.Lines(
                x=[[case.start_conf, case.end_conf]] * 2,
                y=[[case.start_conf_px] * 2, [case.end_conf_px] * 2],
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
            mark.tooltip.value = self.get_move_html(case)

        f([case.start], [case.start_px], color, mark.marker, handler_start)

        handler = self._handler_not_start
        f([case.start_conf], [case.start_conf_px], color, "circle", handler)
        if case.closed:
            f([case.end], [case.end_px], color, "cross", handler)
            f([case.end_conf], [case.end_conf_px], color, "square", handler)

        color_break, color_limit = "white", "slategray"
        if not case.by_break:
            color_break, color_limit = color_limit, color_break
        fl(case.line_break, color_break, "dashed", "Break line")
        fl(case.line_limit, color_limit, "dashed", "Limit line")

        chart.hide_cases()
        chart.add_marks(marks, group)


@dataclass(frozen=True)
class MovementsAlt(MovementsBase, MovementsSupportChartAnaly):
    """Movements identified over an analysis period by `Trends`."""

    cases: Sequence[MovementAlt]

    MARKER_MAP: typing.ClassVar = {
        "cross": "End",
        "circle": "Start conf",
        "square": "End conf",
    }

    def event_to_case(self, mark: bq.Scatter, event: dict) -> MovementAlt:
        # Only for type checker to know move is a Movement as defined on this module.
        case = super().event_to_case(mark, event)
        if typing.TYPE_CHECKING:
            assert isinstance(case, MovementAlt)
        return case

    @staticmethod
    def get_move_html(case: MovementAlt) -> str:
        """Return html to describe a movement."""
        color = "crimson" if case.is_dec else "limegreen"
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>Start: " + formatter_datetime(case.start)
        s += f"<br>Start px: {formatter_float(case.start_px)}"
        s += f"<br>End: {'None' if case.end is None else formatter_datetime(case.end)}"
        s += f"<br>Chg: {formatter_percent(case.chg_pct)}"
        by = "None" if case.open else ("consol" if case.by_consol else "reversal")
        s += f"<br>By: {by}"
        s += f"<br>Duration: {case.duration}"
        if case.rvr is not None:
            s += f"<br>Rrv: {formatter_percent(case.rvr)}"
        s += f"<br>Rrv Max: {formatter_percent(case.rvr_max)}"
        s += f"<br>Start conf: {formatter_datetime(case.start_conf)}"
        end_conf = (
            "None" if case.end_conf is None else formatter_datetime(case.end_conf)
        )
        s += f"<br>End conf: {end_conf}"

        s += "<br>Conf chg: "
        if case.conf_chg_pct is None:
            s += "None"
        else:
            chg_color = "crimson" if case.conf_chg_pct < 0 else "limegreen"  # type: ignore
            chg_style = tooltip_html_style(color=chg_color, line_height=1.3)
            s += f"<span {chg_style}>{formatter_percent(case.conf_chg_pct)}</span>"

        s += "</p"
        return s

    def handler_hover_start(self, mark: bq.marks.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""
        case = self.event_to_case(mark, event)
        mark.tooltip.value = self.get_move_html(case)

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

    def handler_click_case(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking mark representing a movement start.

        Removes all existing scatters from figure.

        Adds scatters and lines for the single trend represented by the
        clicked scatter point.
        """
        case = self.event_to_case(mark, event)
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

        group = Groups.CASE
        # remove any existing marks for a selected case.
        if group in chart.added_marks_groups:
            chart.remove_added_marks(group)

        if case.closed:
            if typing.TYPE_CHECKING:
                assert case.conf_chg_pct is not None
                assert case.end_conf_px is not None
            style = tooltip_html_style(color=color)
            s = f"<p {style}>Conf chg: "
            chg_color = COL_DEC if case.conf_chg_pct < 0 else COL_ADV  # type: ignore
            chg_style = tooltip_html_style(color=chg_color)
            s += f"<span {chg_style}>{formatter_percent(case.conf_chg_pct)}</span></p>"

            if case.is_adv:
                color_area = (
                    COL_ADV if case.start_conf_px < case.end_conf_px else COL_DEC
                )
            else:
                color_area = (
                    COL_ADV if case.start_conf_px > case.end_conf_px else COL_DEC
                )

            mark_chg = bq.Lines(
                x=[[case.start_conf, case.end_conf]] * 2,
                y=[[case.start_conf_px] * 2, [case.end_conf_px] * 2],
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
            mark.tooltip.value = self.get_move_html(case)

        f([case.start], [case.start_px], color, mark.marker, handler_start)

        handler = self._handler_not_start
        f([case.start_conf], [case.start_conf_px], color, "circle", handler)
        if case.closed:
            f([case.end], [case.end_px], color, "cross", handler)
            f([case.end_conf], [case.end_conf_px], color, "square", handler)

        fl(case.sel, color, "dashed", "Start Establishment Line")
        fl(case.start_conf_line, color, "dash_dotted", "Confirmed Start Line")
        fl(
            case.end_line_consol,
            "white" if case.by_consol else "slategray",
            "dashed",
            "Conf End Line by Consolidation",
        )
        end_line_rvr_col = "white" if case.by_rvr_and_by_pct else "slategray"
        end_line_rvr = fl(
            case.end_line_rvr,
            end_line_rvr_col,
            "dash_dotted",
            "Conf End Line by Reversal",
        )

        # create labels for end_line_rvr
        rvr_arr = case.rvr_arr
        xs, ys, texts = [end_line_rvr.x[0]], [end_line_rvr.y[0]], [str(rvr_arr[0])]
        for i, (x, y, rvr_) in enumerate(
            zip(end_line_rvr.x[1:], end_line_rvr.y[1:], rvr_arr[1:])
        ):
            if rvr_ == rvr_arr[i]:
                continue
            xs.append(x)
            ys.append(y)
            texts.append(str(rvr_))
        label_mark = bq.Label(
            scales=chart.scales,
            x=xs,
            y=ys,
            text=texts,
            colors=[end_line_rvr_col],
            default_size=11,
            y_offset=15 if case.is_adv else -15,
            enable_move=True,
            restrict_y=True,
            visible=False,
        )
        marks.append(label_mark)

        def end_line_rvr_handler(*_):
            label_mark.visible = not label_mark.visible

        end_line_rvr.on_element_click(end_line_rvr_handler)

        if case.end_line_rvr_opp is not None:
            fl(
                case.end_line_rvr_opp,
                "white",
                "dash_dotted",
                "Conf End Line by Opposing Movement",
            )
        if case.eel is not None:
            fl(case.eel, color, "dotted", "End Establishment Line")

        chart.hide_cases()
        chart.add_marks(marks, group)
