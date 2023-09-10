"""Protocols and base classes to describe trend movements."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import typing

import bqplot as bq
import numpy as np
import pandas as pd

from market_analy.config import COL_ADV

if typing.TYPE_CHECKING:
    from market_analy.charts import OHLCTrends


@dataclass(frozen=True)
class MovementProto(typing.Protocol):
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

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return False
        for name in self.__dataclass_fields__.keys():
            v, v_other = getattr(self, name), getattr(other, name)
            if isinstance(v, pd.Series):
                try:
                    pd.testing.assert_series_equal(v, v_other)
                except AssertionError:
                    return False
            elif isinstance(v, np.ndarray):
                try:
                    assert (v == v_other).all()
                except AssertionError:
                    return False
            elif v != v_other:
                return False
        return True

    def __lt__(self, other):
        return self.start < other.start

    def __le__(self, other):
        return self.start <= other.start

    def __gt__(self, other):
        return self.start > other.start

    def __ge__(self, other):
        return self.start >= other.start

    @property
    def open(self) -> bool:
        """Movement has not ended."""

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


@dataclass
class MovementsChartProto(typing.Protocol):
    """Protocol for 'movements' param of `charts.OHLCTrends`.

    Attributes
    ----------
    moves
        Ordered list of all movements identified in `data`, where each
        movement confirms with `MovementProto`.

    data
        OHLC data in which `moves` identified.

    interval
        Period represented by each row of `data`.
    """

    moves: list[MovementProto]
    data: pd.DataFrame
    interval: pd.Timedelta | pd.DateOffset

    @property
    def starts_adv(self) -> pd.DatetimeIndex:
        """Start bars of advances."""

    @property
    def starts_dec(self) -> pd.DatetimeIndex:
        """Start bars of declines."""

    @property
    def ends_adv_solo(self) -> pd.DatetimeIndex:
        """End bars of advances that do not coincide with Starts.

        Bars of end of advances that do not coincide with the start bar
        of a subsequent movement (advance or decline).
        """

    @property
    def ends_dec_solo(self) -> pd.DatetimeIndex:
        """End bars of declines that do not coincide with Starts.

        Bars of end of declines that do not coincide with the start bar
        of a subsequent movement (advance or decline).
        """

    @property
    def starts_conf_adv(self) -> pd.DatetimeIndex:
        """Bars of advances when movement start confirmed."""

    @property
    def starts_conf_dec(self) -> pd.DatetimeIndex:
        """Bars of declines when movement start confirmed."""

    @property
    def starts_conf_adv_px(self) -> list[float]:
        """Prices when advance movements confirmed."""

    @property
    def starts_conf_dec_px(self) -> list[float]:
        """Prices when decline movements confirmed."""

    @property
    def ends_conf_adv(self) -> pd.DatetimeIndex:
        """Bars of advances when movement end confirmed."""

    @property
    def ends_conf_dec(self) -> pd.DatetimeIndex:
        """Bars of declines when movement end confirmed."""

    @property
    def ends_conf_adv_px(self) -> list[float]:
        """Prices when end of advance movements confirmed."""

    @property
    def ends_conf_dec_px(self) -> list[float]:
        """Prices when end of decline movements confirmed."""

    def handler_hover_start(self, mark: bq.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""

    def handler_hover_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing a movement end."""

    def handler_hover_conf_start(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing when movement start confirmed."""

    def handler_hover_conf_end(self, mark: bq.Scatter, event: dict):
        """Handler for hovering over mark representing when movement end confirmed."""

    def handler_click_trend(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking on mark representing a movement start."""

    def mark_to_move(self, mark: bq.Scatter, event: dict):
        """Get movement corresonding to a mark representing a trend start.

        Parameters as those passed to an event handler.
        """

    @property
    def trend(self) -> pd.Series:
        """Trend.

        `pd.Series` with index as bars (`pd.DatetimeIndex`) and values as
        corresponding trend value:
            1 advance
            0 consolidation
            -1 decline
        """


@dataclass
class MovementsBase:
    """Movements identified over an analysis period.

    Fulfills `MovementsChartProto` with exception of handlers which should
    be added by the subclass if it iss intended that instances will be
    passed to the 'movements' parameter of `charts.OHLCTrends`.

    Attributes
    ----------
    moves
        Ordered list of all movements identified in `data`.

    data
        OHLC data in which `moves` identified.

    interval
        Period represented by each row of `data`.
    """

    moves: list[MovementProto]
    data: pd.DataFrame
    interval: pd.Timedelta | pd.DateOffset

    @cached_property
    def advances(self) -> list[MovementProto]:
        """Movements that represent advances."""
        return [move for move in self.moves if move.is_adv]

    @cached_property
    def declines(self) -> list[MovementProto]:
        """Movements that represent declines."""
        return [move for move in self.moves if move.is_dec]

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
        return pd.DatetimeIndex([move.start for move in self.moves])

    @property
    def ends(self) -> pd.DatetimeIndex:
        """End bar of all movements."""
        return pd.DatetimeIndex([move.end for move in self.moves if move.closed])

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
        for move in self.moves:
            stop = None if move.open else move.end - self.interval
            srs[move.start : stop] += move.trend
        return srs

    def get_index(self, move: MovementProto, direction: bool = False) -> int | None:
        """Get index position of a movement.

        None if `move` not in `self.moves`.

        Parameters
        ----------
        move
            Movement to query

        direction
            False: return index of `move` in `.moves`

            True: return index of `move` in either `.advances` or
            `.declines`.
        """
        is_adv = move.is_adv
        seq = (self.advances if is_adv else self.declines) if direction else self.moves
        for i, m in enumerate(seq):
            if m == move:
                return i
        return None

    def mark_to_move(self, mark: bq.Scatter, event: dict) -> MovementProto:
        """Get movement corresonding to a mark representing a trend start.

        Parameters as those passed to an event handler.
        """
        i = event["data"]["index"]
        move = self.advances[i] if mark.colors[0] == COL_ADV else self.declines[i]
        return move
