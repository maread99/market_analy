"""Trend Analysis.

Classes to evaluate and store trends. Trends evalutaed from OHLC data.

    Trends
        Broadly, a trend is considered to exist whenever a movement
        (advance or decline) is continually extended, by way of exceeding
        a limit line, whilst at the same time NOT exceeding a break line
        that underpins the trend. See `Trends.__doc__` for more specific
        documentation.

    TrendsAlt
        Broadly, a trend is considered to exist when the price has extended
        the high/low of a given period by a given percentage and has not
        subsequently 'lost' more than a given reversal limit. See
        `TrendsAlt.__doc__` for more specific documentation.
"""

from __future__ import annotations

import functools
import operator
import typing
from collections import abc
from dataclasses import dataclass
from functools import cached_property

import market_prices as mp
import numpy as np
import pandas as pd

from .movements import Movement, MovementAlt, Movements, MovementsAlt


class TrendParams(typing.TypedDict):
    """Trend Parameters.

    Trend parameters for a movement identified with `Trends`.
    """

    interval: mp.intervals.RowInterval
    prd: int
    ext_break: float
    ext_limit: float
    min_bars: int


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

            pd.Timedelta(15, "min") (or "15min"): each row of data
            represents 15 minutes

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
        interval of 30 minutes (i.e. `interval` is "30min") then passing 10
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
            if move.end is None:
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

        return Movements(tuple(moves), self.data, self.interval)


@dataclass(frozen=True)
class RvrAlt:
    """Parameters for ALTERNATIVE definition of reversal percent limits.

    Parameters provide for defining reversal percent limits according to
    the maximum extent of the movement as a factor of the instrument's
    'usual' change over `prd` number of bars. The 'usual' change is
    evaluated:
        For advances, as the median positive "change over prd" over
        the full extent of the data set.

        For declines, as the median negative "change over prd" over
        the full extent of the data set.

    Parameters
    ----------
    multiples : list[int | float]
        Multiples of the 'usual' change up to which to the reversal percent
        limit defined by the corresponding index of `rvr` should be
        applied (see `rvr`).

    rvr : list[float]
        Reversal percent limits that should be applied when the movement's
        maximum extent, as a multiple of the 'usual' change, is less than
        the value of the corresponding index of `multiples`. List should
        have length one greater than the length of `multiples` with the last
        value defining the reversal limit to apply beyond the last multiple
        of `multiples`.

        For example, if the 'usual' positive change over prd is 2% and...
            multiples = (1.5, 2.5, 4, 6)
            rvr = (0.8, 0.6, 0.4, 0.2, 0.15)
        ...then the reversal limit would be 0.8* until an advance exceeds 3%
        (i.e. 2% * the corresponding 1.5 factor), thereafter 0.6 until an
        advance exceeds 5% (2% * 2.5), thereafter 0.4 until the advance
        exceeds 8%, thereafter 0.2 until the advance exceeds 12% and
        thereafter 0.15.

        * i.e. the advance would be considered over if the movement were to
        'give back' 80% of it's maximum extent since the movement was
        confirmed.

    start_limit : tuple[float, int] | None, default: None
        Defines a minimum reversal limit to apply over the start of the
        movement. 2-tuple describing:
            [0] : float. Minimum reversal limit that should be applied
            over the period described by [1].

            [1] : int. Number of bars, from and inclusive of the bar on
            which the movement was confirmed, over which the start limit
            should apply.

    end_limit : tuple[int, float] | None, default: None
        Defines a maximum reversal limit to apply over the 'end' of the
        movement. 2-tuple describing:
            [0] : float. Maximum reversal limit that should be applied
            over the period described by [1].

            [1] : int. Number of bars, from and inclusive of the bar on
            which the movement was confirmed, after which the end limit
            should be applied. If `start_limit` passed then `end_limit`[1]
            must by higher than `start_limit`[1].
    """

    multiples: list[int | float]
    rvr: list[float]
    start_limit: tuple[float, int] | None = None
    end_limit: tuple[float, int] | None = None

    def __post_init__(self):
        if len(self.rvr) != len(self.multiples) + 1:
            raise ValueError(
                "`rvr` must have a length one greater than the length of `multiples`."
            )
        if self.start_limit is not None and self.end_limit is not None:
            if not self.end_limit[1] > self.start_limit[1]:
                raise ValueError("`end_limit` must commence after `start_limit`.")


class TrendAltParams(typing.TypedDict):
    """Trend Parameters.

    Trend parameters for a movement identified with `Trends`.
    """

    interval: mp.intervals.RowInterval
    prd: int
    ext: float
    rvr: list | RvrAlt
    rvr_init: np.ndarray
    grad: float | None
    min_bars: int


class TrendsAlt:
    """Evaluate trends over OHLC data.

    --Movement Start Confirmed Bar--
    A movement is confirmed as having started on a bar if price extends
    high/low of prior `prd` by `ext`. Specifically, an advance/decline is
    confirmed if the price exceeds, for the first time, a Limit Line which
    increases/decreases linearly and is defined:
        TO the bar on which movement could be confirmed, at the price
        that the movement could be confirmed.

        FROM `prd` bars prior to the 'TO' bar, at a price of `ext` %
        below/above the 'TO' price.

    --Start Bar--
    The bar on which an advance/decline starts is, in the first instance,
    considered as the low/high of the `prd` bars prior to the bar on which
    the movement was confirmed. If, in accordance with the initial reversal
    limits (`rvr_init`), the movement would be considered to have ended
    before it was confirmed then the start bar will be advanced to the
    first subsequent low/high from where the movement would not be
    considered to have ended prior to the start confirmation bar.

    Separately, see 'Notes' section for notes for circumstances that can
    result in a start bar being defined later than would seen most
    appropriate.

    --Movement End Confirmed Bar--
    A movement is confirmed as having ended by way of:
        Consolidation - on any bar when the criteria to consider a movement
        to have started is not met over any of the prior `prd` bars.

        Reversal - in accordance with the reversal limits set by `rvr`.

    --End Bar--
    The bar on which an advance/decline ends is, in the first instance,
    considered as the high/low of the `prd` bars prior to the bar on which
    the movement end was confirmed. If the movement ended by way of
    consolidation then this end bar will be moved back to the earliest of
    any bar that exceeds the End Establishment Line (EEL) over the prior
    `prd` bars (see `grad` parameter).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns that include "high", "low" and "close". Any
        other columns will be ignored.

        DataFrame must be indexed with a `pd.DatetimeIndex`.

    interval : pd.Timedelta | str
        Interval that each row of `data` represents. Examples:

            pd.Timedelta(15, "min") (or "15min"): each row of data
            represents 15 minutes

            pd.Timedelta(1, "D") (or "1D"): for daily data.

    prd : int
        Period over which extreme should have been advanced by no less than
        `ext` in order for a movement to be considered to having started
        or be continuing. Passed in terms of number of rows of `data` (i.e.
        bars on a OHLC chart). For example, if each row of `data`
        represents one session (i.e. if `interval` is "1D") then passing 10
        will set the period as 10 sessions, whilst if each row represents
        an interval of 30 minutes (i.e. `interval` is "30min") then passing
        10 will set the period as 5 hours.

    ext : float
        If advance/decline fails to extend the high/low of the previous
        `prd` by `ext` then movement will be considered to have ended
        by way of consolidation. Pass as a float, for example 0.005 for
        0.5%. Extensions of the high/low within `prd` are considered on
        a pro-rata basis (see 'Limit Line' comments within the introduction
        to the class doc).

    rvr : float | abc.Sequence[float] | RvrAlt
        When passing as a float or sequence of float:

            A movement will be considered to have ended by way of reversal
            on any counter movement greater than `rvr` percent of the trend
            to date.

            Pass as a float to define as a single static percentage. For
            example, 0.5 will result in a movement that's advanced 12% to
            be continuing if it were to fall 5.9% from it's high, although
            the movement would be considered to have ended by way of
            reversal if the price were to fall to 6% under the movement
            high (i.e. 50% of the movement would have been given up).

            Alternatively, pass as a sequence of float to define a
            differing percentage for every bar following the bar when the
            movement is confirmed, for example [0.8, 0.8, 0.8, 0.7, 0.7,
            0.6. 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4 0.4, 0.3]
            would consider the advance to have ended by way of reversal if
            during the three bars following the movements confirmation the
            price 'gave up' more than 80% of the trend to date, whilst over
            the following 2 bars only 70% of the trend would need given up
            in order for the trend to be confirmed as having given way to
            reversal, 60% over the next 2 bars, 50% over the next 5, 40%
            over the next 5 and then 30% from then on (the last value will
            be considered static henceforth).

        If passing as RvrAlt:

            A movement will be considered to have ended by way of reversal
            on any counter movement greater than reversal limits evaluated
            according to the extent of the movement. The extent of the
            movement is evaluated in terms of a "multiple of the
            instrument's usual change over `prd`", where 'usual' is
            evaluated as:
                For advances, the median positive "change over `prd`" over
                the full extent of the data set.

                For declines, the median negative "change over `prd`" over
                the full extent of the data set.

            The `RvrAlt` dataclass provides for defining the multiples and
            corresponding reversal limits together with, optionally,
            minimum / maximum reversal limits to apply over the start / end
            of a movement.

    rvr_init : float | tuple | list
        Reversal limits for the period prior to confirmation that a
        movement has started. These limits determine the start bar such
        that from the bar after the start through to the confirmed start
        the price will not have reversed by more than `rvr_init`.

        Similar to `rvr`, pass as a float to define a static percentage to
        apply throughout this initial period, or pass as a list to define
        a differing percentage for every bar following the start bar. If
        the list is longer than the number of bars from the bar after the
        assessed start through to the confirmed start then the excess
        elements at the end of the list will be ignored. If the list is
        shorter than the number of bars then the last element of the list
        will be repeated.

        Alternatively, pass as a 2-tuple of floats to define a value for
        the bar after the start and a value for a bar `prd` - 1 bars
        later. A list of length `prd` will be created with the values in
        between evaluated by linear interpolation. Whenever this list is
        longer than the number of bars between an assessed start and the
        confirmed start the excess elements at the end of the list will be
        ignored.

    grad : float | None
        Defines gradient of End Establishment Line (EEL).

        Default (if None), `ext` / (`prd` * 2)

        A movement's end bar is determined as the earliest bar that the
        price touches the EEL, where the EEL is defined as a line that:

            - end on the high/low of the advance/decline as registered
            over the `prd` bars prior to the bar on which the movement
            end is confirmed.

            - has a length `prd` + 1 bars.

            - has a gradient such that for an advance/decline the line
            rises/falls by `grad` every bar.

    min_bars : int, default: 0
        The minimum number of bars that should comprise any movement.

    Notes
    -----
    The start should be earlier!
    Sometimes the start bar of a trend appears later than would seem most
    appropriate. For example, the start of an advance might be set a couple
    of bars after a lower low. This is usually because subsequently the
    movement is not extended by `ext` over `prd` bars - the criteria to
    confirm a movement's end cannot be fulfiled 'during' an ongoing
    movement! Integrity is important.

    If a notable number of movements are 'starting late', consider
    investigating if there are more suitable parameters to represent the
    instrument.

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
        ext: float,
        rvr: float | abc.Sequence[float] | RvrAlt,
        rvr_init: float | tuple[float, float] | list[float],
        grad: float | None,
        min_bars: int = 0,
    ):
        self.data = data.copy()
        self.interval = mp.intervals.to_ptinterval(interval)
        self.prd = prd
        self.ext = ext
        self.grad = grad if grad is not None else self.ext / (self.prd * 2)
        self.min_bars = min_bars

        self.rvr: list[float] | RvrAlt
        if isinstance(rvr, RvrAlt):
            self.rvr = rvr
        else:
            self.rvr = [rvr] if isinstance(rvr, (float, int)) else list(rvr)

        self.rvr_init: np.ndarray  # set as array of len no less than prd
        if isinstance(rvr_init, tuple):
            self.rvr_init = np.linspace(rvr_init[0], rvr_init[-1], prd)
        elif isinstance(rvr_init, (float, int)):
            self.rvr_init = np.array([rvr_init] * prd)
        else:
            if len(rvr_init) < prd:
                rvr_init = rvr_init + ([rvr_init[-1]] * (prd - len(rvr_init)))
            self.rvr_init = np.array(rvr_init)

    @property
    def params(self) -> TrendAltParams:
        """Parameters received by constructor."""
        return {
            "interval": self.interval,
            "prd": self.prd,
            "ext": self.ext,
            "rvr": self.rvr,
            "rvr_init": self.rvr_init,
            "grad": self.grad,
            "min_bars": self.min_bars,
        }

    @cached_property
    def factors_break_pos(self) -> np.ndarray:
        """Positive factors to identify break, increasing left to right."""
        return np.concatenate(
            [np.array([1]), np.linspace(1, 1 + self.ext, self.prd - 1)]
        )

    @cached_property
    def factors_break_neg(self) -> np.ndarray:
        """Negative factors to identify break, decreasing left to right."""
        return np.concatenate(
            [np.array([1]), np.linspace(1, 1 - self.ext, self.prd - 1)]
        )

    @cached_property
    def factors_break_pos_rev(self) -> np.ndarray:
        """Positive factors to identify break, increasing right to left."""
        return self.factors_break_pos[::-1]

    @cached_property
    def factors_break_neg_rev(self) -> np.ndarray:
        """Negative factors to identify break, decreasing right to left."""
        return self.factors_break_neg[::-1]

    @cached_property
    def grad_factors_pos(self) -> list[float]:
        """Positive gradient factors, increasing from left to right."""
        return [1, 1] + [1 + (x * self.grad) for x in range(3, self.prd + 1)]

    @cached_property
    def grad_factors_neg(self) -> list[float]:
        """Negative gradient factors, decreasing from left to right."""
        return [1, 1] + [1 - (x * self.grad) for x in range(3, self.prd + 1)]

    @cached_property
    def grad_factors_pos_rev(self) -> list[float]:
        """Positive gradient factors, increasing right to left."""
        return list(reversed(self.grad_factors_pos))

    @cached_property
    def grad_factors_neg_rev(self) -> list[float]:
        """Negative gradient factors, increasing right to left."""
        return list(reversed(self.grad_factors_neg))

    @staticmethod
    def get_reversals(
        px_start: float,
        df: pd.DataFrame,
        is_adv: bool,
        limits: np.ndarray | None = None,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Get percentage reversals from high/low relative to a start price.

        Parameters
        ----------
        px_start
            Start price against which to measure movements and subsequent
            reversals.

        df
            pd.DataFrame covering period to assess. First row should be the
            row that immediately follows the row corresponding with
            `px_start`.

        is_adv
            True: movement represents an advance
            False: movement represents a decline

        limits
            List of same length as `df` describing reversal limits for each
            bar.

            Optional, if passed then return will include a pd.Series
            describing a line that tracks these limits.

        Returns
        -------
        rvr : pd.Series
            pd.Series with index as `df` and values as reversal from
            current high/low.

        line : pd.Series
            Only returned if `limits` passed.

            pd.Series, with index as `df` describing a line that tracks
            `limits`.
        """
        end_cur = df.high.cummax() if is_adv else df.low.cummin()
        end_rvr = df.low.copy() if is_adv else df.high.copy()
        # for intervals where end_cur has been moved out, not possible to
        # know if the high/low came before the new low/high or afterwards,
        # given which assumes counter movement reached only interval close.
        bv = end_cur != end_cur.shift(1)
        end_rvr[bv] = df.close[bv]
        chg = end_cur - px_start if is_adv else px_start - end_cur
        chg_rvr = (end_rvr - end_cur).abs()
        rvr = chg_rvr / chg
        if limits is None:
            return rvr
        line = px_start + (chg * ((1 - limits) * (1 if is_adv else -1)))
        return rvr, line

    def get_start(
        self,
        minstart: pd.Timestamp,
        start_conf: pd.Timestamp,
        is_adv: bool,
    ) -> tuple[pd.Timestamp, float, pd.Series]:
        """Get the start date of a potential movement.

        Parameters
        ----------
        minstart
            Returned start will be no earlier than `minstart`.

        start_conf
            The bar on which the start of the movement is confirmed.

        is_adv
            True: movement would represent an advance.
            False: movement would represent a decline.

        Returns
        -------
        3-tuple of [pd.Timestamp | pd.Interval, float | pd.Series]
            [0] Bar that would represent start of movement.
            [1] Price at what would be start of movement.
            [2] Start establishement line.

        Notes
        -----
        Initially assumes start as the lowest/highest price of the
        `self.prd` period prior to the bar on which the movement was
        confirmed.

        If with this start the movement would fulfil the critiera to
        confirm the end of the movememnt before the confirmed start then
        moves the start forwards to a bar following which the criteria to
        confrim the end of the movement is not met.
        """
        index = self.data.index
        idx_conf = index.get_loc(start_conf)
        idx_min = 0 if minstart not in index else index.get_loc(minstart)

        idx_frm = max(idx_min, idx_conf - self.prd)
        subset = self.data.iloc[idx_frm : idx_conf + 1]
        _idx_start = subset.low.argmin() if is_adv else subset.high.argmax()
        row = subset.iloc[_idx_start]
        idx_start = index.get_loc(row.name)

        px_start = row.low if is_adv else row.high
        subset = self.data.iloc[idx_start + 1 : idx_conf + 1]  # <= prd len subset
        rvr_limits = self.rvr_init[: len(subset)]
        rvr, line = self.get_reversals(px_start, subset, is_adv, rvr_limits)
        bv = rvr.values > rvr_limits
        while bv.any():
            _idx_start = bv.argmax()
            row = subset.iloc[_idx_start]
            px_start = row.low if is_adv else row.high
            subset = subset[_idx_start + 1 :]
            rvr_limits = self.rvr_init[: len(subset)]
            rvr, line_ = self.get_reversals(px_start, subset, is_adv, rvr_limits)
            line = pd.concat([line[: -len(line_)], line_])
            bv = rvr.values > rvr_limits

        return row.name, row.low if is_adv else row.high, line

    def find_movement(
        self, df: pd.DataFrame, minstart: pd.Timestamp, minstart_conf: pd.Timestamp
    ) -> (
        tuple[bool, pd.Timestamp, float, pd.Series, pd.Timestamp, float, pd.Series]
        | None
    ):
        """Identify the first movement to occur.

        Identifies the first movement to be confirmed no earlier than
        `minstart_conf`.

        An advance/decline is considered to be confirmed on a bar when the
        price is greater/lesser than all values of the prior `self.prd` - 1
        bars and has extended those prior highs/lows by a greater degree
        than would be considered as consolidation (i.e. as determined by
        having exceeded the Limit Line - see introduction to class doc).

        Parameters
        ----------
        df
            pd.DataFrame with rows representing bar interval and columns as
            OHLC. Method makes no changes to this dataframe.

        minstart
            Returned movement will have a start bar no earlier than
            `minstart`.

        minstart_conf
            Returned movement will have a confirmation bar no earlier than
            `minstart_conf`. Cannot be earlier than `minstart`.

        Returns
        -------
        None | 7-tuple of
            [bool, pd.Timestamp, float, pd.Series, pd.Timestamp, float,
            pd.Series]:
            [0] Movement direction, True if an advance, False if a decline.
            [1] Bar representing movement's start.
            [2] Price at start of movement.
            [3] Start establishement line.
            [4] Bar when movement broke consolidation of prior `self.prd`.
            [5] Break price of prior consolidation.
            [6] Line indicating limit of broken consolidation range.

            None if no movement is identified.
        """
        new_high = df.high == df.high.rolling(self.prd, min_periods=0).max()
        new_low = df.low == df.low.rolling(self.prd, min_periods=0).min()

        # only consider as a movement if register new prd high/low but NOT if register
        # both a new prd high and new prd low...possible for both to register on the
        # same interval if the range of the last prd falls fully within the interval's
        # range. In this case wait for the trend to be clarified.
        new_high_xor_low = (new_high & ~new_low) | (new_low & ~new_high)
        new_high_xor_lows = df.index[
            new_high_xor_low & (new_high_xor_low.index >= minstart_conf)
        ]

        for high_xor_low in new_high_xor_lows:
            is_adv = new_high[high_xor_low]
            is_dec = not is_adv
            idx_conf = df.index.get_loc(high_xor_low)
            row = df.loc[high_xor_low]
            px = row.high if is_adv else row.low
            fctrs = self.factors_break_pos_rev if is_adv else self.factors_break_neg_rev
            subset = df.iloc[idx_conf - self.prd : idx_conf]
            vals = fctrs * (subset.high if is_adv else subset.low)

            if (is_adv and not (vals > px).any()) or (is_dec and not (vals < px).any()):
                start, start_px, sel = self.get_start(minstart, high_xor_low, is_adv)
                idx_start = df.index.get_loc(start)
                if idx_conf - idx_start + 1 < self.min_bars:
                    continue

                break_px = vals.max() if is_adv else vals.min()
                conf_px = max(break_px, row.open) if is_adv else min(break_px, row.open)
                f = self.factors_break_neg_rev if is_adv else self.factors_break_pos_rev
                line = pd.Series(f, subset.index) * break_px
                line[high_xor_low] = break_px
                return is_adv, start, start_px, sel, high_xor_low, conf_px, line
        return None

    def _find_adv(
        self, df: pd.DataFrame, minstart_conf: pd.Timestamp | None = None
    ) -> tuple[pd.Timestamp | pd.Interval, float, pd.Series] | tuple[None, None, None]:
        """Identify the first advance to occur.

        Returns
        -------
        3-tuple of
            [pd.Timestamp, float, pd.Series] | [None, None, None]:
            [0] Bar of `df` when movement broke consolidation of the prior
                `self.prd`.
            [1] break price of prior consolidation (or nearest traded
                price).
            [2] line indicating limit of broken consolidation range.

            (None, None, None) if no advance is identified.

        Notes
        -----
        Skimmed down version of `.find_movement` concerned only with
        identifying the next advance.
        """
        new_high = df.high == df.high.rolling(self.prd, min_periods=0).max()
        new_highs = df.index[new_high]
        if minstart_conf is not None:
            new_highs = new_highs[new_highs >= minstart_conf]

        for high in new_highs:
            row = df.loc[high]
            px = row.high
            idx_conf = df.index.get_loc(high)
            subset = df.iloc[idx_conf - self.prd : idx_conf]
            vals = self.factors_break_pos_rev * subset.high
            if not (vals > px).any():
                start, _, _ = self.get_start(minstart_conf, high, True)
                idx_start = df.index.get_loc(start)
                if idx_conf - idx_start + 1 < self.min_bars:
                    continue
                break_px = vals.max()
                fctrs = self.factors_break_neg_rev
                line = pd.Series(fctrs, subset.index) * break_px
                line[high] = break_px
                return high, max(break_px, row.open), line
        return None, None, None

    def _find_dec(
        self, df: pd.DataFrame, minstart_conf: pd.Timestamp | None = None
    ) -> tuple[pd.Timestamp, float, pd.Series] | tuple[None, None, None]:
        """Identify the first decline to occur.

        Returns
        -------
        3-tuple of
            [pd.Timestamp, float, pd.Series] | [None, None, None]:
            [0] Bar of `df` when movement broke consolidation of the prior
                `self.prd`.
            [1] break price of prior consolidation (or nearest traded
                price).
            [2] line indicating limit of broken consolidation range.

            (None, None, None) if no decline is identified.

        Notes
        -----
        Skimmed down version of `.find_movement` concerned only with
        identifying the next decline.
        """
        new_low = df.low == df.low.rolling(self.prd, min_periods=0).min()
        new_lows = df.index[new_low]
        if minstart_conf is not None:
            new_lows = new_lows[new_lows >= minstart_conf]

        for low in new_lows:
            row = df.loc[low]
            px = row.low
            idx_conf = df.index.get_loc(low)
            subset = df.iloc[idx_conf - self.prd : idx_conf]
            vals = self.factors_break_neg_rev * subset.low
            if not (vals < px).any():
                start, _, _ = self.get_start(minstart_conf, low, False)
                idx_start = df.index.get_loc(start)
                if idx_conf - idx_start + 1 < self.min_bars:
                    continue
                break_px = vals.min()
                fctrs = self.factors_break_pos_rev
                line = pd.Series(fctrs, subset.index) * break_px
                line[low] = break_px
                return low, min(break_px, row.open), line
        return None, None, None

    def _conf_end_rvr_opp(
        self, is_adv: bool, conf_start: pd.Timestamp
    ) -> tuple[pd.Timestamp, float, pd.Series] | tuple[None, None, None]:
        """Get info for next movement in opposing direction.

        Get info for when a movement would end as a result of giving way to
        a movement being confirmed in the opposite direction.

        Parameters
        ----------
        conf_start
            Interval when movement confirmed.

        is_adv
            True: movement represents an advance.
            false: movement represents a decline.

        Returns
        -------
        3-tuple of
            [pd.Timestamp, float, pd.Series] | [None, None, None]:
            [0] Bar when opposing movement confirmed.
            [1] Price when opposing movement confirmed.
            [2] Line indicating limit of broken consolidation range.

            (None, None, None) if no opposing movement is identified.
        """
        i = self.data.index.get_loc(conf_start) - self.prd
        df = self.data.iloc[i:]
        f = self._find_dec if is_adv else self._find_adv
        return f(df, conf_start)

    def _prd_chgs(self) -> pd.Series:
        df = self.data
        opens = df.open
        closes = df.close.shift(-self.prd + 1)
        return (closes - opens) / opens

    @cached_property
    def _av_chg_pos(self) -> float:
        srs = self._prd_chgs()
        return srs[srs > 0].median()

    @cached_property
    def _av_chg_neg(self) -> float:
        srs = self._prd_chgs()
        return srs[srs < 0].median()

    def _get_rvr_arr(
        self, is_adv: bool, end_cur: pd.Series, px_start: float
    ) -> np.ndarray:
        """Get array of reversal percent limits.

        Array will be as long as `end_cur` and correspond with the bars as
        described by the index of `end_cur`.

        Parameters
        ----------
        is_adv
            True: movement being evaluated represents an advance.
            false: movement being evaluated represents a decline.

        end_cur : pd.Series
            Index, dtype 'datetime64[ns]', representing bars over which
            require reversal limits. First row should represent the bar on
            which the movement was confirmed.

        px_start : float
            Price at start of movement.
        """
        if not isinstance(self.rvr, RvrAlt):
            arr = np.array([self.rvr[-1]] * len(end_cur))
            lngth = min(len(self.rvr), len(end_cur))
            arr[:lngth] = self.rvr[:lngth]
            return arr

        av = self._av_chg_pos if is_adv else abs(self._av_chg_neg)
        breaks = [0] + [m * av for m in self.rvr.multiples] + [np.inf]
        index = pd.IntervalIndex.from_breaks(breaks, closed="left")
        srs = pd.Series(self.rvr.rvr, index=index)

        move_pct = ((end_cur - px_start).abs()) / px_start
        mapping = {chg: srs[chg] for chg in move_pct.unique()}
        arr = move_pct.map(mapping).values

        if self.rvr.start_limit is not None:
            subset = arr[: self.rvr.start_limit[1]]
            min_limit = self.rvr.start_limit[0]
            subset[subset < min_limit] = min_limit
        if self.rvr.end_limit is not None:
            subset = arr[self.rvr.end_limit[1] :]
            max_limit = self.rvr.end_limit[0]
            subset[subset > max_limit] = max_limit
        return arr

    def _conf_end_rvr_pct(
        self,
        is_adv: bool,
        start: pd.Timestamp,
        conf_start: pd.Timestamp,
        end_alt: pd.Timestamp | None,
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series, float, np.ndarray]:
        """Get info for when movement would be confirmed as ending by exceeding rvr.

        Parameters
        ----------
        end_alt
            Earliest date by which movement would end by any other means.

        Returns
        -------
        5-tuple of [pd.Timestamp | None, float | None, pd.Series, float, np.ndarray]
            [0] Bar on which movement would end by reversal. None if
                movement would not be considered to have given way to
                reversal as at `end_alt`.
            [1] Price at reversal limit (or nearest traded price to). None
                if movement would not be considered to have exceeded
                reversal limit as at end of available data.
            [2] Line representing reversal limit.
            [3] Maximum reversal within period before when would have, in
                any event, been considered to have ended by consolidation.
                If movement had not ended by end of available data then
                maximum reversal as at end of available data.
            [4] Array describing reversal limits. First value corresponds
                with reversal limit on bar when movement was confirmed.
        """
        row = self.data.loc[start]
        px_start = row.low if is_adv else row.high

        df = self.data.loc[conf_start:end_alt]
        end_cur = df.high.cummax() if is_adv else df.low.cummin()
        end_rvr = df.low.copy() if is_adv else df.high.copy()
        # for intervals where end_cur has been moved out, not possible to
        # know if the high/low came before the new low/high or afterwards,
        # given which assumes counter movement reached only interval close.
        bv = end_cur != end_cur.shift(1)
        end_rvr[bv] = df.close[bv]
        end_cur, end_rvr = end_cur[1:], end_rvr[1:]  # do not include conf_start

        chg = end_cur - px_start if is_adv else px_start - end_cur

        chg_rvr = (end_rvr - end_cur).abs()
        rvr = chg_rvr / chg
        rvr_max = rvr.max()

        rvr_arr = self._get_rvr_arr(is_adv, end_cur, px_start)
        line = px_start + (chg * ((1 - rvr_arr) * (1 if is_adv else -1)))
        bv = end_rvr < line if is_adv else end_rvr > line
        if not bv.any():
            return None, None, line, rvr_max, rvr_arr

        idx = bv.argmax()
        stop = idx + 1 if end_alt is None else min(idx + 1, len(bv))
        line_ = line.iloc[:stop]

        px_break = line.iloc[idx]
        row = df.iloc[idx + 1]
        px = min(px_break, row.open) if is_adv else max(px_break, row.open)
        return bv.index[idx], px, line_, rvr_max, rvr_arr

    def _conf_end_consol(
        self, is_adv: bool, conf_start: pd.Timestamp
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series]:
        """Return bar when movement would be confirmed as ending by consolidation.

        Returns
        -------
        3-tuple of [pd.Timestamp | None, float | None, pd.Series]
            [0] Indice of `df` that movement would end by consolidation.
                None if movement ongoing as at end of available data.
            [1] Open price of bar when movement would end by consolidation.
                None if movement ongoing as at end of available data.
            [2] Line representing limit beyond which movement must be
                extended in order to be considered continuing as opposed to
                having given way to consolidation.
        """
        df = self.data.loc[conf_start:]
        col = df.high if is_adv else df.low
        win = col.rolling(self.prd, min_periods=0)
        fctrs = self.factors_break_pos_rev if is_adv else self.factors_break_neg_rev

        def rolling_win_func(arr: np.ndarray) -> float:
            win = arr * fctrs[-len(arr) :]
            return win.max() if is_adv else win.min()

        # line that, if not exceeded over prd, movement will be considered to have ended
        line_ = win.apply(rolling_win_func)
        # shift as evaluating limits for subsequent bar to exceed
        line = line_.shift(1, fill_value=line_.iloc[0])

        if len(df) <= self.prd:
            return None, None, line

        # end is first instance when self.prd consecutive bars fail to break line
        bv = (col > line if is_adv else col < line)[1:].rolling(self.prd).sum() == 0
        if not bv.any():
            return None, None, line
        end = bv.index[bv.argmax()]
        return end, df.loc[end].open, line[:end]

    def conf_end(
        self, start: pd.Timestamp, conf_start: pd.Timestamp, is_adv: bool
    ) -> tuple[
        pd.Timestamp | None,
        float | None,
        bool | None,
        bool | None,
        pd.Series,
        pd.Series,
        pd.Series | None,
        float,
        np.ndarray,
    ]:
        """Confirm movement end.

        Parameters
        ----------
        start
            Bar on which movement started.

        conf_start
            Bar when movement was confirmed.

        is_adv
            True: movement represents an advance.
            false: movement represents a decline.

        Returns
        -------
        9-tuple of
        [pd.Timestamp | None, float | None, bool | None, bool | None, pd.Series,
        pd.Series, pd.Series | None, float, np.ndarray]
            [0] pd.Timestamp | None
                Timestamp when movement's end confirmed.

                None if movement ongoing as at end of available data.

            [1] float | None
                If movement ended by consolidation, close price of bar when
                movement's end confirmed.

                If movement ended by recovery, limit price on breaking
                which movement was considered to have given way to
                recovery.

                None if movement ongoing as at end of available data.

            [2] bool | None
                True: movement ended by way of consolidation.
                False: movement ended by way of reversal.
                None: movement ongoing as at end of available data.

            [3] bool | None
                If movement ended by way of reversal, indicates nature of
                reversal.

                True: movement ended by way of price reversing more than
                    the reversal limit (i.e. more than rvr percent of the
                    trend movement).
                False: movement ended by way movement coming through in the
                    opposite direction.
                None: movement did not end by way of reversal.

            [4] pd.Series
                Line representing limit beyond which movement must be
                extended in order to be considered continuing as opposed
                to having given way to consolidation.

            [5] pd.Series
                Line representing reversal limit.

            [6] pd.Series | None
                If movement gave way to a movement in the opposite
                direction, the line indicating limit of consolidation range
                the breaking of which confirmed the opposing movement.

                None if movement did not give way to a movement in the
                opposite direction.

            [7] float
                Maximum percentage reversal before movement would have been
                considered to have given way to consolidation.

            [8] np.ndarray
                Array describing reversal limits. First value corresponds
                with reversal limit on bar when movement was confirmed.
        """
        end_rvr_opp, end_rvr_opp_px, line_rvr_opp = self._conf_end_rvr_opp(
            is_adv, conf_start
        )
        end_cnsl, end_cnsl_px, line_cnsl = self._conf_end_consol(is_adv, conf_start)

        tss = [ts for ts in [end_cnsl, end_rvr_opp] if ts is not None]
        end_alt = min(tss) if tss else None
        end_rvr, end_rvr_px, line_rvr_pct, rvr, rvr_arr = self._conf_end_rvr_pct(
            is_adv, start, conf_start, end_alt
        )

        lines = line_cnsl, line_rvr_pct
        rvrs = rvr, rvr_arr
        if end_alt is None and end_rvr is None:
            return None, None, None, None, *lines, None, *rvrs

        if end_rvr_opp is None:
            end_rvr_by_opp = False
        elif end_rvr is None:
            end_rvr_by_opp = True
        elif end_rvr_opp == end_rvr:
            # would be considered to have ended in both manners on same bar
            end_rvr_by_opp = (
                end_rvr_opp_px >= end_rvr_px if is_adv else end_rvr_opp_px <= end_rvr_px
            )
        else:
            end_rvr_by_opp = end_rvr_opp < end_rvr

        if end_rvr_by_opp:
            end_rvr = end_rvr_opp
            end_rvr_px = end_rvr_opp_px
        else:
            line_rvr_opp = None

        if end_cnsl is None:
            return (
                end_rvr,
                end_rvr_px,
                False,
                end_rvr_by_opp,
                *lines,
                line_rvr_opp,
                *rvrs,
            )
        if end_rvr is None or end_cnsl < end_rvr:
            return end_cnsl, end_cnsl_px, True, None, *lines, None, *rvrs
        return (end_rvr, end_rvr_px, False, end_rvr_by_opp, *lines, line_rvr_opp, *rvrs)

    def get_end_est_line(
        self, base: float, index: pd.DatetimeIndex, is_adv: bool
    ) -> pd.Series:
        """Get end establishment line (EEL).

        Parameters
        ----------
        base:
            Last y-value of EEL.

        index:
            x-values of EEL.

        is_adv:
            True if establishing end of an advance.
            False if establishing end of a decline.
        """
        factors = self.grad_factors_neg_rev if is_adv else self.grad_factors_pos_rev
        factors = factors[-len(index) :]
        return pd.Series(factors, index=index, dtype="float64") * base

    def get_movement_end(
        self, start: pd.Timestamp, end_conf: pd.Timestamp, is_adv: bool, by_consol: bool
    ) -> tuple[pd.Timestamp | pd.Interval, float, pd.Series | None]:
        """Get movement end bar and price.

        Parameters
        ----------
        start
            Bar on which movement started.

        end_conf
            Bar on which end of movement was confirmed.

        is_adv
            True: movement represents an advance.
            false: movement represents a decline.

        by_consol
            True: movement ended by way of consolidation.
            False: movement ended by way of reversal.

        Returns
        -------
        2-tuple of [pd.Timestamp | pd.Interval, float, pd.Series | None]
            [0] Bar representing end of movement.
            [1] Price as at movement end.
            [2] End Establishment Line if ended by consol, otherwise None.
        """
        df = self.data.loc[start:end_conf]

        idx = df.high.argmax() if is_adv else df.low.argmin()
        row = df.iloc[idx]
        end = row.name
        if not by_consol:
            return end, row.high if is_adv else row.low, None

        px = row.high if is_adv else row.low
        subset = df.iloc[max(0, idx - self.prd + 1) : idx]
        eel = self.get_end_est_line(px, subset.index, is_adv)
        bv = subset.high > eel if is_adv else subset.low < eel
        eel[row.name] = px
        if bv.any():
            idx_bv = subset[bv].high.argmax() if is_adv else subset[bv].low.argmin()
            end = subset[bv].index[idx_bv]
            return end, subset.loc[end].high if is_adv else subset.loc[end].low, eel
        return end, row.high if is_adv else row.low, eel

    def get_movement(
        self, df: pd.DataFrame, minstart_conf: pd.Timestamp, minstart: pd.Timestamp
    ) -> MovementAlt | None:
        """Define the first concluded movement.

        Defines the first movement which:
            is confirmed no earlier than `minstart_conf`.
            starts no earlier than `minstart`
            concludes before end of `df`.

        An advance/decline is considered to be confirmed on a bar when the
        y value is greater/lesser than all values of the prior
        "`self.prd` - 1" bars and has extened those prior highs/lows by a
        greater degree than would be considered as consolidation (i.e. as
        determined by having exceeded the Limit Line - see introduction to
        class doc).

        Returns None if no movement is identified.

        Parameters
        ----------
        df
            pd.DataFrame with rows representing bar interval and columns as
            OHLC. Method makes no changes to this dataframe.

        minstart_conf
            Returned movement will have a confirmation bar no earlier than
            `minstart_conf`.

        minstart
            Returned movement will have a start bar no earlier than
            `minstart`.
        """
        move = self.find_movement(df, minstart, minstart_conf)
        if move is None:
            return None
        is_adv, start, start_px, sel, start_conf, start_conf_px, start_conf_line = move

        (
            end_conf,
            end_conf_px,
            by_consol,
            rvr_by_opp,
            cns_line,
            rvr_pct_line,
            rvr_opp_line,
            rvr,
            rvr_arr,
        ) = self.conf_end(start, start_conf, is_adv)

        if end_conf is None:  # movement had not ended as at end of data
            end, eel = None, None
            subset = self.data.loc[start:]
            end_px = subset.high.max() if is_adv else subset.low.min()
            duration = len(df) - df.index.get_loc(start)
        else:
            end, end_px, eel = self.get_movement_end(start, end_conf, is_adv, by_consol)
            duration = df.index.get_loc(end) - df.index.get_loc(start) + 1

        movement = MovementAlt(
            is_adv,
            start,
            start_px,
            start_conf,
            start_conf_px,
            end,
            end_px,
            end_conf,
            end_conf_px,
            duration,
            self.params,
            sel,
            start_conf_line,
            eel,
            cns_line,
            rvr_pct_line,
            rvr_opp_line,
            by_consol,
            False if rvr_by_opp is None else not rvr_by_opp,
            rvr,
            rvr_arr,
        )
        return movement

    def get_movements(self) -> MovementsAlt:
        """Evaluate all movements."""
        moves = []
        df = self.data.copy()
        minstart_conf = df.index[self.prd]
        move = self.get_movement(df, minstart_conf, df.index[0])
        if move is None:
            return MovementsAlt(tuple(), self.data.copy(), self.interval)
        moves.append(move)
        end = move.end

        while True:
            end_idx = self.data.index.get_loc(end)
            df = self.data[end_idx - self.prd :].copy()
            if end_idx + 1 == len(self.data):
                return MovementsAlt(tuple(moves), self.data.copy(), self.interval)
            minstart_conf = self.data.index[end_idx + 1]
            move = self.get_movement(df, minstart_conf, end)
            if move is None:
                return MovementsAlt(tuple(moves), self.data.copy(), self.interval)
            # assert movement does not overlap with prior movement
            # lose assertion when test suite sufficient
            assert move.start >= moves[-1].end
            moves.append(move)
            if move.end is None:
                return MovementsAlt(tuple(moves), self.data.copy(), self.interval)
            end = move.end
