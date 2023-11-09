"""GUIs to visualise movements identifeid by Trends classes."""

from __future__ import annotations

import typing
from collections import abc
from collections.abc import Callable
from contextlib import contextmanager

import bqplot as bq
import ipyvuetify as v
import market_prices as mp
import pandas as pd

import market_analy

from ..guis import ChartOHLCCaseBase
from ..utils import ipyvuetify_utils as vu
from ..utils.bq_utils import HFixedRule
from . import charts
from .analy import RvrAlt, Trends, TrendsAlt

if typing.TYPE_CHECKING:
    from .. import analysis as ma_analysis
    from . import TrendsProto
    from .movements import Movement, MovementAlt, Movements, MovementsAlt


class TrendsGuiBase(ChartOHLCCaseBase):
    """GUI to display and interact with trend analysis over OHLC Chart.

    Parameters
    ----------
    analysis
        Analysis instance representing instrument to be plotted.

    interval
        Interval covered by one bar (i.e. one x-axis tick). As 'interval`
        parameter described by `help(analysis.prices.get)`.

    trend_cls
        Class to use to analyse trends. Must conform with
        `trends_base.TrendsProto`, for example, `trends.Trends`.

    trend_kwargs
        Arguments to pass to `trend_cls`. Do not include `data` or
        `interval`.

    max_ticks
        Maximum number of bars (x-axis ticks) that will shown by default
        (client can choose to show more via slider). None for no limit.

    log_scale
        True to plot prices against a log scale. False to plot prices
        against a linear scale.

    display
        True to display created GUI.

    narrow_view
        When displaying a movement in 'narrow' view, the number of bars
        that should be shown before the bar representing the movement's
        start and after the bar representing the movement's confirmed end.

    wide_view
        When displaying a movement in 'wide' view, the number of bars that
        should be shown before the bar representing the movement's start
        and after the bar representing the movement's confirmed end.

    chart_kwargs
        Any kwargs to pass on to the chart class.

    **kwargs
        Period for which to plot prices. Passed as period parameters as
        described by market-prices documentation for 'PricesCls.get'
        method where 'PricesCls' is the class that was passed to
        'PricesCls' parameter of `mkt_anlaysis.Analysis` to intantiate
        `analysis` (for example, documenation for
        'market_prices.PricesYahoo.get').

    Notes
    -----
    -- Subclass Implementation --

    This base class can be subclassed, including for the purposes of:

        Passing through the required `trend_cls` and concreting in its own
        constructor arguments passed on to this base as `trend_kwargs`.

        Concrete constructor arguments such as `narrow_view`, `wide_view`
        etc.

        Defining the `_gui_click_case_handler` method to customise the
        handling, at a gui level, of clicking a mark representing the start
        of a trend. NB Alternatively a handler can be passed within
        `chart_kwargs` with the key 'click_case_handler'.
    """

    _HAS_INTERVAL_SELECTOR = False

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval,
        trend_cls: type[TrendsProto],
        trend_kwargs: dict,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        narrow_view: int = 10,
        wide_view: int = 10,
        chart_kwargs: dict | None = None,
        **kwargs,
    ):
        data = self._set_initial_prices(analysis, interval, kwargs).copy()
        if isinstance(data.index, pd.IntervalIndex):
            data.index = data.index.left
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
        self.trends = trend_cls(data, interval, **trend_kwargs)
        cases = self.trends.get_movements()

        super().__init__(
            analysis,
            interval,
            cases,
            max_ticks,
            log_scale,
            display,
            narrow_view,
            wide_view,
            chart_kwargs,
            **kwargs,
        )
        self._rulers: list = []

    @property
    def ChartCls(self) -> type[charts.OHLCTrends]:
        return charts.OHLCTrends

    @property
    def _chart_title(self) -> str:
        return self._analysis.symbol + " Trend Analysis"

    @property
    def _icon_row_top_handlers(self) -> list[Callable]:
        return [self._max_x_ticks, self._resize_chart, self.close]

    @property
    def current_case(self) -> Movement | MovementAlt | None:
        """Current selected case"""
        case = super().current_case
        if typing.TYPE_CHECKING:
            assert isinstance(case, (Movement, MovementAlt))
        return case

    def _create_date_slider(self, **kwargs):
        ds = super()._create_date_slider(**kwargs)
        ds.slider.observe(self.chart.update_trend_mark, ["index"])
        return ds

    def _close_rulers(self):
        for ruler in self._rulers:
            ruler.close()
        self._rulers = []

    def _add_rulers(self):
        self._close_rulers()  # close any existing
        ohlc_mark = next(m for m in self.chart.figure.marks if isinstance(m, bq.OHLC))
        case = self.current_case
        assert case is not None
        self._rulers.append(
            HFixedRule(
                level=case.start_px,
                scales=self.chart.scales,
                figure=self.chart.figure,
                start=case.start.asm8,
                length=case.params["prd"],
                color="yellow",
                draggable=True,
                ordinal_values=list(ohlc_mark.x),
                stroke_width=5,
            )
        )

    def _ruler_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_dark:
            return
        f = self._add_rulers if but.is_light else self._close_rulers
        f()
        # chain handlers...
        self.cases_controls_container.but_ruler_handler(but, event, data)

    def _create_cases_controls_container(self) -> v.Layout:
        controls = super()._create_cases_controls_container()
        controls.but_ruler.on_event("click", self._ruler_handler)
        return controls

    @contextmanager
    def _handler_disabled(self):
        """Undertake an operation within context of disabled handler.

        Undertake an operation with context of slider's handlers being
        disabled.

        Notes
        -----
        Messily overrides method defined on `BaseVariableDates` subclass to
        include the additional `self.chart.update_trend_mark` handler.
        """
        self._slider.unobserve(self._set_chart_x_ticks_to_slider, ["index"])
        self._slider.unobserve(self.chart.update_trend_mark, ["index"])
        yield
        self._slider.observe(self._set_chart_x_ticks_to_slider, ["index"])
        self._slider.observe(self.chart.update_trend_mark, ["index"])

    def _max_x_ticks(self):
        """Show data for all plottable x-ticks (bars).

        Notes
        -----
        Overrides inherited method to ensure trend mark is also updated.
        """
        self.chart.reset_x_ticks()
        self._set_slider_limits_to_all_plottable_x_ticks()
        self.chart.update_trend_mark()


class TrendsGui(TrendsGuiBase):
    """GUI to visualise movements evaluated by `analy.Trends` class.

    Parameters
    ----------
    `prd`, `ext_break`, `ext_limit` and `min_bars` all as `analy.Trends`
    class.

    All other parameters as base class `TrendsGuiBase`.
    """

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
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
        self.cases: Movements
        self.trends: Trends

    def _gui_click_case_handler(self, mark: bq.Scatter, event: dict):
        """Gui level handler for clicking a mark representing a trend start.

        Lightens 'show all scatters' button to indicate option available.
        Displays tooltip to html output.
        """
        self.cases_controls_container.lighten_single_case()
        self.cases_controls_container.but_show_all.darken()
        move = self.cases.event_to_case(mark, event)
        html = self.cases.get_move_html(move)
        self.html_output.display(html)

    @property
    def current_case(self) -> Movement | None:
        """Current selected case"""
        # method included to update type to Movement class as defined on this module
        case = super().current_case
        if typing.TYPE_CHECKING:
            assert isinstance(case, Movement)
        return case

    def _add_rulers(self):
        self._close_rulers()  # close any existing
        case = self.current_case
        assert case is not None
        is_adv = case.is_adv

        ohlc_mark = next(m for m in self.chart.figure.marks if isinstance(m, bq.OHLC))

        fctrs = self.trends.fctrs_pos_break if is_adv else self.trends.fctrs_neg_break
        self._rulers.append(
            market_analy.utils.bq_utils.TrendRule(
                x=case.start.asm8,
                y=case.start_px,
                length=case.params["prd"],
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
        idx = -case.params["prd"]
        self._rulers.append(
            market_analy.utils.bq_utils.TrendRule(
                x=case.line_limit.index[idx].asm8,
                y=case.line_limit.iloc[idx],
                length=case.params["prd"],
                factors=fctrs,
                scales=self.chart.scales,
                ordinal_values=list(ohlc_mark.x),
                figure=self.chart.figure,
                color="skyblue",
                draggable=True,
                stroke_width=3,
            )
        )


class TrendsAltGui(TrendsGuiBase):
    """GUI to visualise movements evaluated by `analy_alt.Trends` class.

    Parameters
    ----------
    `prd`, `ext`, `rvr`, `grad`, `rvr_init` and `min_bars` all as
    `analy_alt.Trends` class.

    All other parameters as base class `TrendsGuiBase`.
    """

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval,
        prd: int,
        ext: float,
        rvr: float | abc.Sequence[float] | RvrAlt,
        grad: float | None,
        rvr_init: float | tuple[float, float] | list[float],
        min_bars: int,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        **kwargs,
    ):
        trend_kwargs = {
            "prd": prd,
            "ext": ext,
            "grad": grad,
            "rvr": rvr,
            "rvr_init": rvr_init,
            "min_bars": min_bars,
        }
        super().__init__(
            analysis=analysis,
            interval=interval,
            trend_cls=TrendsAlt,
            trend_kwargs=trend_kwargs,
            max_ticks=max_ticks,
            log_scale=log_scale,
            display=display,
            narrow_view=prd,
            wide_view=prd * 3,
            **kwargs,
        )
        self.cases: MovementsAlt
        self.trends: TrendsAlt

    def _gui_click_case_handler(self, mark: bq.Scatter, event: dict):
        """Gui level handler for clicking a mark representing a trend start.

        Lightens 'show all scatters' button to indicate option available.
        Displays tooltip to html output.
        """
        self.cases_controls_container.lighten_single_case()
        self.cases_controls_container.but_show_all.darken()
        move = self.cases.event_to_case(mark, event)
        html = self.cases.get_move_html(move)
        self.html_output.display(html)
