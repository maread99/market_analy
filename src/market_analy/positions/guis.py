"""Base GUIs to visualise and analyse position data."""

from __future__ import annotations

import typing

from market_analy.guis import GuiOHLCCaseBase

from . import charts

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import market_prices as mp

    from market_analy import analysis as ma_analysis

    from .positions import PositionBase, PositionsBase


class PositionsGuiBase(GuiOHLCCaseBase):
    """Base GUI to display and interact with positions.

    See `GuiOHLCCaseBase` for documentation of parameters
    and methods defined on the base class.

    Parameters
    ----------
    analysis
        Analysis instance representing instrument to be
        plotted.

    interval
        Interval covered by one bar.

    cases
        Positions to display.

    max_ticks
        Maximum number of bars (x-axis ticks) that will shown
        by default. None for no limit.

    log_scale
        True to plot prices against a log scale. False to plot
        prices against a linear scale.

    display
        True to display created GUI.

    narrow_view
        Number of bars shown before/after a position in
        'narrow' view.

    wide_view
        Number of bars shown before/after a position in
        'wide' view.

    chart_kwargs
        Any kwargs to pass on to the chart class.

    **kwargs
        Period parameters as described by market-prices
        documentation.

    Notes
    -----
    -- Subclass Implementation --

    Subclasses should prepare data and cases (positions) in
    their own ``__init__`` and pass them to this base class.
    Subclasses can also:

        Override `ChartCls` to return a specific chart class.

        Override `_chart_title` to provide a specific title.

        Concrete constructor arguments such as `narrow_view`,
        `wide_view` etc.
    """

    _HAS_INTERVAL_SELECTOR = False

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval,
        cases: PositionsBase,
        max_ticks: int | None = None,
        *,
        log_scale: bool = True,
        display: bool = True,
        narrow_view: int = 15,
        wide_view: int = 50,
        chart_kwargs: dict | None = None,
        **kwargs,
    ):
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

        self.cases: PositionsBase

    @property
    def ChartCls(self) -> type[charts.ChartPositionsBase]:  # noqa: N802
        """Chart class."""
        return charts.ChartPositionsBase

    @property
    def _chart_title(self) -> str:
        return self._analysis.symbol + " Positions Analysis"

    @property
    def _icon_row_top_handlers(self) -> list[Callable]:
        return [
            self._max_x_ticks,
            self._resize_chart,
            self.close,
        ]

    @property
    def current_case(self) -> PositionBase | None:
        """Current selected case."""
        case = super().current_case
        if case is None:
            return None
        return typing.cast("PositionBase", case)
