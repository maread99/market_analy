"""Charts for trend analysis."""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import cached_property, partialmethod

import bqplot as bq
import pandas as pd

from ..cases import CaseSupportsChartAnaly
from ..charts import Groups, OHLCCaseBase
from ..config import COL_ADV, COL_DEC
from ..utils import bq_utils as ubq
from .movements import MovementBase, MovementsSupportChartAnaly


class OHLCTrends(OHLCCaseBase):
    """OHLC and Trend analysis for single financial instrument.

    OHLC chart with trend overlay. Trend dislayed as coloured line where
    green indicates a rising trend, red indicates a  falling trend and
    yellow indicates no trend (consolidation).

    For each movement described by `movements` a scatter mark denotes:
        movement start (triangle)
        movement end (cross)
        bar when movement start confirmed (circle)
        bar when movement end confirmed (square)

    Scatter marks for advancing movements are green, those for declining
    movements are red.

    Handlers on `movements`, as defined on `MovementsChartProto`, will be
    enabled.

    Parameters
    ----------
    As for OHLCCaseBase, except:

    cases
        Movements representing trends over period to be charted. Must
        be passed.

    click_case_handler
        Any client-defined handler to be additionally called when user
        clicks on any scatter point representing a trend start. Should have
        signature:
            f(mark: bq.Scatter, event: dict)

        NB handler will be called after any handlers defined on
        `cases`.

    inc_conf_marks
        Whether to include scatter marks indicating the bar / price at
        which movements were confirmed to have started and ended.
    """

    COLOR_MAP = ["yellow", COL_ADV, COL_DEC]  # 0 consol, 1 adv, -1 dec

    def __init__(
        self,
        prices: pd.DataFrame,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = False,
        cases: MovementsSupportChartAnaly | None = None,
        click_case_handler: Callable | None = None,
        inc_conf_marks: bool = True,
    ):
        if cases is None:
            raise ValueError("'cases' is a required argument.")
        self._inc_conf_marks = inc_conf_marks
        super().__init__(
            prices,
            cases,
            title,
            visible_x_ticks,
            max_ticks,
            log_scale,
            display,
            click_case_handler,
        )

        self.cases: MovementsSupportChartAnaly

        if max_ticks is not None or visible_x_ticks is not None:
            self.update_trend_mark()

    @cached_property
    def _y_trend(self) -> pd.Series:
        """y data for all points on trend line."""
        # set starts of advances as day low, starts of declines as day high, ends
        # of advances (that do not conincide with a subsequent trend start) as day
        # high, ends of declines (that do not conincide with a subsequent trend
        # start) as day low, otherwise as day close.
        y = self.prices.close.copy()
        subset = self.cases.starts_adv
        y[subset] = self.prices.low[subset]
        subset = self.cases.starts_dec
        y[subset] = self.prices.high[subset]
        subset = self.cases.ends_adv_solo
        y[subset] = self.prices.high[subset]
        subset = self.cases.ends_dec_solo
        y[subset] = self.prices.low[subset]
        return y

    @cached_property
    def _cols_trend(self) -> list[str]:
        """color data for all points on trend line."""
        return [self.COLOR_MAP[t] for t in self.cases.trend.values]

    def _create_scales(self) -> dict[ubq.ScaleKeys, bq.Scale]:
        scales = super()._create_scales()
        scales["width"] = bq.LinearScale()
        return scales

    create_scatter = partialmethod(OHLCCaseBase.create_scatter, opacities=[0.65])
    _add_scatter = partialmethod(OHLCCaseBase._add_scatter, opacities=[0.65])

    def hide_cases(self):
        """Hide scatter marks for all trends.

        Hide scatter marks denoting movements' starts, ends and, if
        applicable, bars when movements' starts and ends were confirmed.
        """
        return super().hide_cases()

    def show_cases(self):
        """show scatter marks for all trends.

        Shows scatter marks denoting movements' starts, ends and, if
        applicable, bars when movements' starts and ends were confirmed.
        """
        return super().show_cases()

    @property
    def current_case(self) -> MovementBase | None:
        """Last selected case."""
        case = super().current_case
        if typing.TYPE_CHECKING:
            assert isinstance(case, MovementBase)
        return case

    def _create_mark(self, **_) -> bq.OHLC:
        opacities = [0.7] * len(self.prices)
        return super()._create_mark(opacities=opacities)

    def _add_marks(self, **_):
        """Add initial extra marks.

        Adds FlexLine mark to represent trend (to `Groups.TRENDLINE`).

        Adds Scatter marks to `Groups.SCATTERS`.
            To represent, for each trend:
                Start (triangle-up/triangle-down for adv/dec).
                End (cross) only if does not coincide with subsequent
                    trend start.

            If `inc_conf_marks` passed as True (default) to constructor
            then also adds scatter marks to represent, for each trend:
                Conf Start (circle)
                Conf End (square).
        """
        trend_line = bq.FlexLine(
            scales=self.scales,
            colors=self._cols_trend,
            x=self._x_data,
            y=self._y_trend,
            stroke_width=2,
        )
        self.add_marks([trend_line], Groups.TRENDLINE)

        movements = self.cases

        # scatter for starts of advancing trends
        subset = movements.starts_adv
        self._add_scatter(
            subset,
            self._y_trend[subset],
            COL_ADV,
            "triangle-up",
            movements.handler_hover_start,
            self._handler_click_case,
        )
        # scatter for starts of declining trends
        subset = movements.starts_dec
        self._add_scatter(
            subset,
            self._y_trend[subset],
            COL_DEC,
            "triangle-down",
            movements.handler_hover_start,
            self._handler_click_case,
        )
        # scatters for ends of trends, only if do not coincide with subsequent trend start
        handler = movements.handler_hover_end
        subset = movements.ends_adv_solo
        self._add_scatter(subset, self._y_trend[subset], COL_ADV, "cross", handler)
        subset = movements.ends_dec_solo
        self._add_scatter(subset, self._y_trend[subset], COL_DEC, "cross", handler)

        if self._inc_conf_marks:
            # scatters for confirmed starts of trends
            handler = movements.handler_hover_conf_start
            self._add_scatter(
                movements.starts_conf_adv,
                movements.starts_conf_adv_px,
                COL_ADV,
                "circle",
                handler,
            )
            self._add_scatter(
                movements.starts_conf_dec,
                movements.starts_conf_dec_px,
                COL_DEC,
                "circle",
                handler,
            )
            # scatters for confirmed ends of trends
            handler = movements.handler_hover_conf_end
            self._add_scatter(
                movements.ends_conf_adv,
                movements.ends_conf_adv_px,
                COL_ADV,
                "square",
                handler,
            )
            self._add_scatter(
                movements.ends_conf_dec,
                movements.ends_conf_dec_px,
                COL_DEC,
                "square",
                handler,
            )

    @property
    def mark_trend(self) -> bq.FlexLine:
        return self.added_marks[Groups.TRENDLINE][0]

    def update_trend_mark(self, *_):
        """Update trend mark to reflect plotted x ticks.

        Notes
        -----
        Like for the OHLC class, the OHLC mark is not updated to reflect changes
        to the plotted data (`_update_mark_data_attr_to_reflect_plotted` is False
        for the class) as the nature of the mark (discrete renders rather than a
        continuous line) does not result in any side-effects when the visible
        domain is shorter than that the mark data. The same is true (more or less)
        for the Scatter marks. However, it is necessary to update the FlexLine
        to reflect the plotted data to avoid the line doubling back over the render
        as it tries to plot the 'next point after the visible range' to the start
        of the x-axis.

        This method serves as a handler which should be called by a client
        whenever the plotted dates are changed. To handle everything within this
        class had originally overriden the `self._x_domain_chg_handler` method
        which is invoked whenever the x scales domain is changed (i.e. whenever
        the plotted dates are changed), however, and regardless of whether
        holding off the sync to the frontend of not, the 'wrapped back round'
        line could continue to be left and other rendering issues were more
        common than with the implemented solution.

        An alternative approach to resolving the 'wrapped around line' is to set
        line widths to 0 for the segments prior to the first and subsequent to the
        last plotted interval. This still leaves a thin trace, for which also set
        the colour of these segments to the same as the background colour. It's an
        option, although found it rendered less smoothly that the implemented
        approach (i.e. updating the marks values to reflect the plotted intervals).
        """
        self.mark_trend.x = self.plotted_x_ticks
        self.mark_trend.y = self._y_trend[self._domain_bv]
        self.mark_trend.colors = [
            col for col, b in zip(self._cols_trend, self._domain_bv) if b
        ]

        index = next(
            (i for i, m in enumerate(self.figure.marks) if m.name == "Flexible lines")
        )
        self.figure.marks = [m for m in self.figure.marks if m.name != "Flexible lines"]
        self.figure.marks = (
            self.figure.marks[:index] + [self.mark_trend] + self.figure.marks[index:]
        )

    def select_case(self, case: CaseSupportsChartAnaly):
        """Select a specific case."""
        case = typing.cast(MovementBase, case)
        i = self.cases.get_index_for_direction(case)
        scatter_index = 0 if case.is_adv else 1
        self._click_case(i, scatter_index)
