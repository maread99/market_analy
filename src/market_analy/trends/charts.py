"""Charts for trend analysis."""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import cached_property, partialmethod

import bqplot as bq
import ipywidgets as w
import numpy as np
import pandas as pd

from ..cases import CaseSupportsChartAnaly
from ..charts import TOOLTIP_STYLE, Groups, OHLCCaseBase, tooltip_html_style
from ..config import COL_ADV, COL_DEC
from ..formatters import formatter_datetime, formatter_float, formatter_percent
from ..utils import bq_utils as ubq
from .movements import MovementBase, MovementsSupportChartAnaly, Movement, MovementAlt


class TrendsChart(OHLCCaseBase):
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

    Parameters
    ----------
    As for OHLCCaseBase, except:

    cases
        Movements representing trends over period to be charted. Must
        be passed.

    handler_click_case
        Any client-defined handler to be additionally called when user
        clicks on any scatter point representing a trend start. Should have
        signature:
            f(case: Movement, mark: bq.Scatter, event: dict)

        NB handler will be called after any handlers defined on this class.

    inc_conf_marks
        Whether to include scatter marks indicating the bar / price at
        which movements were confirmed to have started and ended.
    """

    COLOR_MAP = ["yellow", COL_ADV, COL_DEC]  # 0 consol, 1 adv, -1 dec
    MARKER_MAP: typing.ClassVar = {
        "cross": "End",
        "circle": "Start conf",
        "square": "End conf",
    }

    def __init__(
        self,
        prices: pd.DataFrame,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = False,
        cases: MovementsSupportChartAnaly | None = None,
        handler_click_case: Callable | None = None,
        inc_conf_marks: bool = True,
    ):
        if cases is None:
            raise ValueError("'cases' is a required argument.")
        self._inc_conf_marks = inc_conf_marks

        self._SCATTER_INDEXES_ADV: list[int]  # set by _add_marks
        self._SCATTER_INDEXES_DEC: list[int]  # set by _add_marks

        super().__init__(
            prices,
            cases,
            title,
            visible_x_ticks,
            max_ticks,
            log_scale,
            display,
            handler_click_case,
        )

        self.cases: MovementsSupportChartAnaly

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

    @property
    def current_case(self) -> MovementBase | None:
        """Currently selected case."""
        case = super().current_case
        if typing.TYPE_CHECKING:
            assert isinstance(case, MovementBase)
        return case

    def _add_marks(self, **_):
        """Add initial extra marks.

        Adds FlexLine mark to represent trend (to `Groups.PERSIST`).

        Adds Scatter marks to `Groups.CASES_SCATTERS`.
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
        self.add_marks([trend_line], Groups.PERSIST)

        cases = self.cases

        self._SCATTER_INDEXES_ADV = [0, 2]
        self._SCATTER_INDEXES_DEC = [1, 3]

        # add conf marks first as want them at the bottom of any pile up
        if self._inc_conf_marks:
            self._SCATTER_INDEXES_ADV += [4, 6]
            self._SCATTER_INDEXES_DEC += [5, 7]

            # scatters for confirmed starts of trends
            handler = self.handler_hover_conf_start
            self._add_scatter(
                cases.starts_conf_adv,
                cases.starts_conf_adv_px,
                COL_ADV,
                "circle",
                handler,
            )
            self._add_scatter(
                cases.starts_conf_dec,
                cases.starts_conf_dec_px,
                COL_DEC,
                "circle",
                handler,
            )
            # scatters for confirmed ends of trends
            handler = self.handler_hover_conf_end
            self._add_scatter(
                cases.ends_conf_adv,
                cases.ends_conf_adv_px,
                COL_ADV,
                "square",
                handler,
            )
            self._add_scatter(
                cases.ends_conf_dec,
                cases.ends_conf_dec_px,
                COL_DEC,
                "square",
                handler,
            )

        # scatters for ends of trends
        handler = self.handler_hover_end
        subset = cases.ends_adv
        self._add_scatter(subset, self._y_trend[subset], COL_ADV, "cross", handler)
        subset = cases.ends_dec
        self._add_scatter(subset, self._y_trend[subset], COL_DEC, "cross", handler)

        # add starts of trends last so that they appear on top of any pile
        # scatter for starts of advancing trends
        subset = cases.starts_adv
        self._add_scatter(
            subset,
            self._y_trend[subset],
            COL_ADV,
            "triangle-up",
            self.handler_hover_start,
            self._handler_click_case,
        )
        # scatter for starts of declining trends
        subset = cases.starts_dec
        self._add_scatter(
            subset,
            self._y_trend[subset],
            COL_DEC,
            "triangle-down",
            self.handler_hover_start,
            self._handler_click_case,
        )

    @property
    def mark_trend(self) -> bq.FlexLine:
        return self.added_marks[Groups.PERSIST][0]

    def update_trend_mark(self, *_):
        """Update trend mark to reflect plotted x ticks.

        See doc on base `OHLCCaseBase` class.

        Notes
        -----
        Necessary to update FlexLine to reflected plotted data to avoid the
        line doubling back over the render as it tries to plot the
        'next point after the visible range' to the start of the x-axis.

        An alternative approach, to that implemented here, to resolve the
        'wrapped around line' is to set line widths to 0 for the segments
        prior to the first and subsequent to the last plotted interval.
        This still leaves a thin trace, for which also set the colour of
        these segments to the same as the background colour. It's an
        option, although found it rendered less smoothly that the
        implemented approach (i.e. updating the marks values to reflect the
        plotted intervals).
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
        self._simulate_click_case(i, scatter_index)

    # HANDLERS
    def handler_hover_start(self, mark: bq.marks.Scatter, event: dict):
        """Handler for hovering on mark representing a movement start."""
        case = self.cases.event_to_case(mark, event)
        mark.tooltip.value = self.cases.get_case_html(case)

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

    def _create_line(
        self, data: pd.Series, color_: str, line_style: str, desc: str
    ) -> bq.Lines:
        """Create a `bq.Lines` mark."""
        tooltip_str = f"<p {tooltip_html_style(color=color_)}>{desc}</p>"
        line = bq.Lines(
            x=data.index,
            y=data,
            scales=self.scales,
            colors=[color_],
            line_style=line_style,
            tooltip=w.HTML(value=tooltip_str),
            tooltip_style=TOOLTIP_STYLE,
        )
        return line

    show_added_marks_all = partialmethod(OHLCCaseBase.show_added_marks_all, 0.65)
    opacate_added_marks = partialmethod(OHLCCaseBase.opacate_added_marks, opacity=0.65)

    @staticmethod
    def show_only_one_scatter_point(index: int, scatter: bq.Scatter):
        """Show only the point of a given `scatter` at a given `index`.

        Notes
        -----
        Overrides inherited method to set opacities to 0.65.
        """
        arr = np.array([0.65 if i == index else 0 for i in range(len(scatter.x))])
        scatter.opacities = arr

    def _add_case_rectangle(self, case: MovementBase, mark: bq.Scatter):
        """Add rect representing chg between conf pxs for specific case."""
        color = mark.colors[0]
        if case.closed:
            if typing.TYPE_CHECKING:
                assert case.conf_chg_pct is not None
                assert case.end_conf_px is not None
            style = tooltip_html_style(color=color)
            s = f"<p {style}>Conf chg: "
            chg_color = COL_DEC if case.conf_chg_pct < 0 else COL_ADV
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
                scales=self.scales,
                opacities=[0],
                fill="between",
                fill_colors=[color_area],
                fill_opacities=[0.2],
                tooltip=w.HTML(value=s),
                tooltip_style=TOOLTIP_STYLE,
            )
            self.add_marks([mark_chg], Groups.CASE, under=True)

    def _add_case_marks(self, case: CaseSupportsChartAnaly, mark: bq.Scatter):
        """Add marks to represent specific case."""
        case = typing.cast(Movement, case)
        self._add_case_rectangle(case, mark)

        color_break, color_limit = "white", "slategray"
        if not case.by_break:
            color_break, color_limit = color_limit, color_break

        self.add_marks(
            [
                self._create_line(case.line_break, color_break, "dashed", "Break line"),
                self._create_line(case.line_limit, color_limit, "dashed", "Limit line"),
            ],
            Groups.CASE,
        )

    def focus_case(self, case: CaseSupportsChartAnaly, mark: bq.Scatter):
        """Focus on a given case.

        Adds scatters and lines to represent specific case.
        """
        case = typing.cast(Movement, case)
        self.remove_case_marks()
        self.reset_marks()
        idx = self.cases.get_index_for_direction(case)
        exclude_scatters = (
            self._SCATTER_INDEXES_DEC if case.is_adv else self._SCATTER_INDEXES_ADV
        )
        self.show_only_one_cases_scatter_point(idx, exclude_scatters=exclude_scatters)
        for i, mark in enumerate(self.added_marks[Groups.CASES_SCATTERS]):
            if i in exclude_scatters:
                mark.visible = False
        self._add_case_marks(case, mark)


class TrendsAltChart(TrendsChart):
    """OHLC and Trend analysis for single financial instrument.

    As TrendsChart, save for this class creating charts to visualise
    trends evaluted by the `trends.analy.TrendsAlt` class.
    """

    def _add_case_marks(self, case: CaseSupportsChartAnaly, mark: bq.Scatter):
        """Add marks to represent specific case."""
        case = typing.cast(MovementAlt, case)
        self._add_case_rectangle(case, mark)

        marks = []
        color = mark.colors[0]

        def fl(data: pd.Series, color_: str, line_style: str, desc: str) -> bq.Lines:
            line = self._create_line(data, color_, line_style, desc)
            marks.append(line)
            return line

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
            scales=self.scales,
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

        self.add_marks(marks, Groups.CASE)
