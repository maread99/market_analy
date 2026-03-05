"""Base charts for analysis of positions."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import bqplot as bq
import numpy as np
from bqplot import Lines

import market_analy.utils.bq_utils as ubq
from market_analy.cases import ChartSupportsCasesGui
from market_analy.charts import Groups, OHLCCaseBase, tooltip_html_style
from market_analy.formatters import formatter_datetime, formatter_float

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from .positions import PositionBase, PositionsBase


class ChartPositionsBase(OHLCCaseBase, ChartSupportsCasesGui):
    """Chart positions taken in a single instrument.

    OHLC chart with positions overlay. Each position represented
    by:
        Single line tracking price (as `PositionBase.line`).
        Line is green for profitable positions, red for
        positions that are not profitable.

        Scatter mark point representing position open (circle).
        This point also represents the position more generally.

        Scatter mark point representing position close (cross).

    Parameters
    ----------
    As for OHLCCaseBase except:

    cases
        Positions over period to be charted.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        cases: PositionsBase,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        *,
        log_scale: bool = True,
        display: bool = False,
        handler_click_case: Callable | None = None,
        data_y2: None = None,
    ):
        super().__init__(
            prices,
            cases,
            title,
            visible_x_ticks,
            max_ticks,
            log_scale,
            display,
            handler_click_case,
            data_y2,
        )

        self.cases: PositionsBase

    @property
    def current_case(self) -> PositionBase | None:
        """Currently selected position."""
        case = super().current_case
        if typing.TYPE_CHECKING:
            assert isinstance(case, PositionBase)
        return case

    def _add_marks(self, **_) -> None:
        """Add marks.

        Represents each position with:
            Line tracking position.
            Point on a Scatter ('circle') indicating open.
            Point on a Scatter ('cross') indicating close.
        """
        colors = [
            "lime" if profitable else "crimson" for profitable in self.cases.profitables
        ]
        lines = []
        for line, color in zip(self.cases.lines, colors, strict=True):
            lines.append(
                Lines(
                    scales=self.scales,
                    colors=[color],
                    x=line.index,
                    y=line.values,
                    stroke_width=5,
                    line_style="solid",
                    interpolation="linear",
                )
            )
        self.add_marks(lines, Groups.CASES_OTHER_0)

        # scatter for position opens
        self._add_scatter(
            self.cases.open_bars,
            self.cases.open_pxs,
            colors,
            "circle",
            self.handler_hover_open,
            self._handler_click_case,
        )

        # scatters for closes
        self._add_scatter(
            self.cases.close_bars,
            self.cases.close_pxs,
            colors,
            "cross",
            self.handler_hover_close,
        )

    def handler_hover_open(self, mark: bq.marks.Scatter, event: dict) -> None:
        """Handler for hovering on position open mark.

        Displays tooltip describing scatter point.
        """
        pos = self.cases.event_to_case(mark, event)
        mark.tooltip.value = self.cases.get_case_html(pos)

    def handler_hover_close(self, mark: bq.Scatter, event: dict) -> None:
        """Handler for hovering on position close mark.

        Displays tooltip describing scatter point.
        """
        pos = self.cases.event_to_case(mark, event)
        i = self.cases.get_index(pos)
        color = mark.colors[i]
        style = tooltip_html_style(color=color, line_height=1.3)
        date = ubq.discontinuous_date_to_timestamp(event["data"]["x"])
        s = f"<p {style}>Close: {formatter_datetime(date)}"
        s += f"<br>Close px: {formatter_float(event['data']['y'])}</p"
        mark.tooltip.value = s


def distributed_rtrns(
    rtrns: pd.Series,
) -> bq.Figure:  # type: ignore[return-value]
    """Return chart describing distribution of returns.

    Parameters
    ----------
    rtrns
        Series describing return of each scenario.

    Notes
    -----
    Very rough. Develop / Refactor as required.
    """
    if not rtrns.is_monotonic_decreasing:
        rtrns = rtrns.sort_values(ascending=False)

    srs = rtrns.reset_index(drop=True)

    scale_x = bq.LinearScale()  # type: ignore[attr-defined]
    scale_y = bq.LinearScale()  # type: ignore[attr-defined]
    kwargs = {
        "scales": {"x": scale_x, "y": scale_y},
        "stroke_width": 3,
        "fill_opacities": [0.7],
        "fill": "between",
        "display_legend": True,
    }
    axis_x = bq.Axis(
        orientation="horizontal",
        side="bottom",
        scale=scale_x,
        offset={"scale": scale_y, "value": 0},
    )

    axis_y = bq.Axis(
        orientation="vertical",
        side="left",
        scale=scale_y,
    )

    marks = []
    bv = srs > 0
    if bv.any():
        label1 = str(round(srs[bv].sum(), 2))
        if str(rtrns.index.dtype) == "object":
            label1 += " " + ", ".join(rtrns.index[bv][:5].tolist())
        marks.append(
            bq.Lines(  # type: ignore[attr-defined]
                y=[
                    srs[bv].to_numpy(),
                    np.array([0] * len(srs[bv])),
                ],
                x=srs[bv].index,
                colors=["Lime", "Gray"],
                fill_colors=["Green"],
                labels=[label1, str(len(srs[bv]))],
                **kwargs,
            )
        )
    bv = srs < 0
    if bv.any():
        label1 = str(round(srs[bv].sum(), 2))
        if str(rtrns.index.dtype) == "object":
            label1 += " " + ", ".join(rtrns.index[bv][-5:].tolist()[::-1])
        marks.append(
            bq.Lines(  # type: ignore[attr-defined]
                y=[
                    srs[bv].to_numpy(),
                    np.array([0] * len(srs[bv])),
                ],
                x=srs[bv].index,
                colors=["Red", "Gray"],
                fill_colors=["Firebrick"],
                labels=[label1, str(len(srs[bv]))],
                **kwargs,
            )
        )
    bv = srs == 0
    if bv.any():
        marks.append(
            bq.Lines(  # type: ignore[attr-defined]
                y=srs[bv],
                x=srs[bv].index,
                colors=["SkyBlue"],
                labels=[str(len(srs[bv]))],
                **kwargs,
            )
        )

    return bq.Figure(  # type: ignore[attr-defined]
        title="Distributed Returns",
        title_style={"fill": "white"},
        axes=[axis_x, axis_y],
        marks=marks,
        scale_x=scale_x,
        scale_y=scale_y,
        fig_margin={
            "left": 100,
            "right": 180,
            "top": 30,
            "bottom": 30,
        },
        legend_location="bottom-left",
    )
