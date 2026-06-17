"""Tests for `guis` modules."""

import itertools
from collections import abc

import pandas as pd
import pytest

from market_analy import analysis, charts
from market_analy.subplots import Subplot, _volume
from market_analy.utils.maths_utils import nice_ticks

# TODO: tests are extremely incomplete!


def _close(prices: pd.DataFrame) -> pd.Series:
    """DataCreator returning close price as a single series."""
    return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")


class TestGuiSubplots:
    """Tests for subplots stacked beneath a price chart GUI."""

    @pytest.fixture
    def analy(self, prices_analysis) -> abc.Iterator[analysis.Analysis]:
        yield analysis.Analysis(prices_analysis)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_no_subplots_default(self, analy, pp):
        """Without `subplots` the gui has no subplots."""
        gui = analy.plot(**pp, display=False)
        assert gui._subplots == []
        assert gui.chart.figure in gui._gui_box_contents
        # x-tick labels remain visible on the price chart
        assert "display" not in gui.chart.axes[0].tick_style

    def test_subplots_shared_scale(self, analy, pp):
        """Subplot reuses the price chart's x-axis scale."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        assert len(gui._subplots) == 1
        pane = gui._subplots[0]
        assert isinstance(pane, charts.SubplotBars)
        assert pane.scales["x"] is gui.chart.scales["x"]
        assert pane.x_ticks.equals(gui.chart.x_ticks)

    def test_subplots_in_gui_box(self, analy, pp):
        """Subplot figure sits between the price chart and the date slider."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        contents = gui._gui_box_contents
        i_chart = contents.index(gui.chart.figure)
        i_pane = contents.index(gui._subplots[0].figure)
        i_slider = contents.index(gui.date_slider)
        assert i_chart < i_pane < i_slider

    def test_subplots_data(self, analy, pp):
        """Subplot data reflects the data_creator output."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        expected = _volume(gui.prices)
        assert (pane.data.values == expected.values).all()

    def test_subplots_multiple_ordered(self, analy, pp):
        """Multiple subplots stack, in order, all sharing the x scale."""
        custom = Subplot(data_creator=_close, kind="lines", title="Close")
        gui = analy.plot(**pp, subplots=["volume", custom], display=False)
        assert len(gui._subplots) == 2
        assert isinstance(gui._subplots[0], charts.SubplotBars)
        assert isinstance(gui._subplots[1], charts.SubplotLines)
        scale = gui.chart.scales["x"]
        assert all(sp.scales["x"] is scale for sp in gui._subplots)
        contents = gui._gui_box_contents
        assert contents.index(gui._subplots[0].figure) < contents.index(
            gui._subplots[1].figure
        )

    def test_x_label_policy(self, analy, pp):
        """X-tick labels show only on the bottom-most pane."""
        custom = Subplot(data_creator=_close, kind="lines", title="Close")
        gui = analy.plot(**pp, subplots=["volume", custom], display=False)
        assert gui.chart.axes[0].tick_style.get("display") == "none"
        assert gui._subplots[0].axes[0].tick_style.get("display") == "none"
        assert "display" not in (gui._subplots[-1].axes[0].tick_style or {})

    def test_shared_scale_lockstep(self, analy, pp):
        """Changing the price chart's plotted window moves the subplot too."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        x_ticks = gui.chart.x_ticks
        narrow = pd.Interval(x_ticks[1], x_ticks[-1], closed="left")
        gui.chart.plotted_x_ticks = narrow
        assert pane.plotted_x_ticks.equals(gui.chart.plotted_x_ticks)

    def test_update_subplots(self, analy, pp):
        """Recomputing subplots from new prices refreshes the pane data."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        new_prices = gui.prices * 2
        gui._update_subplots(new_prices)
        expected = _volume(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_update_chart_funnel_refreshes_subplots(self, analy, pp):
        """The `_update_chart` funnel refreshes subplots when prices change."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        new_prices = gui.prices * 2
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        expected = _volume(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_reset_chart_resets_subplots(self, analy, pp):
        """Resetting the chart resets subplot."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        expected = _volume(gui.prices)
        new_prices = gui.prices * 2
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        gui._reset_chart()
        assert (pane.data.values == expected.values).all()

    def test_ref_levels_add_marks(self, analy, pp):
        """A subplot with `ref_levels` adds persistent reference marks."""
        custom = Subplot(
            data_creator=_close, kind="lines", title="Close", ref_levels=[1.0]
        )
        gui = analy.plot(**pp, subplots=[custom], display=False)
        pane = gui._subplots[0]
        assert len(pane.added_marks[charts.Groups.PERSIST]) == 1

    def test_volume_y_axis_min_non_negative(self, analy, pp):
        """Volume subplot y-axis min is below the lowest bar but never < 0."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        y_min = pane.scales["y"].min
        lo = float(pane._plotted_y.values.min())
        assert y_min >= 0
        assert y_min <= lo
        # the figure must not pad the y-scale beyond the set min/max (else
        # the rendered axis can extend below zero given the bar baseline at 0)
        assert pane.scales["y"].allow_padding is False

    def test_volume_y_axis_ticks_are_nice(self, analy, pp):
        """Y-axis ticks are round, equally spaced values within the range."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        ticks = list(pane.axes[1].tick_values)
        scale = pane.scales["y"]
        assert ticks == pytest.approx(nice_ticks(scale.min, scale.max))
        # within the axis range and equally spaced (linear)
        assert scale.min <= min(ticks)
        assert max(ticks) <= scale.max
        diffs = [b - a for a, b in itertools.pairwise(ticks)]
        assert all(d == pytest.approx(diffs[0]) for d in diffs)

    def test_line_subplot_reflects_plotted(self, analy, pp):
        """Line subplot mark data tracks the plotted window.

        (So the line keeps rendering when the first date is excluded.)
        """
        custom = Subplot(data_creator=_close, kind="lines", title="Close")
        gui = analy.plot(**pp, subplots=[custom], display=False)
        pane = gui._subplots[0]
        assert pane._update_mark_data_attr_to_reflect_plotted is True
        x_ticks = gui.chart.x_ticks
        # narrow the window so the first date is excluded
        gui.chart.plotted_x_ticks = pd.Interval(x_ticks[1], x_ticks[-1], closed="left")
        assert pane.plotted_x_ticks[0] == x_ticks[1]
        assert len(pane.mark.x) == len(pane.plotted_x_ticks)
        assert len(pane.mark.y) == len(pane.mark.x)

    def test_line_subplot_reflects_plotted_at_init(self, analy, pp):
        """Verify subplot bars When initial chart bars exclude first bar.

        A line subplot whose initial window excludes the first date should
        have it's mark reflect the plotted window from the outset.
        """
        x_ticks = analy.plot(**pp, display=False).chart.x_ticks
        n = len(x_ticks)
        custom = Subplot(data_creator=_close, kind="lines", title="Close")
        gui = analy.plot(**pp, subplots=[custom], max_ticks=n - 1, display=False)
        pane = gui._subplots[0]
        assert pane.plotted_x_ticks[0] != x_ticks[0]
        assert pane.plotted_x_ticks[0] == x_ticks[1]
        assert len(pane.plotted_x_ticks) == n - 1  # first date excluded
        assert len(pane.mark.x) == len(pane.plotted_x_ticks)

    def test_volume_tooltip_single_symbol(self, analy, pp):
        """Hovering a volume bar shows its value, labelled by the title."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        y = float(pane.data.iloc[0])
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0, "y": y}})
        assert "Volume:" in html
        assert f"{int(y):,}" in html

    def test_close(self, analy, pp):
        """Closing the gui closes subplots without error."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        gui.close()


class TestGuiMultLineSubplots:
    """Tests for subplots beneath a multi-symbol (Compare) price chart."""

    @pytest.fixture
    def comp(self, prices_compare) -> abc.Iterator[analysis.Compare]:
        yield analysis.Compare(prices_compare)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_volume_colors_match_price_lines(self, comp, pp):
        """Each symbol's volume bars take the symbol's price-line color."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        # one volume series per symbol, in symbol (and price-line) order
        assert isinstance(pane.data, pd.DataFrame)
        assert list(pane.data.columns) == comp.symbols
        # volume bars share the main chart's per-symbol colors
        assert list(pane.mark.colors) == list(gui.chart.mark.colors)

    def test_volume_tooltip_identifies_hovered_symbol(self, comp, pp):
        """Stacked-bar tooltip: date, total, and hovered symbol lines.

        The date and total lines take a common color; only the symbol
        line takes the hovered symbol's color.
        """
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        ci = 1  # the second symbol's part of the stack
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0, "colorIndex": ci}})
        symbol = comp.symbols[ci]
        color = list(gui.chart.mark.colors)[ci]
        value = int(pane.data.iloc[0, ci])
        total = int(pane.data.iloc[0].sum())
        bar_color = charts.SubplotBars.TOOLTIP_BAR_COLOR
        bar_line, total_line, symbol_line = html.split("<br>")
        # the date line takes the common bar color, not the symbol's color
        assert f"color: {bar_color}" in bar_line
        assert color not in bar_line
        # the total line (between the others) shows the total, in the bar color
        assert f"value: {total:,}" in total_line
        assert f"color: {bar_color}" in total_line
        # the symbol line shows the symbol and value, in the symbol's color
        assert symbol in symbol_line
        assert f"{value:,}" in symbol_line
        assert f"color: {color}" in symbol_line

    def test_volume_y_axis_spans_stacked_totals(self, comp, pp):
        """The y-axis spans the stacked bar totals (sum across symbols)."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        stacked_totals = pane._plotted_y.sum(axis=1)
        # the extent considered is the per-bar stacked total, not each part
        assert list(pane._plotted_y_extent()) == pytest.approx(list(stacked_totals))
        assert pane.scales["y"].max >= float(stacked_totals.max())

    def test_volume_y_axis_min_is_zero(self, comp, pp):
        """Multi-symbol volume y-axis anchors at zero (full stack in view)."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        assert gui._subplots[0].scales["y"].min == 0

    def test_spec_colors_override_auto_match(self, comp, pp):
        """Explicit `Subplot.colors` take precedence over the auto-match."""
        colors = ["red", "lime", "cyan"]
        custom = Subplot(data_creator=_volume, kind="bars", title="Vol", colors=colors)
        gui = comp.plot(**pp, subplots=[custom], display=False)
        assert list(gui._subplots[0].mark.colors) == colors
