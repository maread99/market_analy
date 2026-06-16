"""Tests for `guis` modules."""

from collections import abc

import pandas as pd
import pytest

from market_analy import analysis, charts
from market_analy.subplots import Subplot, _volume

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

    def test_close(self, analy, pp):
        """Closing the gui closes subplots without error."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        gui.close()
