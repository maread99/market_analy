"""Tests for `guis` module.

These tests exercise the wiring of subplots into a price chart gui. The
subplot chart classes themselves are tested directly in `test_charts`.
"""

import pandas as pd
import pytest

from market_analy import analysis, charts

# TODO: tests are extremely incomplete!


class _SubplotClose(charts.SubplotLines):
    """Lines subplot of close price (used as a custom subplot in tests)."""

    TITLE = "Close"

    def data_create(self, prices):
        return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")


class TestGuiSubplots:
    """Tests for subplots stacked beneath a price chart GUI."""

    @pytest.fixture
    def analy(self, prices_analysis) -> analysis.Analysis:
        return analysis.Analysis(prices_analysis)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_no_subplots_default(self, analy, pp):
        """Without `subplots` the gui has no subplots."""
        gui = analy.plot(**pp, display=False)
        assert gui.subplots == []
        assert gui.chart.figure in gui._gui_box_contents
        # x-tick labels remain visible on the price chart
        assert "display" not in gui.chart.axes[0].tick_style

    def test_subplots_alias(self, analy, pp):
        """A `str` alias resolves to the associated subplot class."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        assert len(gui.subplots) == 1
        assert isinstance(gui.subplots[0], charts.SubplotVolume)

    def test_subplots_class(self, analy, pp):
        """A `BaseSubplot` subclass can be passed directly."""
        gui = analy.plot(**pp, subplots=[charts.SubplotVolume], display=False)
        assert isinstance(gui.subplots[0], charts.SubplotVolume)

    def test_unknown_subplot_raises(self, analy, pp):
        with pytest.raises(ValueError, match="not a valid built-in subplot"):
            analy.plot(**pp, subplots=["not_a_subplot"], display=False)

    def test_subplots_shared_scale(self, analy, pp):
        """Subplot reuses the price chart's x-axis scale."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        assert pane.scales["x"] is gui.chart.scales["x"]
        assert pane.x_ticks.equals(gui.chart.x_ticks)

    def test_subplots_in_gui_box(self, analy, pp):
        """Subplot figure sits between the price chart and the date slider."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        contents = gui._gui_box_contents
        i_chart = contents.index(gui.chart.figure)
        i_pane = contents.index(gui.subplots[0].figure)
        i_slider = contents.index(gui.date_slider)
        assert i_chart < i_pane < i_slider

    def test_subplots_data(self, analy, pp):
        """Subplot data is evaluated from the gui's full price data."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        expected = gui.prices.xs("volume", axis=1, level=-1).iloc[:, 0]
        assert (pane.data.values == expected.values).all()

    def test_subplots_multiple_ordered(self, analy, pp):
        """Multiple subplots stack, in order, all sharing the x scale."""
        gui = analy.plot(**pp, subplots=["volume", _SubplotClose], display=False)
        assert len(gui.subplots) == 2
        assert isinstance(gui.subplots[0], charts.SubplotVolume)
        assert isinstance(gui.subplots[1], _SubplotClose)
        scale = gui.chart.scales["x"]
        assert all(sp.scales["x"] is scale for sp in gui.subplots)
        contents = gui._gui_box_contents
        assert contents.index(gui.subplots[0].figure) < contents.index(
            gui.subplots[1].figure
        )

    def test_x_label_policy(self, analy, pp):
        """X-tick labels show only on the bottom-most pane."""
        gui = analy.plot(**pp, subplots=["volume", _SubplotClose], display=False)
        assert gui.chart.axes[0].tick_style.get("display") == "none"
        assert gui.subplots[0].axes[0].tick_style.get("display") == "none"
        assert "display" not in (gui.subplots[-1].axes[0].tick_style or {})

    def test_shared_scale_lockstep(self, analy, pp):
        """Changing the price chart's plotted window moves the subplot too."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        x_ticks = gui.chart.x_ticks
        narrow = pd.Interval(x_ticks[1], x_ticks[-1], closed="left")
        gui.chart.plotted_x_ticks = narrow
        assert pane.plotted_x_ticks.equals(gui.chart.plotted_x_ticks)

    def test_update_subplots(self, analy, pp):
        """Recomputing subplots from new prices refreshes the pane data."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        new_prices = gui.prices * 2
        expected = pane.data_create(new_prices)
        gui._update_subplots(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_update_chart_funnel_refreshes_subplots(self, analy, pp):
        """The `_update_chart` funnel refreshes subplots when prices change."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        new_prices = gui.prices * 2
        expected = pane.data_create(new_prices)
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        assert (pane.data.values == expected.values).all()

    def test_reset_chart_resets_subplots(self, analy, pp):
        """Resetting the chart resets subplot."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        expected = pane.data_create(gui.prices)
        new_prices = gui.prices * 2
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        gui._reset_chart()
        assert (pane.data.values == expected.values).all()

    def test_close(self, analy, pp):
        """Closing the gui closes subplots without error."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        gui.close()


class TestGuiMultLineSubplots:
    """Tests for subplots beneath a multi-symbol (Compare) price chart.

    These verify the full price data (which the rebased close-only chart
    does not itself carry) reaches the subplot's `data_create`.
    """

    @pytest.fixture
    def comp(self, prices_compare) -> analysis.Compare:
        return analysis.Compare(prices_compare)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_volume_data_per_symbol(self, comp, pp):
        """Multi-symbol volume is evaluated from the full price data."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        assert isinstance(pane.data, pd.DataFrame)
        assert list(pane.data.columns) == comp.symbols
        expected = gui.prices.xs("volume", axis=1, level=-1)
        assert (pane.data.values == expected.values).all()

    def test_bar_colors_match_price_lines(self, comp, pp):
        """Volume bars take the main chart's per-symbol price-line colors."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        pane = gui.subplots[0]
        assert list(pane.mark.colors) == list(gui.chart.mark.colors)

    def test_y_axis_min_is_zero(self, comp, pp):
        """Multi-symbol y-axis anchors at zero (full stack in view)."""
        gui = comp.plot(**pp, subplots=["volume"], display=False)
        assert gui.subplots[0].scales["y"].min == 0
