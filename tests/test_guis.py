"""Tests for `guis` module."""

import bqplot as bq
import pandas as pd
import pytest

from market_analy import analysis, charts
from market_analy.guis import GuiOHLCCaseBase
from market_analy.utils.bq_utils import dates_to_posix

# NOTE: tests are extremely incomplete! Currently limited to only testing
# Subplots


# --------
# Subplots
# --------

# These subplot tests exercise the wiring of subplots into a price chart gui. The
# subplot chart classes themselves are tested directly in `test_charts`.


@pytest.fixture
def SubplotVol() -> type[charts.BaseSubplot]:
    """Direct subclass of `BaseSubplot` (volume).

    A minimal concrete subplot for exercising base-level gui wiring without
    coupling the tests to the built-in `SubplotVolume` (which could diverge
    from the base class as it's developed). Mimics `SubplotVolume`'s data.
    """

    class _SubplotVol(charts.BaseSubplot):
        TITLE = "Vol"

        @property
        def MarkCls(self) -> type[bq.Mark]:
            return bq.Bars

        def get_subplot_data(self, prices):
            vol = prices.xs("volume", axis=1, level=-1)
            return vol.iloc[:, 0] if vol.shape[1] == 1 else vol

    return _SubplotVol


@pytest.fixture
def SubplotClose() -> type[charts.SubplotLines]:
    class _SubplotClose(charts.SubplotLines):
        """Lines subplot of close price (used as a custom subplot in tests)."""

        TITLE = "Close"

        def get_subplot_data(self, prices):
            return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")

    return _SubplotClose


class TestGuiSubplots:
    """Tests for subplots stacked beneath a price chart GUI."""

    @pytest.fixture
    def analy(self, prices_analysis) -> analysis.Analysis:
        return analysis.Analysis(prices_analysis)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_default_includes_volume(self, analy, pp):
        """`Analysis.plot` includes a volume subplot by default."""
        gui = analy.plot(**pp, display=False)
        assert len(gui.subplots) == 1
        assert isinstance(gui.subplots[0], charts.SubplotVolume)
        # x-tick labels show on the (bottom-most) subplot, not the price chart
        assert gui.chart.axes[0].tick_style.get("display") == "none"

    def test_subplots_false_no_subplots(self, analy, pp):
        """`subplots=False` creates a gui with no subplots."""
        gui = analy.plot(**pp, subplots=False, display=False)
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

    def test_subplots_shared_scale(self, analy, pp, SubplotVol):
        """Subplot reuses the price chart's x-axis scale."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        assert pane.scales["x"] is gui.chart.scales["x"]
        assert pane.x_ticks.equals(gui.chart.x_ticks)

    def test_subplots_in_gui_box(self, analy, pp, SubplotVol):
        """Subplot figure sits between the price chart and the date slider."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        contents = gui._gui_box_contents
        i_chart = contents.index(gui.chart.figure)
        i_pane = contents.index(gui.subplots[0].figure)
        i_slider = contents.index(gui.date_slider)
        assert i_chart < i_pane < i_slider

    def test_subplots_multiple_ordered(self, analy, pp, SubplotVol, SubplotClose):
        """Multiple subplots stack, in order, all sharing the x scale."""
        gui = analy.plot(**pp, subplots=[SubplotVol, SubplotClose], display=False)
        assert len(gui.subplots) == 2
        assert isinstance(gui.subplots[0], SubplotVol)
        assert isinstance(gui.subplots[1], SubplotClose)
        scale = gui.chart.scales["x"]
        assert all(sp.scales["x"] is scale for sp in gui.subplots)
        contents = gui._gui_box_contents
        assert contents.index(gui.subplots[0].figure) < contents.index(
            gui.subplots[1].figure
        )

    def test_x_label_policy(self, analy, pp, SubplotVol, SubplotClose):
        """X-tick labels show only on the bottom-most pane."""
        gui = analy.plot(**pp, subplots=[SubplotVol, SubplotClose], display=False)
        assert gui.chart.axes[0].tick_style.get("display") == "none"
        assert gui.subplots[0].axes[0].tick_style.get("display") == "none"
        assert "display" not in (gui.subplots[-1].axes[0].tick_style or {})

    def test_data(self, analy, pp, SubplotVol):
        """Verify gui passes through full price data."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        assert isinstance(pane.data, pd.Series)
        expected = gui.prices.xs("volume", axis=1, level=-1)
        assert (pane.data.values == expected.values.ravel()).all()

    def test_shared_scale_lockstep(self, analy, pp, SubplotVol):
        """Changing the price chart's plotted window moves the subplot too."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        x_ticks = gui.chart.x_ticks
        narrow = pd.Interval(x_ticks[1], x_ticks[-1], closed="left")
        gui.chart.plotted_x_ticks = narrow
        assert pane.plotted_x_ticks.equals(gui.chart.plotted_x_ticks)

    def test_update_subplots(self, analy, pp, SubplotVol):
        """Recomputing subplots from new prices refreshes the pane data."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        new_prices = gui.prices * 2 + 1
        expected = pane.get_subplot_data(new_prices)
        assert (pane.data.values != expected.values).all()
        gui._update_subplots(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_update_chart_funnel_refreshes_subplots(self, analy, pp, SubplotVol):
        """The `_update_chart` funnel refreshes subplots when prices change."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        new_prices = gui.prices * 2 + 1
        expected = pane.get_subplot_data(new_prices)
        assert (pane.data.values != expected.values).all()
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        assert (pane.data.values == expected.values).all()

    def test_reset_chart_resets_subplots(self, analy, pp, SubplotVol):
        """Resetting the chart resets subplot."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        expected = pane.get_subplot_data(gui.prices)
        assert (pane.data.values == expected.values).all()
        new_prices = gui.prices * 2 + 1
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        assert (pane.data.values != expected.values).all()
        gui._reset_chart()
        assert (pane.data.values == expected.values).all()

    def test_close(self, analy, pp, SubplotVol):
        """Closing the gui closes subplots without error."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        gui.close()


class TestSyncedTooltips:
    """Tests for tooltips synchronised across the chart and subplots."""

    @pytest.fixture
    def analy(self, prices_analysis) -> analysis.Analysis:
        return analysis.Analysis(prices_analysis)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_panes_are_chart_and_subplots(self, analy, pp, SubplotVol, SubplotClose):
        gui = analy.plot(**pp, subplots=[SubplotVol, SubplotClose], display=False)
        assert gui._synced_panes == [gui.chart, *gui.subplots]
        # each pane has a synced-tooltip mark, hidden initially
        for pane in gui._synced_panes:
            assert pane._synced_tooltip_mark.visible is False

    def test_no_subplots_no_panes(self, analy, pp):
        gui = analy.plot(**pp, subplots=False, display=False)
        assert gui._synced_panes == []

    def test_show_synced_tooltips_shows_others_hides_source(
        self, analy, pp, SubplotVol
    ):
        """Hovering one pane shows the others' tooltips, hides the source's."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        chart, pane = gui.chart, gui.subplots[0]
        x = chart.x_ticks[2]
        gui._show_synced_tooltips(x, source=chart)
        # subplot (not the source) shows its tooltip for the bar
        assert pane._synced_tooltip_mark.visible is True
        assert list(pane._synced_tooltip_mark.x) == [x]
        # the source chart shows its own native tooltip, not a synced one
        assert chart._synced_tooltip_mark.visible is False

    def test_clear_hides_all(self, analy, pp, SubplotVol):
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        gui._show_synced_tooltips(gui.chart.x_ticks[2], source=gui.chart)
        gui._clear_synced_tooltips()
        for pane in gui._synced_panes:
            assert pane._synced_tooltip_mark.visible is False

    def test_hover_dispatch_triggers_sync(self, analy, pp, SubplotVol):
        """Firing the chart mark's hover shows the subplot's synced tooltip."""
        gui = analy.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        i = 2
        # invoke the mark's hover callbacks as bqplot's frontend would
        gui.chart.mark._hover_handlers(gui.chart.mark, {"data": {"index": i}})
        assert pane._synced_tooltip_mark.visible is True
        assert list(pane._synced_tooltip_mark.x) == [gui.chart.x_ticks[i]]


class TestScatterSyncedTooltip:
    """Tests for triggering synced tooltips from case scatter marks."""

    def test_scatter_event_x_maps_to_bar(self):
        """A case scatter's hover x maps to the corresponding x-tick.

        A scatter point's `event['data']['x']` is the posix value of the
        bar's x-tick (as set on the shared ordinal scale), which
        `_scatter_event_x` must invert back to the bar's timestamp.
        """
        tick = pd.Timestamp("2023-01-06")
        posix = dates_to_posix(pd.DatetimeIndex([tick]).as_unit("ns"))[0]
        event = {"data": {"x": posix}}
        assert GuiOHLCCaseBase._scatter_event_x(None, event) == tick


class TestGuiMultLineSubplots:
    """Tests for subplots associated with a multi-symbol (Compare) price chart.

    These verify the full price data (which the rebased close-only chart
    does not itself carry) reaches the subplot's `get_subplot_data`.
    """

    @pytest.fixture
    def comp(self, prices_compare) -> analysis.Compare:
        return analysis.Compare(prices_compare)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_default_includes_volume(self, comp, pp):
        """`Compare.plot` includes a volume subplot by default."""
        gui = comp.plot(**pp, display=False)
        assert len(gui.subplots) == 1
        assert isinstance(gui.subplots[0], charts.SubplotVolume)

    def test_subplots_false_no_subplots(self, comp, pp):
        """`subplots=False` creates a gui with no subplots."""
        gui = comp.plot(**pp, subplots=False, display=False)
        assert gui.subplots == []

    def test_data(self, comp, pp, SubplotVol):
        """Verify gui passes through full price data."""
        gui = comp.plot(**pp, subplots=[SubplotVol], display=False)
        pane = gui.subplots[0]
        assert isinstance(pane.data, pd.DataFrame)
        assert list(pane.data.columns) == comp.symbols
        expected = gui.prices.xs("volume", axis=1, level=-1)
        assert (pane.data.values == expected.values).all()
