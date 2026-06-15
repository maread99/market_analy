"""Tests for the `subplots` module and sub-plot integration with GUIs."""

from collections import abc

import pandas as pd
import pytest
import valimp

from market_analy import analysis, charts
from market_analy.subplots import (
    SUBPLOT_REGISTRY,
    Subplot,
    _volume,
    resolve_subplots,
)


def _close(prices: pd.DataFrame) -> pd.Series:
    """Producer returning close price as a single series."""
    return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")


class TestSubplotSpec:
    """Tests for `Subplot` and `resolve_subplots`."""

    def test_registry_volume(self):
        assert "volume" in SUBPLOT_REGISTRY
        spec = SUBPLOT_REGISTRY["volume"]()
        assert isinstance(spec, Subplot)
        assert spec.kind == "bars"
        assert spec.name == "Volume"
        assert spec.producer is _volume

    def test_resolve_builtin(self):
        resolved = resolve_subplots(["volume"])
        assert len(resolved) == 1
        assert isinstance(resolved[0], Subplot)
        assert resolved[0].name == "Volume"

    def test_resolve_passthrough(self):
        custom = Subplot(producer=_close, kind="lines", name="Close")
        resolved = resolve_subplots([custom, "volume"])
        assert resolved[0] is custom
        assert resolved[1].name == "Volume"

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="not a valid built-in sub-plot"):
            resolve_subplots(["not_a_subplot"])

    def test_validation_bad_producer(self):
        with pytest.raises(valimp.InputsError):
            Subplot(producer="not_callable")

    def test_validation_bad_kind(self):
        with pytest.raises(valimp.InputsError):
            Subplot(producer=_close, kind="scatter")


class TestVolumeProducer:
    """Tests for the built-in `_volume` producer."""

    def test_volume_single_symbol(self, prices_analysis):
        prices = prices_analysis.get(
            start="2023-01-06", end="2023-01-10", composite=False
        )
        vol = _volume(prices)
        assert isinstance(vol, pd.Series)
        assert vol.index.equals(prices.index)
        expected = prices.xs("volume", axis=1, level=-1).iloc[:, 0]
        assert (vol.values == expected.values).all()

    def test_volume_missing_raises(self):
        idx = pd.date_range("2020", periods=3)
        cols = pd.MultiIndex.from_tuples([("X", "close")])
        prices = pd.DataFrame([[1.0], [2.0], [3.0]], index=idx, columns=cols)
        with pytest.raises(ValueError, match="does not include a 'volume' column"):
            _volume(prices)


class TestGuiSubplots:
    """Tests for sub-plots stacked beneath a price chart GUI."""

    @pytest.fixture
    def analy(self, prices_analysis) -> abc.Iterator[analysis.Analysis]:
        yield analysis.Analysis(prices_analysis)

    @pytest.fixture
    def pp(self) -> dict:
        return {"start": pd.Timestamp("2023-01-06"), "end": pd.Timestamp("2023-01-10")}

    def test_no_subplots_default(self, analy, pp):
        """Without `subplots` the gui has no sub-plots."""
        gui = analy.plot(**pp, display=False)
        assert gui._subplots == []
        assert gui.chart.figure in gui._gui_box_contents
        # x-tick labels remain visible on the price chart
        assert "display" not in (gui.chart.axes[0].tick_style or {})

    def test_subplots_shared_scale(self, analy, pp):
        """Sub-plot reuses the price chart's x-axis scale."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        assert len(gui._subplots) == 1
        pane = gui._subplots[0]
        assert isinstance(pane, charts.SubplotBars)
        assert pane.scales["x"] is gui.chart.scales["x"]
        assert pane.x_ticks.equals(gui.chart.x_ticks)

    def test_subplots_in_gui_box(self, analy, pp):
        """Sub-plot figure sits between the price chart and the date slider."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        contents = gui._gui_box_contents
        i_chart = contents.index(gui.chart.figure)
        i_pane = contents.index(gui._subplots[0].figure)
        i_slider = contents.index(gui.date_slider)
        assert i_chart < i_pane < i_slider

    def test_subplots_data(self, analy, pp):
        """Sub-plot data reflects the producer output."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        expected = _volume(gui.prices)
        assert (pane.data.values == expected.values).all()

    def test_subplots_multiple_ordered(self, analy, pp):
        """Multiple sub-plots stack, in order, all sharing the x scale."""
        custom = Subplot(producer=_close, kind="lines", name="Close")
        gui = analy.plot(**pp, subplots=["volume", custom], display=False)
        assert len(gui._subplots) == 2
        assert isinstance(gui._subplots[0], charts.SubplotBars)
        assert isinstance(gui._subplots[1], charts.SubplotLines)
        scale = gui.chart.scales["x"]
        assert all(p.scales["x"] is scale for p in gui._subplots)
        contents = gui._gui_box_contents
        assert contents.index(gui._subplots[0].figure) < contents.index(
            gui._subplots[1].figure
        )

    def test_x_label_policy(self, analy, pp):
        """X-tick labels show only on the bottom-most pane."""
        custom = Subplot(producer=_close, kind="lines", name="Close")
        gui = analy.plot(**pp, subplots=["volume", custom], display=False)
        assert gui.chart.axes[0].tick_style.get("display") == "none"
        assert gui._subplots[0].axes[0].tick_style.get("display") == "none"
        assert "display" not in (gui._subplots[-1].axes[0].tick_style or {})

    def test_shared_scale_lockstep(self, analy, pp):
        """Changing the price chart's plotted window moves the sub-plot too."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        x_ticks = gui.chart.x_ticks
        narrow = pd.Interval(x_ticks[1], x_ticks[-1], closed="left")
        gui.chart.plotted_x_ticks = narrow
        assert pane.plotted_x_ticks.equals(gui.chart.plotted_x_ticks)

    def test_update_subplots(self, analy, pp):
        """Recomputing sub-plots from new prices refreshes the pane data."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        new_prices = gui.prices * 2
        gui._update_subplots(new_prices)
        expected = _volume(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_update_chart_funnel_refreshes_subplots(self, analy, pp):
        """The `_update_chart` funnel refreshes sub-plots when prices change."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        new_prices = gui.prices * 2
        data, data2 = gui._prices_to_chart_data(new_prices)
        gui._update_chart(data, data2, prices=new_prices)
        expected = _volume(new_prices)
        assert (pane.data.values == expected.values).all()

    def test_reset_chart_keeps_subplots(self, analy, pp):
        """Resetting the chart leaves the sub-plot data intact."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        pane = gui._subplots[0]
        gui._reset_chart()
        expected = _volume(gui.prices)
        assert (pane.data.values == expected.values).all()

    def test_ref_levels_add_marks(self, analy, pp):
        """A sub-plot with `ref_levels` adds persistent reference marks."""
        custom = Subplot(producer=_close, kind="lines", name="Close", ref_levels=[1.0])
        gui = analy.plot(**pp, subplots=[custom], display=False)
        pane = gui._subplots[0]
        assert len(pane.added_marks[charts.Groups.PERSIST]) == 1

    def test_close(self, analy, pp):
        """Closing the gui closes sub-plots without error."""
        gui = analy.plot(**pp, subplots=["volume"], display=False)
        gui.close()
