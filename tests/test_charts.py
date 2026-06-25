"""Tests for the subplot chart classes of the `charts` module."""

import functools
from types import SimpleNamespace

import bqplot as bq
import numpy as np
import pandas as pd
import pytest

import market_analy.utils.bq_utils as ubq
from market_analy import charts
from market_analy.utils.maths_utils import discretize_range_nicely

# NOTE: tests are extremely incomplete! Currently limited to only testing
# Subplots


_FIELDS = ["open", "high", "low", "close", "volume"]


def _make_prices(
    symbols: list[str],
    n: int = 6,
    *,
    multiindex: bool = True,
    with_volume: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OHLCV price data indexed with a `pd.IntervalIndex`.

    Columns are indexed with a `pd.MultiIndex` (symbol, field) unless
    `multiindex` is False (only valid for a single symbol), in which case
    columns are a flat index of fields.
    """
    rng = np.random.default_rng(seed)
    starts = pd.date_range("2023-01-02", periods=n, freq="D")
    index = pd.IntervalIndex.from_arrays(
        starts, starts + pd.Timedelta("1D"), closed="left"
    )
    fields = _FIELDS if with_volume else _FIELDS[:-1]
    frames = {}
    for symbol in symbols:
        df = pd.DataFrame(
            rng.uniform(10, 100, size=(n, len(fields))), index=index, columns=fields
        )
        if with_volume:
            df["volume"] = rng.integers(1_000, 50_000, size=n).astype(float)
        frames[symbol] = df
    prices = pd.concat(frames, axis=1)
    if not multiindex:
        assert len(symbols) == 1
        prices.columns = prices.columns.droplevel(0)
    return prices


# --------
# Subplots
# --------

# The subplot classes are tested directly (rather than via a gui) by
# constructing them with a minimal stand-in for the accompanying price chart
# (see `_mock_chart`) and synthetic price data (see `_make_prices`).


@functools.cache
def _reference_price_chart() -> charts.BasePrice:
    """A price chart against which to validate the `_mock_chart` interface."""
    prices = _make_prices(["REF"])
    close = prices.xs("close", axis=1, level=-1).iloc[:, 0]
    return charts.Line(close, title="")


def _mock_chart(
    prices: pd.DataFrame, colors: list[str] | None = None
) -> SimpleNamespace:
    """Minimal stand-in for the accompanying price chart.

    Exposes only the attributes a subplot accesses on the chart: the
    shared x-axis scale, the plotted interval, the maximum number of ticks
    and the chart mark's colors. The shared scale's domain is populated to
    cover all ticks (as it would be by a real price chart).
    """
    # verify the attributes to be mocked exist on actual chart instance
    ref = _reference_price_chart()
    mocked_attrs = ("scales", "plotted_interval", "max_ticks", "mark")
    assert all(hasattr(ref, name) for name in mocked_attrs)
    assert "x" in ref.scales
    assert hasattr(ref.mark, "colors")

    index = prices.index
    x_scale = bq.OrdinalScale()
    x_scale.domain = list(ubq.dates_to_posix(index.left))
    return SimpleNamespace(
        scales={"x": x_scale},
        plotted_interval=pd.Interval(index[0].left, index[-1].right, closed="left"),
        max_ticks=None,
        mark=SimpleNamespace(colors=list(colors) if colors else []),
    )


# Subplot class fixtures. Each returns a thin subclass created for the purpose
# of testing functionality at the level of the corresponding base class.


@pytest.fixture
def SubplotBase() -> type[charts.BaseSubplot]:
    """Thin direct subclass of `BaseSubplot` (close price)."""

    class SubplotBase(charts.BaseSubplot):
        TITLE = "Base"

        @property
        def MarkCls(self) -> type[bq.Mark]:
            return bq.Bars

        def get_subplot_data(self, prices):
            data = prices.xs("close", axis=1, level=-1)
            return data.iloc[:, 0] if data.shape[1] == 1 else data

    return SubplotBase


@pytest.fixture
def SubplotBaseRef(SubplotBase) -> type[charts.BaseSubplot]:
    """`SubplotBase` with a reference level."""

    class SubplotBaseRef(SubplotBase):
        REF_LEVELS = (50.0,)

    return SubplotBaseRef


@pytest.fixture
def SubplotBaseColors(SubplotBase) -> type[charts.BaseSubplot]:
    """`SubplotBase` with explicit colors."""

    class SubplotBaseColors(SubplotBase):
        COLORS = ("magenta",)

    return SubplotBaseColors


@pytest.fixture
def SubplotVol() -> type[charts.SubplotBars]:
    """Thin subclass of `SubplotBars` (volume), akin to `SubplotVolume`."""

    class SubplotVol(charts.SubplotBars):
        TITLE = "Vol"

        def get_subplot_data(self, prices):
            vol = prices.xs("volume", axis=1, level=-1)
            return vol.iloc[:, 0] if vol.shape[1] == 1 else vol

    return SubplotVol


@pytest.fixture
def SubplotClose() -> type[charts.SubplotLines]:
    """Thin subclass of `SubplotLines` (close price)."""

    class SubplotClose(charts.SubplotLines):
        TITLE = "Close"

        def get_subplot_data(self, prices):
            return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")

    return SubplotClose


class TestResolveSubplotClass:
    """Tests for `resolve_subplot_class` and `SUBPLOTS`."""

    def test_aliases_registered(self):
        assert charts.SUBPLOTS["volume"] is charts.SubplotVolume

    def test_resolve_alias(self):
        assert charts.resolve_subplot_class("volume") is charts.SubplotVolume

    def test_resolve_subclass_passthrough(self, SubplotClose):
        f = charts.resolve_subplot_class
        assert f(SubplotClose) is SubplotClose
        assert f(charts.SubplotVolume) is charts.SubplotVolume

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="not a valid built-in subplot"):
            charts.resolve_subplot_class("not_a_subplot")

    def test_non_subclass_type_raises(self):
        with pytest.raises(TypeError, match="must be a `BaseSubplot` subclass"):
            charts.resolve_subplot_class(int)

    def test_instance_raises(self):
        match = "must be a `BaseSubplot` subclass"
        with pytest.raises(TypeError, match=match):
            charts.resolve_subplot_class(object())
        with pytest.raises(TypeError, match=match):
            charts.resolve_subplot_class(3)


class TestAbstract:
    """The base and intermediary subplot classes cannot be instantiated."""

    @pytest.mark.parametrize(
        "cls", [charts.BaseSubplot, charts.SubplotBars, charts.SubplotLines]
    )
    def test_cannot_instantiate(self, cls):
        with pytest.raises(TypeError, match="abstract"):
            cls(None, None)


class TestBaseSubplot:
    """Tests for functionality defined on `BaseSubplot`."""

    def test_y_axis_ticks_are_nicely_discretized(self, SubplotBase):
        """Y-axis ticks are nicely_discretized and equally spaced."""
        prices = _make_prices(["AZN.L"])
        pane = SubplotBase(_mock_chart(prices), prices)
        ticks = list(pane.axes[1].tick_values)
        scale = pane.scales["y"]
        assert ticks == pytest.approx(discretize_range_nicely(scale.min, scale.max))
        assert scale.min <= min(ticks)
        assert scale.max >= max(ticks)

    def test_default_colors_single_symbol_none(self, SubplotBase):
        prices = _make_prices(["AZN.L"])
        pane = SubplotBase(_mock_chart(prices, ["red"]), prices)
        assert pane._default_colors(pane.data) is None

    def test_default_colors_multi_symbol_from_chart(self, SubplotBase):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        colors = ["yellow", "orange"]
        pane = SubplotBase(_mock_chart(prices, colors), prices)
        # a multi-symbol subplot defaults to the accompanying chart's colors
        assert list(pane.mark.colors) == colors

    def test_explicit_colors(self, SubplotBaseColors):
        prices = _make_prices(["AZN.L"])
        pane = SubplotBaseColors(_mock_chart(prices), prices)
        assert list(pane.mark.colors) == ["magenta"]

    def test_explicit_colors_override_default(self, SubplotBaseColors):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        # chart colors would otherwise be applied for a multi-symbol subplot
        pane = SubplotBaseColors(_mock_chart(prices, ["yellow", "orange"]), prices)
        assert list(pane.mark.colors) == ["magenta"]

    def test_ref_levels_add_persistent_marks(self, SubplotBaseRef):
        prices = _make_prices(["AZN.L"])
        pane = SubplotBaseRef(_mock_chart(prices), prices)
        assert len(pane.added_marks[charts.Groups.PERSIST]) == 1

    def test_no_ref_levels_no_marks(self, SubplotBase):
        prices = _make_prices(["AZN.L"])
        pane = SubplotBase(_mock_chart(prices), prices)
        assert not pane.added_marks[charts.Groups.PERSIST]

    def test_domain_change_to_foreign_ticks_no_error(self, SubplotBase):
        """Handler tolerates a shared domain selecting none of the subplot's ticks.

        The subplot shares the price chart's x-scale. During an update like
        a tick-interval change the chart can set the shared domain to ticks
        that are not included in the subplot's not-yet-updated data.
        The domain-change handler must skip such a transient empty window
        rather than raise.
        """
        prices = _make_prices(["AZN.L"], n=6)
        chart = _mock_chart(prices)
        pane = SubplotBase(chart, prices)
        foreign = list(ubq.dates_to_posix(pd.DatetimeIndex(["2099-01-01"])))
        # setting the shared domain fires the subplot's _x_domain_chg_handler handler,
        # the handler must not raise an error despite the fact that none of the values
        # in the new domain are included in the subplots current (stale) data.
        chart.scales["x"].domain = foreign
        assert len(pane.plotted_x_ticks) == 0


class TestSubplotBars:
    """Tests for functionality defined on `SubplotBars`."""

    def test_mark_cls_and_title(self, SubplotVol):
        prices = _make_prices(["AZN.L"])
        pane = SubplotVol(_mock_chart(prices), prices)
        assert pane.MarkCls is bq.Bars
        assert pane.title == "Vol"

    def test_y_axis_min_non_negative(self, SubplotVol):
        prices = _make_prices(["AZN.L"])
        pane = SubplotVol(_mock_chart(prices), prices)
        y_min = pane.scales["y"].min
        lo = float(pane._plotted_y.values.min())
        assert 0 <= y_min <= lo
        # the figure must not pad the y-scale beyond the set min/max (else
        # the rendered axis can extend below zero given the otherwise 0 baseline)
        assert pane.scales["y"].allow_padding is False

    def test_multi_symbol_extent_is_stacked_totals(self, SubplotVol):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        pane = SubplotVol(_mock_chart(prices, ["yellow", "orange"]), prices)
        stacked_totals = prices.xs("volume", axis=1, level=-1).sum(axis=1)
        assert list(pane._plotted_y_extent()) == pytest.approx(list(stacked_totals))
        assert pane.scales["y"].max >= float(stacked_totals.max())

    def test_multi_symbol_y_axis_min_is_zero(self, SubplotVol):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        pane = SubplotVol(_mock_chart(prices, ["yellow", "orange"]), prices)
        # y min should be 0 to ensure all parts of each stacked bar are visible
        assert pane.scales["y"].min == 0

    def test_tooltip_single_symbol(self, SubplotVol):
        prices = _make_prices(["AZN.L"])
        pane = SubplotVol(_mock_chart(prices), prices)
        y = float(pane.data.iloc[0])
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0}})
        assert f"Bar: {prices.index[0].left.strftime('%Y-%m-%d')}" in html
        assert f"{pane.title}: {int(y):,}" in html

    def test_tooltip_multi_symbol_identifies_hovered(self, SubplotVol):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        colors = ["yellow", "orange"]
        pane = SubplotVol(_mock_chart(prices, colors), prices)
        ci = 1  # the second symbol's part of the stack
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0, "colorIndex": ci}})
        total = int(pane.data.iloc[0].sum())
        value = int(pane.data.iloc[0, ci])
        dflt_color = "white"
        ts_line, total_line, symbol_line = html.split("<br>")
        # the bar (timestamp) and total lines take the default color
        assert f"Bar: {prices.index[0].left.strftime('%Y-%m-%d')}" in html
        assert f"color: {dflt_color}" in ts_line  # check tag includes expected color
        assert f"Value: {total:,}" in total_line
        assert f"color: {dflt_color}" in total_line
        # the symbol line shows the symbol and value in the symbol's color
        assert symbols[ci] in symbol_line
        assert f"{value:,}" in symbol_line
        assert f"color: {colors[ci]}" in symbol_line


class TestSubplotLines:
    """Tests for functionality defined on `SubplotLines`."""

    def test_mark_cls_and_title(self, SubplotClose):
        prices = _make_prices(["AZN.L"])
        pane = SubplotClose(_mock_chart(prices), prices)
        assert pane.MarkCls is bq.Lines
        assert pane.title == "Close"

    def test_mark_reflects_plotted_on_domain_change(self, SubplotClose):
        """Narrowing the shared x-scale domain refreshes the line's data."""
        prices = _make_prices(["AZN.L"], n=8)
        chart = _mock_chart(prices)
        pane = SubplotClose(chart, prices)
        assert pane._update_mark_data_attr_to_reflect_plotted is True
        posix = pane._x_ticks_posix_raw()
        # narrow domain so the first tick is excluded
        chart.scales["x"].domain = list(posix[1:-1])
        assert pane.plotted_x_ticks[0] != pane.x_ticks[0]
        assert (pane.mark.x == pane.plotted_x_ticks.values).all()
        assert len(pane.mark.y) == len(pane.mark.x)


class TestSubplotVolume:
    """Tests for the built-in `SubplotVolume` subplot."""

    def test_concretised_attributes(self):
        assert issubclass(charts.SubplotVolume, charts.SubplotBars)
        assert charts.SubplotVolume.TITLE == "Volume"
        assert charts.SubplotVolume.Y_TICK_FORMAT == "~s"

    def test_single_symbol_series(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        assert isinstance(pane, charts.SubplotBars)
        expected = prices.xs("volume", axis=1, level=-1).iloc[:, 0]
        assert (pane.data.values == expected.values).all()
        assert pane.data.index.equals(prices.index)

    def test_flat_columns_series(self):
        prices = _make_prices(["AZN.L"], multiindex=False)
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        assert isinstance(pane.data, pd.Series)
        assert (pane.data.values == prices["volume"].values).all()

    def test_missing_volume_col_raises_on_construction(self):
        prices = _make_prices(["AZN.L"], with_volume=False)
        with pytest.raises(ValueError, match="does not include a 'volume' column"):
            charts.SubplotVolume(_mock_chart(prices), prices)


@pytest.fixture
def SubplotColored() -> type[charts.SubplotLineColored]:
    """Thin `SubplotLineColored` subclass plotting close price."""

    class _SubplotColored(charts.SubplotLineColored):
        TITLE = "Close"

        def get_subplot_data(self, prices):
            return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")

    return _SubplotColored


class TestSubplotLineColored:
    """Tests for the value-coloured line subplot base class."""

    def test_is_lines_subplot(self, SubplotColored):
        prices = _make_prices(["AZN.L"])
        pane = SubplotColored(_mock_chart(prices), prices)
        assert isinstance(pane, charts.SubplotLines)
        assert pane.MarkCls is bq.Lines

    def test_line_is_segmented_per_pair_of_bars(self, SubplotColored):
        """The line is plotted as one bqplot line per adjacent pair of bars.

        Verifies `_create_mark` returns a segmented mark.
        """
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        n = len(prices)
        mark = pane._create_mark()
        assert np.asarray(mark.x).shape == (n - 1, 2)
        assert np.asarray(mark.y).shape == (n - 1, 2)
        assert len(np.asarray(mark.color)) == n - 1

    def test_segment_colour_is_mean_of_end_bars(self, SubplotColored):
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        values = pane.data.to_numpy()
        expected = (values[:-1] + values[1:]) / 2.0
        np.testing.assert_allclose(np.asarray(pane.mark.color), expected)

    def test_colour_scale_spans_full_value_range(self, SubplotColored):
        """The colour scale is fixed to the full value range over the data."""
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        values = pane.data.to_numpy()
        scale = pane.scales["color"]
        assert scale.colors == ["blue", "red"]
        assert scale.min == pytest.approx(np.nanmin(values))
        assert scale.max == pytest.approx(np.nanmax(values))

    def test_event_x_maps_segment_to_left_bar(self, SubplotColored):
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        x = pane._event_x(pane.mark, {"data": {"index": 3}})
        assert x == pane.plotted_x_ticks[3]


class TestSyncedTooltip:
    """Tests for the `SyncedTooltip` mixin (via a subplot host)."""

    def test_y_values_indexed_by_x_ticks(self, SubplotColored):
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        series = pane._synced_tooltip_y_values()
        assert series.index.equals(pane.x_ticks)
        np.testing.assert_allclose(series.to_numpy(), pane.data.to_numpy())

    def test_show_and_hide(self, SubplotColored):
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        pane._init_synced_tooltip()
        label = pane._synced_tooltip_mark
        assert label.visible is False
        x = pane.x_ticks[4]
        pane.show_synced_tooltip(x)
        assert label.visible is True
        assert list(label.x) == [x]
        assert label.y[0] == pytest.approx(pane.data.to_numpy()[4])
        assert label.text[0].startswith("Close: ")
        pane.hide_synced_tooltip()
        assert label.visible is False

    def test_show_hidden_when_no_value_at_x(self, SubplotColored):
        prices = _make_prices(["AZN.L"], n=8)
        pane = SubplotColored(_mock_chart(prices), prices)
        pane._init_synced_tooltip()
        pane.show_synced_tooltip(pane.x_ticks[4])  # show first
        foreign_x = pane.x_ticks[-1] + pd.Timedelta("365D")
        pane.show_synced_tooltip(foreign_x)
        assert pane._synced_tooltip_mark.visible is False

    def test_volume_subplot_formats_value(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        pane._init_synced_tooltip()
        pane.show_synced_tooltip(pane.x_ticks[2])
        text = pane._synced_tooltip_mark.text[0]
        assert text.startswith("Volume: ")
        # value formatted with thousands separator (no decimal for integers)
        assert "." not in text.split(": ")[1]

    def test_multi_symbol_bars_show_total(self):
        """A multi-symbol bars subplot shows the stacked total, in white."""
        prices = _make_prices(["AZN.L", "MSFT"])
        pane = charts.SubplotVolume(_mock_chart(prices, ["yellow", "orange"]), prices)
        pane._init_synced_tooltip()
        x = pane.x_ticks[2]
        pane.show_synced_tooltip(x)
        label = pane._synced_tooltip_mark
        total = float(prices.xs("volume", axis=1, level=-1).iloc[2].sum())
        assert label.visible is True
        assert label.y[0] == pytest.approx(total)
        assert label.text[0] == f"Volume: {pane._format_value(total)}"
        # the native multi-symbol tooltip is multi-coloured, so the synced
        # tooltip falls back to the default (white)
        assert label.colors == ["white"]

    def test_ohlc_synced_tooltip_color_reflects_up_down(self):
        """The OHLC synced tooltip colour matches the bar's up/down colour."""
        prices = _make_prices(["AZN.L"], multiindex=False)
        chart = charts.OHLC(prices, title="AZN.L")
        chart._init_synced_tooltip()
        up_colour, down_colour = chart.mark.colors[0], chart.mark.colors[1]
        for i in range(len(prices)):
            x = chart.x_ticks[i]
            chart.show_synced_tooltip(x)
            row = prices.iloc[i]
            expected = up_colour if row["close"] >= row["open"] else down_colour
            assert chart._synced_tooltip_mark.colors == [expected]
