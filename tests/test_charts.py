"""Tests for the subplot chart classes of the `charts` module.

The subplot classes are tested directly (rather than via a gui) by
constructing them with a minimal stand-in for the accompanying price chart
(see `_mock_chart`) and synthetic price data (see `_make_prices`).
"""

from types import SimpleNamespace

import bqplot as bq
import numpy as np
import pandas as pd
import pytest

import market_analy.utils.bq_utils as ubq
from market_analy import charts
from market_analy.utils.maths_utils import discretize_range_nicely

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
    for i, symbol in enumerate(symbols):
        df = pd.DataFrame(
            rng.uniform(10, 100, size=(n, len(fields))), index=index, columns=fields
        )
        if with_volume:
            df["volume"] = rng.integers(1_000, 50_000, size=n).astype(float) + i
        frames[symbol] = df
    prices = pd.concat(frames, axis=1)
    if not multiindex:
        assert len(symbols) == 1
        prices.columns = prices.columns.droplevel(0)
    return prices


def _mock_chart(
    prices: pd.DataFrame, colors: list[str] | None = None
) -> SimpleNamespace:
    """Minimal stand-in for the accompanying price chart.

    Exposes only the attributes a subplot accesses on the chart: the
    shared x-axis scale, the plotted interval, the maximum number of ticks
    and the chart mark's colors. The shared scale's domain is populated to
    cover all ticks (as it would be by a real price chart).
    """
    index = prices.index
    x_scale = bq.OrdinalScale()
    x_scale.domain = list(ubq.dates_to_posix(index.left))
    return SimpleNamespace(
        scales={"x": x_scale},
        plotted_interval=pd.Interval(index[0].left, index[-1].right, closed="left"),
        max_ticks=None,
        mark=SimpleNamespace(colors=list(colors) if colors else []),
    )


class _SubplotClose(charts.SubplotLines):
    """Lines subplot of close price (a single series)."""

    TITLE = "Close"

    def data_create(self, prices):
        return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")


class _SubplotCloseRef(_SubplotClose):
    """`_SubplotClose` with a reference level."""

    REF_LEVELS = (50.0,)


class _SubplotCloseColors(_SubplotClose):
    """`_SubplotClose` with explicit colors."""

    COLORS = ("magenta",)


class _SubplotVolumeColors(charts.SubplotVolume):
    """`SubplotVolume` with explicit colors."""

    COLORS = ("red", "lime", "cyan")


class TestResolveSubplotClass:
    """Tests for `resolve_subplot_class` and `SUBPLOT_ALIASES`."""

    def test_alias_registered(self):
        assert charts.SUBPLOT_ALIASES["volume"] is charts.SubplotVolume

    def test_resolve_alias(self):
        assert charts.resolve_subplot_class("volume") is charts.SubplotVolume

    def test_resolve_subclass_passthrough(self):
        assert charts.resolve_subplot_class(_SubplotClose) is _SubplotClose
        assert (
            charts.resolve_subplot_class(charts.SubplotVolume) is charts.SubplotVolume
        )

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="not a valid built-in subplot"):
            charts.resolve_subplot_class("not_a_subplot")

    def test_non_subclass_type_raises(self):
        with pytest.raises(TypeError, match="must be a `BaseSubplot` subclass"):
            charts.resolve_subplot_class(int)

    def test_instance_raises(self):
        with pytest.raises(TypeError, match="must be a `BaseSubplot` subclass"):
            charts.resolve_subplot_class(object())


class TestAbstract:
    """The base and intermediary subplot classes cannot be instantiated."""

    @pytest.mark.parametrize(
        "cls", [charts.BaseSubplot, charts.SubplotBars, charts.SubplotLines]
    )
    def test_cannot_instantiate(self, cls):
        with pytest.raises(TypeError, match="abstract"):
            cls(None, None)


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
        assert pane.MarkCls is bq.Bars
        assert pane.title == "Volume"
        assert isinstance(pane.data, pd.Series)
        expected = prices.xs("volume", axis=1, level=-1).iloc[:, 0]
        assert (pane.data.values == expected.values).all()
        assert pane.data.index.equals(prices.index)

    def test_flat_columns_series(self):
        prices = _make_prices(["AZN.L"], multiindex=False)
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        assert isinstance(pane.data, pd.Series)
        assert (pane.data.values == prices["volume"].values).all()

    def test_multi_symbol_dataframe(self):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        colors = ["yellow", "orange"]
        pane = charts.SubplotVolume(_mock_chart(prices, colors), prices)
        assert isinstance(pane.data, pd.DataFrame)
        assert list(pane.data.columns) == symbols
        # default colors taken from the accompanying chart's mark colors
        assert list(pane.mark.colors) == colors

    def test_missing_volume_raises_on_construction(self):
        prices = _make_prices(["AZN.L"], with_volume=False)
        with pytest.raises(ValueError, match="does not include a 'volume' column"):
            charts.SubplotVolume(_mock_chart(prices), prices)

    def test_data_create_missing_volume_raises(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        bad = _make_prices(["AZN.L"], with_volume=False)
        with pytest.raises(ValueError, match="does not include a 'volume' column"):
            pane.data_create(bad)


class TestSubplotBars:
    """Tests for `SubplotBars` presentation (via `SubplotVolume`)."""

    def test_y_axis_min_non_negative(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        y_min = pane.scales["y"].min
        lo = float(pane._plotted_y.values.min())
        assert 0 <= y_min <= lo
        # the figure must not pad the y-scale beyond the set min/max (else
        # the rendered axis can extend below zero given the bar baseline at 0)
        assert pane.scales["y"].allow_padding is False

    def test_y_axis_ticks_are_nice(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        ticks = list(pane.axes[1].tick_values)
        scale = pane.scales["y"]
        assert ticks == pytest.approx(discretize_range_nicely(scale.min, scale.max))
        assert scale.min <= min(ticks)
        assert scale.max >= max(ticks)

    def test_multi_symbol_extent_is_stacked_totals(self):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        pane = charts.SubplotVolume(_mock_chart(prices, ["yellow", "orange"]), prices)
        stacked_totals = pane._plotted_y.sum(axis=1)
        assert list(pane._plotted_y_extent()) == pytest.approx(list(stacked_totals))
        assert pane.scales["y"].max >= float(stacked_totals.max())

    def test_multi_symbol_y_axis_min_is_zero(self):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        pane = charts.SubplotVolume(_mock_chart(prices, ["yellow", "orange"]), prices)
        assert pane.scales["y"].min == 0

    def test_explicit_colors_override_default(self):
        symbols = ["AZN.L", "BARC.L", "MSFT"]
        prices = _make_prices(symbols)
        # chart colors would otherwise be applied for a multi-symbol subplot
        chart = _mock_chart(prices, ["yellow", "orange", "pink"])
        pane = _SubplotVolumeColors(chart, prices)
        assert list(pane.mark.colors) == ["red", "lime", "cyan"]

    def test_tooltip_single_symbol(self):
        prices = _make_prices(["AZN.L"])
        pane = charts.SubplotVolume(_mock_chart(prices), prices)
        y = float(pane.data.iloc[0])
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0}})
        assert "Bar:" in html
        assert "Volume:" in html
        assert f"{int(y):,}" in html

    def test_tooltip_multi_symbol_identifies_hovered(self):
        symbols = ["AZN.L", "BARC.L"]
        prices = _make_prices(symbols)
        colors = ["yellow", "orange"]
        pane = charts.SubplotVolume(_mock_chart(prices, colors), prices)
        ci = 1  # the second symbol's part of the stack
        html = pane._tooltip_value(pane.mark, {"data": {"index": 0, "colorIndex": ci}})
        total = int(pane.data.iloc[0].sum())
        value = int(pane.data.iloc[0, ci])
        dflt_color = charts.SubplotBars.TOOLTIP_TEXT_COLOR
        ts_line, total_line, symbol_line = html.split("<br>")
        # the bar (timestamp) and total lines take the default color
        assert f"color: {dflt_color}" in ts_line
        assert f"Value: {total:,}" in total_line
        assert f"color: {dflt_color}" in total_line
        # the symbol line shows the symbol and value in the symbol's color
        assert symbols[ci] in symbol_line
        assert f"{value:,}" in symbol_line
        assert f"color: {colors[ci]}" in symbol_line


class TestSubplotLines:
    """Tests for `SubplotLines` (via the `_SubplotClose` subclass)."""

    def test_mark_cls_and_title(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotClose(_mock_chart(prices), prices)
        assert pane.MarkCls is bq.Lines
        assert pane.title == "Close"

    def test_reflects_plotted_flag(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotClose(_mock_chart(prices), prices)
        assert pane._update_mark_data_attr_to_reflect_plotted is True

    def test_ref_levels_add_persistent_marks(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotCloseRef(_mock_chart(prices), prices)
        assert len(pane.added_marks[charts.Groups.PERSIST]) == 1

    def test_no_ref_levels_no_marks(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotClose(_mock_chart(prices), prices)
        assert not pane.added_marks[charts.Groups.PERSIST]

    def test_default_colors_single_symbol_none(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotClose(_mock_chart(prices, ["red"]), prices)
        assert pane._default_colors(pane.data) is None

    def test_explicit_colors(self):
        prices = _make_prices(["AZN.L"])
        pane = _SubplotCloseColors(_mock_chart(prices), prices)
        assert list(pane.mark.colors) == ["magenta"]

    def test_mark_reflects_plotted_on_domain_change(self):
        """Narrowing the shared x-scale domain refreshes the line's data."""
        prices = _make_prices(["AZN.L"], n=8)
        chart = _mock_chart(prices)
        pane = _SubplotClose(chart, prices)
        posix = pane._x_ticks_posix_raw()
        # narrow domain so the first tick is excluded
        chart.scales["x"].domain = list(posix[1:])
        assert pane.plotted_x_ticks[0] != pane.x_ticks[0]
        assert len(pane.mark.x) == len(pane.plotted_x_ticks)
        assert len(pane.mark.y) == len(pane.mark.x)
