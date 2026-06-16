"""Tests for the `subplots` module and subplot integration with GUIs."""

import pandas as pd
import pytest

from market_analy.subplots import (
    SUBPLOT_REGISTRY,
    Subplot,
    _volume,
    resolve_subplots,
)


def _close(prices: pd.DataFrame) -> pd.Series:
    """DataCreator returning close price as a single series."""
    return prices.xs("close", axis=1, level=-1).iloc[:, 0].rename("close")


class TestSubplotSpec:
    """Tests for `Subplot` and `resolve_subplots`."""

    def test_registry_volume(self):
        assert "volume" in SUBPLOT_REGISTRY
        spec = SUBPLOT_REGISTRY["volume"]()
        assert isinstance(spec, Subplot)
        assert spec.kind == "bars"
        assert spec.title == "Volume"
        assert spec.data_creator is _volume

    def test_resolve_builtin(self):
        resolved = resolve_subplots(["volume"])
        assert len(resolved) == 1
        assert isinstance(resolved[0], Subplot)
        assert resolved[0].title == "Volume"

    def test_resolve_passthrough(self):
        custom = Subplot(data_creator=_close, kind="lines", title="Close")
        resolved = resolve_subplots([custom, "volume"])
        assert resolved[0] is custom
        assert resolved[1].title == "Volume"

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="not a valid built-in subplot"):
            resolve_subplots(["not_a_subplot"])


class TestVolumeDataCreator:
    """Tests for the built-in `_volume` data_creator."""

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
