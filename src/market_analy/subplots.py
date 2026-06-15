"""Specification of indicator sub-plots to accompany a price chart.

A "sub-plot" is a separate, customisable chart stacked beneath a price
chart and sharing the price chart's x-axis (for example a classic volume
pane). Multiple sub-plots can be shown simultaneously.

This module defines:

`Subplot`:
    A declarative, validated specification of a single sub-plot.

`SUBPLOT_REGISTRY`:
    Registry of named built-in sub-plots (currently only "volume").

`resolve_subplots`:
    Resolve a sequence of sub-plot specifications, mapping any string to
    the corresponding built-in.

The actual charting of a sub-plot is undertaken by the sub-plot chart
classes defined in `market_analy.charts` (resolved from `Subplot.kind`),
whilst the stacking and coordination of sub-plots beneath a price chart
is undertaken by the price GUIs defined in `market_analy.guis`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from valimp import parse, parse_cls

# A producer evaluates the data for a sub-plot from the price data on
# which the accompanying price chart is based.
SubplotProducer = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]

SubplotKind = Literal["bars", "lines"]


@parse_cls
@dataclass
class Subplot:
    """Specification of a single indicator sub-plot.

    A sub-plot is charted beneath a price chart, sharing the price
    chart's x-axis.

    Parameters
    ----------
    producer
        Callable to evaluate the sub-plot data from the price data on
        which the accompanying price chart is based.

        The callable will receive the price data as a single positional
        argument (a `pd.DataFrame` with columns indexed with a
        `pd.MultiIndex` with level 0 as the symbol and level -1 as
        'open', 'high', 'low', 'close' and 'volume').

        The callable must return a `pd.Series` (single set of values) or
        a `pd.DataFrame` (one column per set of values) indexed with the
        same index as the received price data. This requirement ensures
        the sub-plot's x-ticks align with the shared x-axis.

    kind
        Type of mark with which to plot the sub-plot data:
            "bars" - a bar for each value (for example, volume).
            "lines" - a line through the values (for example, an
            oscillator such as RSI or MACD).

    name
        Name of the sub-plot, shown as the sub-plot's y-axis label.

    colors
        Colors to apply to the sub-plot marks.

    height
        Height of the sub-plot, in pixels.

    ref_levels
        Values at which to plot horizontal reference lines, for example
        [30, 70] to plot reference lines for an oscillator. The sub-plot
        y-axis will be extended as required to ensure all reference
        levels are visible.

    y_tick_format
        Format for the sub-plot y-axis tick labels, as a d3-format
        specifier (for example "~s" for SI-prefixed values).
    """

    producer: SubplotProducer
    kind: SubplotKind = "lines"
    name: str | None = None
    colors: Sequence[str] | None = None
    height: int = 140
    ref_levels: Sequence[float] | None = None
    y_tick_format: str | None = None


def _volume(prices: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Evaluate volume data from price data.

    Returns the 'volume' column(s) of `prices`. If `prices` covers a
    single symbol a `pd.Series` is returned, otherwise a `pd.DataFrame`
    with one column per symbol.
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        if "volume" not in prices.columns:
            raise ValueError("Price data does not include a 'volume' column.")
        return prices["volume"]

    level = prices.columns.nlevels - 1
    if "volume" not in prices.columns.get_level_values(level):
        raise ValueError("Price data does not include a 'volume' column.")
    vol = prices.xs("volume", axis=1, level=level)
    if isinstance(vol, pd.Series):
        return vol
    if len(vol.columns) == 1:
        return vol.iloc[:, 0].rename("volume")
    return vol


def _volume_subplot() -> Subplot:
    """Built-in 'volume' sub-plot."""
    return Subplot(
        producer=_volume,
        kind="bars",
        name="Volume",
        colors=["steelblue"],
        height=120,
        y_tick_format="~s",
    )


# Registry of named built-in sub-plots. Each value is a factory that
# returns a new `Subplot` instance. The registry is extensible: further
# built-ins can be added without any change to the public API.
SUBPLOT_REGISTRY: dict[str, Callable[[], Subplot]] = {
    "volume": _volume_subplot,
}


@parse
def resolve_subplots(subplots: Sequence[str | Subplot]) -> list[Subplot]:
    """Resolve a sequence of sub-plot specifications.

    Parameters
    ----------
    subplots
        Sub-plot specifications. Each item can be either:
            A `str` naming a built-in sub-plot (see `SUBPLOT_REGISTRY`).
            A `Subplot` instance describing a custom sub-plot.

    Returns
    -------
    list[Subplot]
        Resolved specifications, in the order received.

    Examples
    --------
    >>> resolved = resolve_subplots(["volume"])
    >>> [(s.name, s.kind) for s in resolved]
    [('Volume', 'bars')]
    """
    resolved: list[Subplot] = []
    for spec in subplots:
        if isinstance(spec, str):
            try:
                factory = SUBPLOT_REGISTRY[spec]
            except KeyError:
                valid = sorted(SUBPLOT_REGISTRY)
                raise ValueError(
                    f"'{spec}' is not a valid built-in sub-plot. Valid"
                    f" options are {valid}."
                ) from None
            resolved.append(factory())
        else:
            resolved.append(spec)
    return resolved
