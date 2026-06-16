"""Specification of indicator subplots to accompany a price chart.

A "subplot" is a separate, customisable chart stacked beneath a price
chart and sharing the price chart's x-axis (for example a classic volume
pane). Multiple subplots can be shown simultaneously.

This module includes:

`Subplot`:
    A declarative, validated specification to create a single subplot.

`SUBPLOT_REGISTRY`:
    Registry of named built-in subplots.

Notes
-----
The actual charting of a subplot is undertaken by the subplot chart
classes defined in `market_analy.charts`, whilst the stacking and
coordination of subplots within a gui is implemented within the `BasePrice`
gui class defined in `market_analy.guis`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated

import pandas as pd
from valimp import Coerce, parse, parse_cls

from market_analy.charts import SubplotKind

# A Data Creator evaluates the data for a subplot from the price data on
# which the accompanying price chart is based.
SubplotDataCreator = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@parse_cls
@dataclass
class Subplot:
    """Specification of a single indicator subplot.

    A subplot is charted beneath a price chart, sharing the price
    chart's x-axis.

    Parameters
    ----------
    data_creator
        Callable to evaluate the subplot data from the price data on
        which the accompanying price chart is based.

        The callable will receive the price data as a single positional
        argument (a `pd.DataFrame` with columns indexed with a
        `pd.MultiIndex` with level 0 as the symbol and level -1 as
        'open', 'high', 'low', 'close' and 'volume').

        The callable must return a `pd.Series` (single set of values) or
        a `pd.DataFrame` (one column per set of values) indexed with the
        same index as the received price data. This requirement ensures
        the subplot's x-ticks align with the shared x-axis.

    kind
        Type of mark with which to plot data, as a
        `market_analy.charts.SubplotKind` member or its string value.
        Options include:
            `SubplotKind.BARS` ("bars") - a bar for each value (for
            example, volume).
            `SubplotKind.LINES` ("lines") - a line through the values.

    title
        Name of the subplot, shown as the subplot's title.

    height
        Height of the subplot, in pixels.

    ref_levels
        Values at which to plot horizontal reference lines. The subplot
        y-axis will be extended as required to ensure all reference
        levels are visible.

    y_tick_format
        Format for the subplot y-axis tick labels, as a d3-format
        specifier (for example ".1%" -> 12.3%, i.e. multiply by 100 and
        state to 1 decimal place, or ".1s" to format a number using SI
        prefixes with one significant decimal place).
    """

    data_creator: SubplotDataCreator
    kind: Annotated[str | SubplotKind, Coerce(SubplotKind)] = SubplotKind.LINES
    title: str | None = None
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
    """Built-in 'volume' subplot."""
    return Subplot(
        data_creator=_volume,
        kind=SubplotKind.BARS,
        title="Volume",
        height=140,
        y_tick_format=".1s",
    )


# Registry of named built-in subplots. Each value is a factory that
# returns a new `Subplot` instance. The registry is extensible: further
# built-ins can be added without any change to the public API.
SUBPLOT_REGISTRY: dict[str, Callable[[], Subplot]] = {
    "volume": _volume_subplot,
}


@parse
def resolve_subplots(subplots: Sequence[str | Subplot]) -> list[Subplot]:
    """Resolve a sequence of subplot specifications.

    Parameters
    ----------
    subplots
        Subplot specifications. Each item can be either:
            A `str` naming a built-in subplot (see `SUBPLOT_REGISTRY`).
            A `Subplot` instance describing a custom subplot.

    Returns
    -------
    list[Subplot]
        Resolved specifications, in the order received.

    Examples
    --------
    >>> resolved = resolve_subplots(["volume"])
    >>> [(s.title, s.kind.value) for s in resolved]
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
                    f"'{spec}' is not a valid built-in subplot. Valid"
                    f" options are {valid}."
                ) from None
            resolved.append(factory())
        else:
            resolved.append(spec)
    return resolved
