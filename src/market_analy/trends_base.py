"""Protocols and base classes to evaluate trends."""

from __future__ import annotations

import typing

import market_prices as mp
import pandas as pd

if typing.TYPE_CHECKING:
    from market_analy.movements_base import MovementsChartProto


class TrendsProto(typing.Protocol):
    """Evaluate trends over OHLC data."""

    data: pd.DataFrame
    interval: mp.intervals.RowInterval

    def get_movements(self) -> MovementsChartProto:
        """Evaluate all movements over `data`."""
