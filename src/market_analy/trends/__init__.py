"""Constructor module for trends subpackage.

Protocols and base classes to evaluate trends.
"""

from __future__ import annotations

import typing

import market_prices as mp
import pandas as pd

# necessary for unpickling test resources. When picked analy was here
# ... i.e. at market_analy.trend. Also serves to provide legacy access.
from .analy import Movement, Movements, Trends, TrendsGui  # noqa: F401

if typing.TYPE_CHECKING:
    from .movements_base import MovementsSupportChartAnaly


class TrendsProto(typing.Protocol):
    """Evaluate trends over OHLC data."""

    data: pd.DataFrame
    interval: mp.intervals.RowInterval

    def get_movements(self) -> MovementsSupportChartAnaly:
        """Evaluate all movements over `data`."""
