"""Constructor module for trends subpackage.

Protocols and base classes to evaluate trends.
"""

from __future__ import annotations

import typing

# ruff: noqa: F401
# necessary for unpickling test resources. When picked analy was here
# ... i.e. at market_analy.trend. Also serves to provide legacy access.
from .analy import Trends
from .guis import TrendsGui
from .movements import Movement, Movements

if typing.TYPE_CHECKING:
    import market_prices as mp
    import pandas as pd

    from .movements import MovementsSupportChartAnaly


class TrendsProto(typing.Protocol):
    """Evaluate trends over OHLC data."""

    data: pd.DataFrame
    interval: mp.intervals.RowInterval

    def __init__(
        self,
        data: pd.DataFrame,
        interval: mp.intervals.RowInterval,
        *args: typing.Any,
        **kwargs: typing.Any,
    ): ...

    def get_movements(self) -> MovementsSupportChartAnaly:
        """Evaluate all movements over `data`."""
        ...
