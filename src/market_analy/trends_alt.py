"""Deprecated module. Moved to `market_analy.trends.analy_alt`.

Maintained here for legacy reasons. Provides for maintaining prior
interface and to unpickle test resources that looked here for the Movement
class when pickled.
"""

from .trends.analy import TrendsAlt  # noqa: F401
from .trends.guis import TrendsAltGui as TrendsGui  # noqa: F401
from .trends.movements import MovementAlt as Movement  # noqa: F401
from .trends.movements import MovementsAlt as Movements  # noqa: F401
