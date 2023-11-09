"""Deprecated module. Moved to market_analy.trends.analy_alt.

Maintained here for legacy reasons. Provides for maintaining prior
interface and to unpickle test resources that looked here for the Movement
class when pickled.
"""

from .trends.analy_alt import Movement, Movements, Trends, TrendsGui  # noqa: F401
