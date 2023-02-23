"""Mathematics utility functions and classes."""

from __future__ import annotations

import math


def log10_floor(x: int | float) -> int:
    """Floor of log10."""
    return math.floor(math.log10(abs(x)))


def sigfig(x: int | float, sf: int) -> int | float:
    """Return `x` to `sf` significant figures."""
    int_digits = log10_floor(x) + 1
    places_to_round = int_digits - sf
    return round(x, -places_to_round)
