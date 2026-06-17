"""Mathematics utility functions and classes."""

import math


def log10_floor(x: float) -> int:
    """Floor of log10."""
    return math.floor(math.log10(abs(x)))


def sigfig(x: float, sf: int) -> int | float:
    """Return `x` to `sf` significant figures."""
    int_digits = log10_floor(x) + 1
    places_to_round = int_digits - sf
    return round(x, -places_to_round)


def nice_ticks(vmin: float, vmax: float, max_ticks: int = 5) -> list[float]:
    """Return 'nice', equally spaced tick values spanning a range.

    Parameters
    ----------
    vmin
        Lower limit of the range to span.

    vmax
        Upper limit of the range to span.

    max_ticks
        Maximum number of ticks (the actual number returned may be lower).

    Returns
    -------
    list[float]
        Round values, each a multiple of a 'nice' step (1, 2, 2.5 or 5
        times a power of ten), that fall within [`vmin`, `vmax`].

    Examples
    --------
    >>> nice_ticks(0, 135_000_000)
    [0.0, 50000000.0, 100000000.0]
    >>> nice_ticks(0, 100)
    [0.0, 25.0, 50.0, 75.0, 100.0]
    """
    if not vmax > vmin or max_ticks < 2:
        return [float(vmin)]
    raw_step = (vmax - vmin) / (max_ticks - 1)
    mag = 10.0 ** log10_floor(raw_step)
    step = next(n * mag for n in (1, 2, 2.5, 5, 10) if n * mag >= raw_step)
    first = math.ceil(vmin / step)
    last = math.floor(vmax / step)
    ticks = [round(i * step, 12) for i in range(first, last + 1)]
    return ticks or [float(vmin), float(vmax)]
