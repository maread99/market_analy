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


def discretize_range_nicely(
    vmin: float, vmax: float, max_ticks: int = 5
) -> list[float]:
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
        List of floats that are sequential multiples of a 'nice' step and
        all fall within [`vmin`, `vmax`].

    Examples
    --------
    >>> discretize_range_nicely(0, 135_000_000)
    [0.0, 50000000.0, 100000000.0]
    >>> discretize_range_nicely(0, 100)
    [0.0, 25.0, 50.0, 75.0, 100.0]
    >>> discretize_range_nicely(0, 100, 4)
    [0.0, 50.0, 100.0]
    >>> discretize_range_nicely(0, 100, 6)
    [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    >>> discretize_range_nicely(0, 0.8, 5)
    [0.0, 0.2, 0.4, 0.6, 0.8]
    """
    if not vmax > vmin or max_ticks < 2:
        return [float(vmin)]
    raw_step = (vmax - vmin) / (max_ticks - 1)
    mag = 10.0 ** log10_floor(raw_step)
    step = next(n * mag for n in (1, 2, 2.5, 5, 10) if n * mag >= raw_step)
    first_step_fctr = math.ceil(vmin / step)
    last_step_fctr = math.floor(vmax / step)
    ticks = [round(i * step, 12) for i in range(first_step_fctr, last_step_fctr + 1)]
    return ticks or [float(vmin), float(vmax)]
