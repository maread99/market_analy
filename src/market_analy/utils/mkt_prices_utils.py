"""Utility funcions for market_prices library."""

from __future__ import annotations

import pandas as pd
import market_prices as mp


def request_daily_prices(**kwargs) -> bool:
    """Query if daily prices could be requested for given period parameters."""
    if "start" in kwargs and not mp.helpers.is_date(pd.Timestamp(kwargs["start"])):
        return False
    if "end" in kwargs and not mp.helpers.is_date(pd.Timestamp(kwargs["end"])):
        return False
    return all((kw not in kwargs for kw in ["minutes", "hours"]))


def period_string(**kwargs) -> str:
    """Return string description of period defined from period parameters.

    Period parameters as defined by `mp.prices.base.Base.get`

    If no parameters passed returns 'all'.

    NB string will assume 'start' and 'end' as the values passed. Even if
    these are the same as values passed to 'start' and 'end' parameters of
    prices.get() the values in the returned string may not necessarily
    align with the start and end dates of the actual returned data.

    NB excess kwargs pass silently.
    """
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    minutes = kwargs.get("minutes", 0)
    hours = kwargs.get("hours", 0)
    days = kwargs.get("days", 0)
    weeks = kwargs.get("weeks", 0)
    months = kwargs.get("months", 0)
    years = kwargs.get("years", 0)

    is_period = bool(sum([minutes, hours, days, weeks, months, years]))

    if end is None and start is None and not is_period:
        return "all"

    fts = mp.helpers.fts
    end_str = fts(pd.Timestamp(end)).strip() if end is not None else None
    start_str = fts(pd.Timestamp(start)).strip() if start is not None else None

    if end_str is not None and start_str is not None:
        return f"{start_str} to {end_str}"

    if start_str is not None and end_str is None and not is_period:
        return f"since {start_str}"

    if sum([minutes, hours]) > 0:
        duration = ""
        if hours:
            duration += f"{hours}H"
        if minutes:
            if duration:
                duration += " "
            duration += f"{minutes}T"
    elif days > 0:
        duration = f"{days}D"
    else:
        mapping = {"Y": years, "M": months, "W": weeks}
        duration = " ".join([f"{v}{s}" for s, v in mapping.items() if v > 0])

    if end_str is None and start_str is None:
        return duration
    elif end_str is not None:
        return f"{duration} to {end_str}".strip()
    else:
        return f"{duration} from {start_str}"


def range_string(
    index: pd.DatetimeIndex | pd.IntervalIndex, close: bool = False, shand: bool = False
) -> str:
    """Return string describing range of dates covered by an index.

    Minutes and Hours ommited if timestamp represents a date.

    Parameters
    ----------
    index
        Index describing dates to be evaluated as a range.

    close
        True: evaluate range as:
            if `index` is `pd.IntervalIndex`, from close of first indice
            through close of last indice.

            if `index` is `pd.DatetimeIndex`, from second indice through
            last indice.

        False: evaluate range as:
            if `index` is `pd.IntervalIndex`, from open of first indice
            through close of last indice.

            if `index` is `pd.DatetimeIndex`, from first indice through
            last indice.

    shand
        True to return shorthand string. Will ommit components of end
        timestamp that are the same as the corresponding component of the
        start timestamp.
    """
    if isinstance(index, pd.DatetimeIndex):
        start_ = index[1] if close else index[0]
        start_str, _, end_str = period_string(start=start_, end=index[-1]).split()
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
    else:
        start = index[0].left if not close else index[0].right
        start_fmt_str = "%Y-%m-%d" if mp.helpers.is_date(start) else "%Y-%m-%d %H:%M"
        start_str = start.strftime(start_fmt_str)
        end = index[-1].right

    fmt_str = "%Y-%m-%d" if mp.helpers.is_date(end) else "%Y-%m-%d %H:%M"

    if shand:
        to_replace = ["%Y-", "%m-", "%d", "%H:"]
        for i, attr in enumerate(["year", "month", "day", "hour"]):
            if getattr(start, attr) == getattr(end, attr):
                fmt_str = fmt_str.replace(to_replace[i], "")
            else:
                break
        fmt_str = fmt_str.strip()

    return start_str + " to " + end.strftime(fmt_str)
