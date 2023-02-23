"""Formatter functions and mapping."""

import pandas as pd


def formatter_datetime(x: pd.Timestamp) -> str:
    if x == x.normalize():
        return x.strftime("%Y-%m-%d")
    else:
        return x.strftime("%Y-%m-%d %H:%M")


def formatter_percent(x: float) -> str:
    if -0.1 < x < 0.1:
        p = 2
    elif x <= -1 or x >= 1:
        p = 0
    else:
        p = 1
    return "{:.{p}%}".format(x, p=p)


def formatter_float(x: float) -> str:
    return "{:.{p}f}".format(x, p=2)


FORMATTERS = {
    "pct_chg": formatter_percent,
    "chg": formatter_float,
    "open": formatter_float,
    "high": formatter_float,
    "low": formatter_float,
    "adjclose": formatter_float,
    "close": formatter_float,
    "volume": lambda x: int(x),
    "date": formatter_datetime,
    "start": formatter_datetime,
    "end": formatter_datetime,
    "days": lambda x: int(x),
    "hours": lambda x: int(x),
    "minutes": lambda x: int(x),
}
