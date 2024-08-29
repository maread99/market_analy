"""Standalone analysis functions."""

import pandas as pd
import valimp


@valimp.parse
def net_of(data: pd.DataFrame, rebase: bool = True) -> pd.DataFrame:
    """Return OHLC data for one symbol net of changes of another.

    For example, return OHLC data for an equity that discounts the effect
    of the equity's wider market.

    Parameters
    ----------
    data
        `pd.DataFrame` with columns as `pd.MultiIndex` with:
            level 0 representing two symbols. The return will reflect
            the values of the first symbol net of those of the second
            symbol.

            level 1 as 'open', 'high', 'low', 'close' and, optionally,
            'volume'.

    rebase
        True (default) to rebase net prices with first close value as
        100.

        False to not rebase net prices.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         ("A", "open"): [18, 90, 50, 52, 78],
    ...         ("A", "high"): [25, 120, 51, 52, 208],
    ...         ("A", "low"): [15, 80, 49, 52, 52],
    ...         ("A", "close"): [20, 100, 50, 52, 104],
    ...         ("A", "volume"): [1, 2, 3, 4, 5],
    ... # NB open, high, low and volume of 'other' have no effect
    ...         ("B", "open"): [222, 333, 444, 555, 666],
    ...         ("B", "high"): [444, 555, 666, 777, 888],
    ...         ("B", "low"): [111, 222, 333, 444, 555],
    ...         ("B", "close"): [10, 40, 30, 28.5, 57],
    ...         ("B", "volume"): [1111, 2222, 3333, 4444, 5555],
    ...     }
    ... )
    ... # 0, As A, no prior close
    ... # 1, A rises 500%, B rises 400%, so net is 100% rise
    ... # 2, A drops 50%, B drops 25%, so net is a 25% drop
    ... # 3, A rises 4%, B falls 5%, so net is a 9% rise
    ... # 4, A rises 100%, B rises 100%, so net is unchanged
    >>> net_of(df, rebase=False).round(3)
         open  high    low  close  volume
    0  18.000  25.0  15.00   20.0       1
    1  36.000  48.0  32.00   40.0       2
    2  30.000  30.6  29.40   30.0       3
    3  32.700  32.7  32.70   32.7       4
    4  24.525  65.4  16.35   32.7       5
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("`data.columns` must be a `pd.MultiIndex`")
    if len(data.columns.levels) != 2:
        raise ValueError("`data.columns` must have two levels")
    if len(data.columns.levels[0]) != 2:
        raise ValueError("level 0 of `data.columns` must represent two symbols")

    cols = [col for col in data.columns if col[1] == "close"]

    chgs = []
    for col in cols:
        shifted = data[col].shift(1)
        chgs.append((data[col] - shifted) / shifted)

    # evaluate net percentage chg
    chg_net = chgs[0] - chgs[1]

    prev_close = 100 if rebase else data[cols[0]].iloc[0]
    closes_ = [prev_close]
    for chg in chg_net[1:]:
        close = prev_close * (1 + chg)
        closes_.append(close)
        prev_close = close

    closes = pd.Series(closes_, index=chg_net.index)
    df = data[data.columns.levels[0][0]]
    mapping = {
        "close": closes,
        "open": (df.open / df.close) * closes,
        "high": (df.high / df.close) * closes,
        "low": (df.low / df.close) * closes,
        "volume": df.volume if "volume" in df.columns else None,
    }

    return pd.DataFrame({col: mapping[col] for col in df.columns})
