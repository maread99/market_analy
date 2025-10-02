"""Standalone analysis functions."""

import datetime

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
        raise TypeError("`data.columns` must be a `pd.MultiIndex`")
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


def get_ath(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Return all-time high as at each timestamp.

    Returns ATH as at end of period represented by corresponding
    timestamp.

    NB: ATH assumed as high since start of `data`.

    Parameters
    ----------
    data
        If pd.Series then values as high during the period represented
        by the corresponding indice.

        DataFrame must include "high" column(s) with values as
        described for a pd.Series. If DataFrame covers multiple
        symbols then columns should be indexed with a `pd.MultiIndex`
        with level 0 as symbol.
    """
    if isinstance(data, pd.Series):
        return data.expanding().max()
    df = data[[c for c in data.columns if "high" in c]]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df.expanding().max()


@valimp.parse
def get_period_high(
    data: pd.Series | pd.DataFrame,
    window: int | str | datetime.timedelta,
    include_current: bool = True,
) -> pd.Series | pd.DataFrame:
    """Return 52wk high as at each timestamp.

    NB: For sessions within the first 52 weeks of data, the high will
    represent the high of prices since the start of the data.

    Parameters
    ----------
    data
        If pd.Series then values as high during the period represented
        by the corresponding indice.

        If pd.DataFrame then must include "high" column(s) with values
        as described for a pd.Series. If DataFrame covers multiple
        symbols then columns should be indexed with a `pd.MultiIndex`
        with level 0 as symbol.

    window
        Value to be passed to 'window` parameter of
        `pd.DataFrame.rolling` (or `pd.Series.rolling`) to determine
        period over which highs shoudl be evaluated. Can be passed as:
            `str` of a pandas frequency representing period over
            which to evaluate highs, for example "365D" for '52wk'
            highs.

            `int` to define window as a fixed number of observations,
            for example every 3 values.

            `datetime.timedelta` to define window as a fixed period.

    include_current : bool
        Determines if the evaluation of the high should include the
        timestamp that represents the end of the period.
    """
    closed = "right" if include_current else "left"
    if isinstance(data, pd.Series):
        rolling = data.rolling(
            window, min_periods=1, center=False, win_type=None, closed=closed
        )
        return rolling.max()

    df = data[[c for c in data.columns if "high" in c]]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    rolling = df.rolling(
        window, min_periods=1, center=False, win_type=None, closed=closed
    )
    return rolling.max()


@valimp.parse
def get_pct_off_high(
    data: pd.DataFrame,
    window: int | str | datetime.timedelta | None,
    include_current: bool = True,
) -> pd.DataFrame:
    """Return percent off high as at each timestamp.

    NB: For timestamps within the first period the return will be
    the percentage off the high as of that timestamp (i.e. the high
    will have been evaluated over shorter period than `pdfreq`)

    Parameters
    ----------
    data
        pd.DataFrame with "high" column(s) with values as the high of
        the period represented by the corresponding indice and "close"
        column(s) representing the price as at the end of th
        corresponding indice. If DataFrame covers multiple symbols then
        columns should be indexed with a `pd.MultiIndex` with level 0 as
        symbol.

    window
        Value to be passed to 'window` parameter of
        `pd.DataFrame.rolling` to determine period over which
        highs shoudl be evaluated, or None to evaluate all-time high.
        Can be passed as:
            `str` of a pandas frequency representing period over
            which to evaluate highs, for example "365D" for '52wk'
            highs.

            `int` to define window as a fixed number of observations,
            for example every 3 values.

            `datetime.timedelta` to define window as a fixed period.

            None to eveluate high as high since first observation
            (all-time high).

    include_current : bool
        Determines if the evaluation of the high should include the
        timestamp that represents the end of the period.
        NB this options has no effect if evaluating high as all-time
        high (i.e. if `window` is None), in which case high will
        always be evaluated to include the current timestamp.
    """
    if window is None:
        high = get_ath(data)
        label = "ath"
    else:
        high = get_period_high(data, window, include_current)
        label = f"{window}_high"

    close = data[[c for c in data.columns if "close" in c]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.droplevel(1)
    else:
        assert len(high.columns) == 1
        cols_index = pd.Index([label])
        high.columns = cols_index
        assert len(close.columns) == 1
        close.columns = cols_index
    return -100 * ((high - close) / high)
