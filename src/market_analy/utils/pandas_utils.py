"""Utilities for operating on pandas DataFrame and Series."""

from __future__ import annotations

from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np


def rebase_to_cell(
    data: pd.DataFrame | pd.Series,
    row: Any,
    col: str,
    value: int = 100,
) -> pd.DataFrame:
    """Rebase a pandas object to a given base value for a given cell.

    Parameters
    ----------
    data
        pandas object to rebase.

    row
        Label of row containing cell to have base value.

    col
        Label of column containing cell to have base value.

    value
        Base value. Cell at `row`, `column` will have this value with all
        other cells rebased relative to this base value.

    Example
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> rebase_to_cell(df, 3, 'close_', 100)
       open_  close_
    0    0.0    62.5
    1   12.5    75.0
    2   25.0    87.5
    3   37.5   100.0
    4   50.0   112.5
    """
    return data / data.loc[row, col] * value


def rebase_to_row(
    data: pd.DataFrame | pd.Series, row: int = 0, value: int = 100
) -> pd.DataFrame:
    """Rebase a pandas object to a given row.

    If `data` is a pd.DataFrame, each column will be rebased independently.

    Parameters
    ----------
    data
        pandas object to rebase.

    row
        index of row against which to rebase data.

    value
        Base value for each cell of `row`. All other rows will be rebased
        relative to this base value.

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> rebase_to_row(df, 2, 100)
       open_      close_
    0    0.0   71.428571
    1   50.0   85.714286
    2  100.0  100.000000
    3  150.0  114.285714
    4  200.0  128.571429
    """
    return data / data.iloc[row] * value


def tolists(df: pd.DataFrame) -> list[list[Any]]:
    """Convert pd.DataFrame to list of lists.

    Each each inner list represents one column of the DataFrame.

    Parameters
    ----------
    df
        DataFrame to convert to list of lists.

    Examples
    --------
    >>> df = pd.DataFrame(dict(open_=range(5), close_=range(5, 10)))
    >>> df
       open_  close_
    0      0       5
    1      1       6
    2      2       7
    3      3       8
    4      4       9
    >>> tolists(df)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    return [col.values.tolist() for name, col in df.items()]


def interval_of_intervals(
    intervals: pd.IntervalIndex,
    closed: Literal["left", "right", "both", "neither"] = "right",
) -> pd.Interval:
    """Return interval covered by a monotonic IntervalIndex.

    Parameters
    ----------
    intervals
        Monotonic pd.IntervalIndex for which require the encompassing
        interval.

    closed
        Side on which to close returned pd.Interval.

    Raises
    ------
    ValueError
        If `intervals` is not monotonic.

    Examples
    --------
    >>> # ignore first part, for testing purposes only...
    >>> import pytest, pandas
    >>> v = pandas.__version__
    >>> if (
    ...     (v.count(".") == 1 and float(v) < 2.2)
    ...     or (
    ...         v.count(".") > 1
    ...         and float(v[:v.index(".", v.index(".") + 1)]) < 2.2
    ...     )
    ... ):
    ...     pytest.skip("printed return only valid from pandas 2.2")
    >>> #
    >>> # example from here...
    >>> left = pd.date_range('2021-05-01 12:00', periods=5, freq='h')
    >>> right = left + pd.Timedelta(30, 'min')
    >>> index = pd.IntervalIndex.from_arrays(left, right)
    >>> index.to_series(index=range(5))
    0    (2021-05-01 12:00:00, 2021-05-01 12:30:00]
    1    (2021-05-01 13:00:00, 2021-05-01 13:30:00]
    2    (2021-05-01 14:00:00, 2021-05-01 14:30:00]
    3    (2021-05-01 15:00:00, 2021-05-01 15:30:00]
    4    (2021-05-01 16:00:00, 2021-05-01 16:30:00]
    dtype: interval
    >>> interval_of_intervals(index)
    Interval(2021-05-01 12:00:00, 2021-05-01 16:30:00, closed='right')
    """
    # NOTE Can lose doctest skip when pandas min support is >= 2.2
    if not intervals.is_monotonic_increasing:
        raise ValueError(f"`intervals` must be monotonic. Received as '{intervals}'.")
    return pd.Interval(intervals[0].left, intervals[-1].right, closed=closed)


def interval_contains(interval: pd.Interval, intervals: pd.IntervalIndex) -> np.ndarray:
    """Boolean array indicating intervals contained within an interval.

    Parameters
    ----------
    interval: pd.Interval
        Intervals within which to check if intervals are contains.

    intervals: pd.IntervalIndex
        Intervals to check if contained in interval.
    """
    if interval.closed == "left":
        left_cond = intervals.left >= interval.left
        right_cond = intervals.right < interval.right
    elif interval.closed == "right":
        left_cond = intervals.left > interval.left
        right_cond = intervals.right <= interval.right
    elif interval.closed == "both":
        left_cond = intervals.left >= interval.left
        right_cond = intervals.right <= interval.right
    else:
        left_cond = intervals.left > interval.left
        right_cond = intervals.right < interval.right
    return left_cond & right_cond


def intervals_subset(
    intervals: pd.IntervalIndex, interval: pd.Interval, overlap: bool = False
) -> pd.IntervalIndex:
    """Subset of intervals overlapping with or contained by an interval.

    Parameters
    ----------
    intervals
        Intervals from which to take subset.

    interval
        Interval of intervals to be taken as a subset.

    overlap
        True to include all intervals that overlap with interval, False to
        only return intervals that are fully contained within interval.
    """
    if overlap:
        bv = intervals.overlaps(interval)
    else:
        bv = interval_contains(interval, intervals)
    return intervals[bv]


def interval_index_new_tz(
    index: pd.IntervalIndex, tz: ZoneInfo | str | None
) -> pd.IntervalIndex:
    """Return pd.IntervalIndex with different timezone.

    Note: `index` is not changed in place.

    Parameters
    ----------
    index
        pd.IntervalIndex on which to base return. Must have left and right
        sides as pd.DatetimeIndex. If these pd.DatetimeIndex are tz-naive
        then indices will be localised to `tz`, if tz_aware then indices
        will be converted to `tz` if `tz` is not None, or localized to
        None if `tz` is None.

    tz
        Timezone for returned pd.IntervalIndex. Examples: "US/Eastern",
        "Europe/Paris", "UTC".

        Pass as None to return as timezone naive.

    Returns
    -------
    pd.IntervalIndex
        pd.IntervalIndex as `index` albeit with timezone as `tz`.

    Examples
    --------
    >>> tz = ZoneInfo("US/Central")
    >>> left = pd.date_range(
    ...     '2021-05-01 12:00', periods=5, freq='h', tz=tz
    ... )
    >>> right = left + pd.Timedelta(30, 'min')
    >>> index = pd.IntervalIndex.from_arrays(left, right)
    >>> index.right.tz
    zoneinfo.ZoneInfo(key='US/Central')
    >>> UTC = ZoneInfo("UTC")
    >>> new_index = interval_index_new_tz(index, tz=UTC)
    >>> new_index.left.tz == new_index.right.tz == UTC
    True
    >>> tz_naive = interval_index_new_tz(index, tz=None)
    >>> tz_naive[0].left
    Timestamp('2021-05-01 12:00:00')
    """
    indices = []
    for indx in [index.left, index.right]:
        try:
            indices.append(indx.tz_localize(tz))
        except TypeError:
            indices.append(indx.tz_convert(tz))
    return pd.IntervalIndex.from_arrays(indices[0], indices[1], closed=index.closed)


def index_dates_to_str(index: pd.DatetimeIndex) -> pd.Index:
    """Convert index representing dates to an index of dtype 'string'.

    Formats dates as %Y-%m-%d.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.date_range("2020-01-09", "2020-01-10", freq="D")
    >>> str_index = index_dates_to_str(index)
    >>> str_index
    Index(['2020-01-09', '2020-01-10'], dtype='string')
    """
    return index.strftime("%Y-%m-%d").astype("string")
