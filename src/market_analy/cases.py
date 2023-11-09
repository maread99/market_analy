"""Base classes and protocols for displaying analyses' results over charts.

`CaseBase` offers a base class to represent a single outcome, event or
case of an analysis, for example a 'position' taken under a criteria or an
evaluated 'trend movement' that manifested.

'CaseSupportsChartAnaly` defines a base protocol that should be fulfilled
by any case class that is to be displayed on a chart.

`CasesBase` offers a base class to represent all outcomes, events or cases
of a specific analysis over a fixed continuous period. For example, all the
'positions' that would have been  taken under a criteria or all the
evaluated 'trend movements' that manifested.

`CasesSupportsChartAnaly` defines a base protocol that should be fulfilled
by any cases class that is to be displayed on a chart.

`ChartSupportsCasesGui` defines protocol that should be fulfilled by any
`charts` class that is be used by a gui that provides for selecting and
navigating between cases.

Notes
-----

HANDLING CHART EVENTS
Nov 23. The handlers for chart events are defined on the `cases` subclass.
The charts subclass's 'cases' argument is typed with
`CasesSupportsChartAnaly` and this in turn defines the handler methods that
the charts class expects the cases instance to have. Alternatively, it
would be possible to define the handlers on the charts class and for the
`CasesSupportsChartAnaly` protocol to instead define all the underlying
attributes that the charts class would need the caese class to have in
order to define the handlers itself.

It's considered beneficial to define the handlers on the `cases` subclass,
as opposed to the `charts` subclass, to best accommodate analyses that have
various versions, for example trends and trends_alt. The supposition is
that different versions are more likely to require different subclasses of
the `cases` class than the `charts` class. If this is the case then the
creation of a variation will be just a matter of writting a new `cases`
class, not both a new `cases` subclass for the version and a new `charts`
subclass to define any differences in handling events.

Time will tell if this supposition holds. For now handlers are defined on
the `cases` class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import typing

import bqplot as bq
import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from market_analy.charts import OHLCTrends


class CaseSupportsChartAnaly(typing.Protocol):
    """Base protocol for `case` classes to be charted.

    A case class
    """

    @property
    def _start(self) -> pd.Timestamp:
        """Bar when case considered to start."""
        ...

    @property
    def _end(self) -> pd.Timestamp | None:
        """Bar when case considered to have concluded.

        None if case had not concluded as at end of available data.
        """
        ...


class CasesSupportsChartAnaly(typing.Protocol):
    """Base protocol for `cases` classes to be charted.

    Attributes
    ----------
    cases
        Ordered list of all cases over a fixed continuous period of
        analysis.
    """

    cases: Sequence[CaseSupportsChartAnaly]
    data: pd.DataFrame

    def event_to_case(self, mark: bq.Scatter, event: dict) -> CaseSupportsChartAnaly:
        """Get case corresonding to an event for mark representing a case.

        Parameters as those passed to event handler on clicking a point of
        the scatter.
        """
        ...

    def get_index(self, case: CaseSupportsChartAnaly) -> int:
        """Get index position of a case"""
        ...

    def handler_click_case(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking on mark representing a case.

        Parameters as those passed to event handler on clicking a point of
        the scatter.
        """
        ...


class ChartSupportsCasesGui(typing.Protocol):
    """Protocol for `charts` classes that support guis supporting cases.

    `charts` classes that conform with this protocol can be used by guis
    that provide for selecting and navigating between cases.
    """

    @property
    def current_case(self) -> CaseSupportsChartAnaly | None:
        ...

    def hide_cases(self):
        ...

    def show_cases(self):
        ...

    def reset_marks(self):
        ...

    def select_next_case(self):
        ...

    def select_previous_case(self):
        ...


class CaseBase(ABC):
    """Base for classes defining a case.

    Defines dunder methods allowing for comparison of attributes that have
    pd.Series and pd.DataFrame values.
    """

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return False
        for name in self.__dataclass_fields__.keys():
            v, v_other = getattr(self, name), getattr(other, name)
            if isinstance(v, pd.DataFrame):
                try:
                    pd.testing.assert_frame_equal(v, v_other)
                except AssertionError:
                    return False
            elif isinstance(v, pd.Series):
                try:
                    pd.testing.assert_series_equal(v, v_other)
                except AssertionError:
                    return False
            elif isinstance(v, np.ndarray):
                try:
                    assert (v == v_other).all()
                except AssertionError:
                    return False
            elif isinstance(v, dict):
                for k, value in v.items():
                    value_other = v_other[k]
                    if isinstance(value, np.ndarray):
                        try:
                            assert (value == value_other).all()
                        except AssertionError:
                            return False
                    elif value != value_other:
                        return False
            elif v != v_other:
                return False
        return True

    def _raise_if_diff_type(self, other: typing.Any):
        if isinstance(other, type(self)):
            return
        raise NotImplementedError(
            f"Instances of type<{type(self)}> can only be compared with other"
            f" instances of the same type, not <{type(other)}>."
        )

    def __lt__(self, other):
        self._raise_if_diff_type(other)
        return self._start < other._start

    def __le__(self, other):
        self._raise_if_diff_type(other)
        return self._start <= other._start

    def __gt__(self, other):
        self._raise_if_diff_type(other)
        return self._start > other._start

    def __ge__(self, other):
        self._raise_if_diff_type(other)
        return self._start >= other._start

    @property
    @abstractmethod
    def _start(self) -> pd.Timestamp:
        """Bar when case considered to start."""
        pass


@dataclass(frozen=True)
class CasesBase(ABC, CasesSupportsChartAnaly):
    """All cases over an analysis period.

    Attributes
    ----------
    cases
        Ordered list of all cases identifed over analysis period defined
        by `data`.

    data
        OHLC data in which `cases` identified.
    """

    cases: Sequence[CaseSupportsChartAnaly]
    data: pd.DataFrame

    def __iter__(self) -> typing.Iterator[CaseSupportsChartAnaly]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(
        self, key: int | slice
    ) -> CaseSupportsChartAnaly | Sequence[CaseSupportsChartAnaly]:
        return self.cases[key]

    def event_to_case(self, mark: bq.Scatter, event: dict):
        """Get case corresonding to an event for mark representing a case.

        Parameters as those passed to event handler on clicking a point of
        the scatter.
        """
        i = event["data"]["index"]
        return self[i]

    def get_index(self, case: CaseSupportsChartAnaly) -> int:
        """Get index position of a case."""
        return self.cases.index(case)

    @abstractmethod
    def handler_click_case(self, chart: OHLCTrends, mark: bq.Scatter, event: dict):
        """Handler for clicking on mark representing a case.

        Parameters as those passed to event handler on clicking a point of
        the scatter.
        """
        pass
