"""bqplot price/date charts for financial instruments.

Classes
-------
Base(metaclass=ABCMeta):
    ABC defining implementation and base functionality.

BaseSubsetDD(Base):
    Base to chart discontinuous dates or selectable subset of.

BasePrice(BaseSubsetDD):
    Base class for price charts.

Line(BasePrice):
    Line Chart

MultLine(BasePrice):
    Line Chart for multiple instruments, prices rebased for comparison.

OHLC(BasePrice):
    OHLC Chart.

PctChgBar(_PctChgBarBase):
    Bar Chart displaying precentage changes.

PctChgBarMult(_PctChgBarBase):
    Bar Chart displaying precentage changes of multiple instruments.
"""

import enum
import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from functools import lru_cache, partialmethod
from typing import TYPE_CHECKING, Any, Literal

import bqplot as bq
import IPython
import ipywidgets as w
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import market_analy.utils.bq_utils as ubq
import market_analy.utils.pandas_utils as upd
from market_analy.formatters import FORMATTERS, formatter_datetime
from market_analy.utils.dict_utils import set_kwargs_from_dflt

from .cases import (
    CasesSupportsChartAnaly,
    CaseSupportsChartAnaly,
    ChartSupportsCasesGui,
)

# ruff: noqa: N802  # invalid-function-name.  Allow Camelcase properties that return types

COLOR_CHART_TEXT = "lightyellow"
STYLE_CHART_TITLE = {"font-size": "20px", "fill": COLOR_CHART_TEXT}

STYLE_TOOLTIP = {
    "opacity": 0.8,
    "background-color": "black",
    "border-color": "white",
    "border-width": "1px",
}


# pre-defined groups against which to add extra marks
class Groups(enum.Enum):
    CASE = enum.auto()  # tempoary marks representing the current selected case.
    PERSIST = enum.auto()  # marks that should always be visible.
    # scatter marks where each point of each scatter represents a specific case.
    CASES_SCATTERS = enum.auto()
    # marks where each mark represents a specific case. All cases represented.
    CASES_OTHER_0 = enum.auto()
    CASES_OTHER_1 = enum.auto()
    CASES_OTHER_2 = enum.auto()


CASES_OTHER = {
    Groups.CASES_OTHER_0,
    Groups.CASES_OTHER_1,
    Groups.CASES_OTHER_2,
}

# Groups representing multiple cases
CASES_GROUPS = {Groups.CASES_SCATTERS} | CASES_OTHER

# type aliases
AddedMarkKeys = (
    str
    | Literal[
        Groups.CASE,
        Groups.PERSIST,
        Groups.CASES_SCATTERS,
        Groups.CASES_OTHER_0,
        Groups.CASES_OTHER_1,
        Groups.CASES_OTHER_2,
    ]
)
AddedMarks = dict[AddedMarkKeys, list[bq.Mark]]
AxesKwargs = dict[ubq.ScaleKeys, dict[str, Any]]


def tooltip_html_style(**kwargs) -> str | None:
    """HTML to define inline style for a tooltip.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of style attributes, for example:
            color='blue'
        NB where style attribute has a hyphen, replace with underscore,
        for example pass line-height as:
            line_height = 3
    """
    if not kwargs:
        return None
    s = 'style="'
    for k, v in kwargs.items():
        k = k.replace("_", "-")  # noqa: PLW2901
        s += f"{k}: {v}; "
    return s + '"'


def hold_mark_update(func) -> Callable:
    """Hold off syncing mark in frontend until `func` executed.

    Employ as a decorator to hold off syncing mark in frontend until
    decorated function has fully executed.
    """

    def wrapper(self, *args, **kwargs):
        with self.mark.hold_sync():
            func(self, *args, **kwargs)

    return wrapper


class Base(metaclass=ABCMeta):
    """ABC for creating a bqplot Chart.

    ABC defines:
        Implementation to create Chart
        Implemntation to update axes presentation
        Functionality via Mixin methods

    Subclass implementation
    -----------------------
    Subclasses must implement the following abstract properties and methods:

    **Abstact Properties**

        _x_data:
            Data represented by x-axis. Subclass can optionally expose
            via more concrete property.

        _y_data: DataFrame | Series
            Data represented by y-axis. Subclass can optionally expose
            via more concrete property.

        MarkCls
            bqplot Mark class, for example bq.Lines.


    **Abstact Methods**

        _create_scales() -> dict[str, bq.Scale]:
            Subclass should override.

        _axes_kwargs() -> dict[ubq.ScaleKeys, dict[str, Any]]:
            Each subclass should extend to add default kwargs applicable to
            the subclass' level.

        _get_mark_y_data():
            Subclass should implement to return y data for Mark.

        _create_mark() -> bq.Mark:
            Subclass should extend by way of passing through kwargs.

        _create_figure() -> bq.Figure:
            Subclass should extend by way of passing through kwargs.

        update_y_axis_presentation()
            Subclass must implemement, if only to execute as defined on
            ABC.

        update_x_axis_presentation()
            Subclass must implemement, if only to execute as defined on
            ABC.

    Subclasses can optionally extend the following methods:

        __init__()
            Subclass can optionally add code before or after executing
            constructor as defined on this ABC.

        _update_data()
            Subclass will need to extend if mark has more than x and y
            data attributes.

        _tooltip_value()
            Subclass can implement to raise a tooltip on hovering over
            mark. Tooltip contents defined by returned HTML string.

        update_presentation()
            Subclass will need to extend if mark has an axis, other than
            'x' and 'y', that requires presentation updates.

        _add_marks(self):
            Subclass can override to add extra marks at initalisation.
            Method should use `add_marks` to add extra marks.

    Subclasses can optionally overwrite or extend the following methods if
    require a plot against a secondary y-axis and the default implemention
    is not otherwise sufficient:

        MarkY2Cls()
            Subclass should overwrite if requires a secondary mark of a
            class other than bq.Lines.

        _get_mark_y2_data()

        _create_y2_mark()

    Subclasses should NOT override or extend the following methods:

        _create_chart()

    It is not intended that subclasses override or extend the following
    methods:

        _create_axes()

    Helper Methods
    --------------
    The following helper methods are made available to subclasses.

        _add_axes_kwargs()
            Helper to facilitate subclasses building up axes_kwargs. See
            _axes_kwargs.__doc__.

        _tooltip_style():
            HTML to define inline style for a tooltip.

    Mixin Properties
    ----------------
    The following public Mixin properties are made available to subclasses.

    title: str (settable property)
        Chart title

    added_marks:
        Extra marks that have been added to the chart, by group.

    added_marks_groups
        Names of groups of added marks.

    Mixin Methods
    -------------
    The following public Mixin Methods are made available to subclasses.

    update():
        Update existing mark and figure.

    close():
        Close all chart widgets.

    delete():
        Delete all chart widgets.

    display():
        Display chart.

    add_marks():
        Add extra marks to the chart.

    remove_added_marks():
        Remove a group of extra marks

    remove_added_marks_all():
        Remove extra marks in all groups.

    hide_y2():
        Hide data_y2 mark and y2 axis.

    show_y2():
        Show data_y2 mark and y2 axis.

    The following private Mixin Methods are made available to subclasses to
    expose if appropriate.

    _cycle_ledend().
        Move any legend to next position.

    _y_max():
        Maximum value of --_y_data--.

    _y_min():
        Minimum value of --_y_data--.

    _y_range():
        Range of --_y_data--.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None = None,
        display: bool = False,
        data_y2: list[float | int | list[float | int]] | None = None,
        **_,
    ):
        """Create chart.

        Parameters
        ----------
        data
            Chart Data.

        title
            Chart title.

        display
            True to display chart.

        data_y2
            Values to be plotted against a secondary y axis.

        Notes
        -----
        Subclass must NOT override constructor although can extend by
        adding code before or after execution of constuctor as defined on
        this ABC.
        """
        self.data = data
        if data_y2 is None:
            self._data_y2 = None
        else:
            self.data_y2 = data_y2

        # Attributes set by --_create_chart--
        self._widgets: list[w.Widget]
        self.scales: dict[str, bq.Scale]
        self.axes: list[bq.BaseAxis]
        self.mark: bq.Mark
        self.figure: bq.Figure

        # set by add_marks and remove_marks
        self._added_marks: AddedMarks = defaultdict(list)

        self._create_chart()
        self._add_marks()
        self.title = title
        self.update_presentation()

        if display:
            self.display()

    # DATA
    @property
    def data(self) -> pd.DataFrame | pd.Series:
        """Chart data."""
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame | pd.Series):
        data = data.copy()
        if isinstance(data.index, pd.IntervalIndex):
            data.index = upd.interval_index_new_tz(data.index, None)
        elif data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        self._data = data

    @property
    def data_y2(self) -> list[float | int] | None:
        """Secondary y-axis values."""
        return self._data_y2

    @data_y2.setter
    def data_y2(self, data: list[float | int | list[float | int]]):
        if isinstance(data[0], list):
            for lst in data:
                assert len(lst) == len(self.data)
        else:
            assert len(data) == len(self.data)
        self._data_y2 = data

    @property
    def _has_y2_plot(self) -> bool:
        """Query if chart has data to plot against the y2 axes."""
        return self.data_y2 is not None

    @property
    def _no_y2_plot(self) -> bool:
        """Query if chart does not have a plot against the y2 axes."""
        return self.data_y2 is None

    @property
    @abstractmethod
    def _x_data(self):
        """Data represented by x_axis.

        Notes
        -----
        Subclass can additionally choose to expose implementation through a
        concrete public method, for example 'dates'.
        """

    @property
    @abstractmethod
    def _y_data(self) -> pd.DataFrame | pd.Series:
        """Data represented by y_axis.

        Notes
        -----
        Subclass can additionally choose to expose implementation through a
        concrete public method, for example 'prices'.
        """

    @property
    def _y_max(self):
        """Maximum value of --_y_data--.

        Notes
        -----
        Subclass can optionally expose via concrete property, for example
        'max_price'.
        """
        return self._y_data.values.max()

    @property
    def _y_min(self):
        """Minimum value of --_y_data--.

        Notes
        -----
        Subclass can optionally expose via concrete property, for example
        'min_price'.
        """
        return self._y_data.values.min()

    @property
    def _y_range(self):
        """Range of --_y_data--.

        Notes
        -----
        Subclass can optionally expose via concrete property, for example
        'price_range'.
        """
        return self._y_max - self._y_min

    # CHART CREATION
    def _create_chart(self):
        """Create chart.

        Notes
        -----
        Subclass should NOT override or extend this method.
        """
        self._widgets = []
        w.Widget.on_widget_constructed(lambda w: self._widgets.append(w))
        self.scales = self._create_scales()
        self.axes = self._create_axes()
        self.mark = self._create_mark()
        self.mark_y2 = None if self._no_y2_plot else self._create_y2_mark()
        self.figure = self._create_figure()
        w.Widget.on_widget_constructed(None)

    @abstractmethod
    def _create_scales(self) -> dict[ubq.ScaleKeys, bq.Scale]:
        """Create scales.

        Notes
        -----
        Subclass should return Dictionary of all scale objects
        that will be passed to Mark and Axes.
        """

    @staticmethod
    def _add_axes_kwargs(
        dflt_axes_kwargs: AxesKwargs, axes_kwargs: AxesKwargs | None = None
    ):
        """Add default kwargs to axes_kwargs. See _axes_kwargs.__doc__."""
        axes_kwargs = axes_kwargs if axes_kwargs is not None else {}

        for axis_name, dflt_axis_kwargs in dflt_axes_kwargs.items():
            axis_kwargs = axes_kwargs.get(axis_name, {})
            axis_kwargs = set_kwargs_from_dflt(axis_kwargs, dflt_axis_kwargs, deep=True)
            axes_kwargs[axis_name] = axis_kwargs

        return axes_kwargs

    @abstractmethod
    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        """Keyword arguments to create an bq.Axis for each bq.Scale.

        Paramaters
        ----------
        axes_kwargs
            Kwargs for creating Axis, defined by axis in the same way as
            the return.

        **general_kwargs:
            Kwargs to be passed to Axis constructor for all created Axis.

        Returns
        -------
        dict[ubq.ScaleKeys, dict[str, Any]] :
            Each dictionary item represents an Axis to be created.
                Key - Corresponding scale key (from 'x', 'y', 'color',
                'rotation', 'skew', 'opacity', 'size').

                Value - Dictionary of kwargs to be passed to Axis
                constructor.

        Implementation
        --------------
        The last subclass of the class hierarchy should define axes_kwargs
        and general kwargs, as applicable to that hierarchal level, and
        pass these on to its parent class via super()._axes_kwargs. Each
        subclass in turn should augment axes_kwargs and general kwargs, as
        applicable to their hierarchal level, and pass these arguments on
        to their parent via super()._axes_kwargs. In this way each subclass
        can add default kwargs applicable to the subcalss level,
        culminating with the method as defined on this ABC, where the
        corresponding scale kwarg is added and the general kwargs are
        included to the kwargs of each axis.

        Each subclass should use the _add_axes_kwargs method to append new
        default dflt_axes_kwargs to the received axes_kwargs. This will
        have the effect that in the event the same kwarg is added at
        different levels, the kwarg as defined at the highest level will
        take precedence, effectively overriding any default defined at a
        lower level.

        Subclass can optionally rely on the default options, as defined by
        this ABC method, by simply implementing to return the return from
        this ABC method.
        """
        # add dflt_kwargs by-axis for this level
        dflt_kwargs = {
            "x": {"orientation": "horizontal"},
            "y": {"orientation": "vertical"},
        }
        if self._has_y2_plot:
            dflt_kwargs["y2"] = {"orientation": "vertical", "side": "right"}
        axes_kwargs_ = self._add_axes_kwargs(dflt_kwargs, axes_kwargs)

        # kwargs for every axis
        general_kwargs = deepcopy(general_kwargs)
        general_kwargs.setdefault("color", COLOR_CHART_TEXT)

        # compile kwargs for each axis in turn
        for axis_name, scale in self.scales.items():
            axis_kwargs = axes_kwargs_.get(axis_name, {})
            axis_kwargs["scale"] = scale
            for key, value in general_kwargs.items():
                axis_kwargs.setdefault(key, value)
            axes_kwargs_[axis_name] = axis_kwargs

        return axes_kwargs_

    def _create_axes(self) -> list[bq.BaseAxis]:
        """Create axes.

        Notes
        -----
        Guarantees List ordered according to:
            ['x', 'y', 'y2', 'color', 'opacity', 'size', 'rotation', 'skew']
        """
        axes = []
        axes_kwargs = self._axes_kwargs()

        for axis_name in [
            "x",
            "y",
            "y2",
            "color",
            "opacity",
            "size",
            "rotation",
            "skew",
        ]:
            axis_kwargs = axes_kwargs.pop(axis_name, None)
            if axis_kwargs is None:
                continue
            cls = bq.ColorAxis if axis_name == "color" else bq.Axis
            axis = cls(**axis_kwargs)
            axes.append(axis)

        return axes

    @property
    @abstractmethod
    def MarkCls(self) -> type[bq.Mark]:
        """Class of Mark.

        Notes
        -----
        Sublcass must implement to return Mark class.
        Instance of this class will instantiated by _create_mark.
        """

    @abstractmethod
    def _get_mark_y_data(self):
        """Get y data to pass to Mark.

        Notes
        -----
        Subclass should return y data to pass to Mark.
        """
        return self._y_data

    @property
    def MarkY2Cls(self) -> type[bq.Mark]:
        """Class of Mark for secondary y axis.

        Notes
        -----
        Subclass should override if alternative Mark class required to be
        plotted against secondary y axis.
        """
        return bq.Lines

    def _get_mark_y2_data(self):
        """Get y data to pass to secondary Mark.

        Notes
        -----
        Subclass should override if alternative return requried.
        """
        return self.data_y2

    def _tooltip_value(self, *args) -> str:
        """Tooltip value.

        Notes
        -----
        Optionally implement on subclass to set value of tooltip on
        hovering over mark.

        If implemented:
            Must have signature:
                _tooltip_value(self, mark, data)

            Should return HTML string to be displayed in tooltip when
            hovering over mark and index as provided by received
            parameters.
        """
        raise NotImplementedError("_tooltip_value is not implemented.")

    def _hover_handler(self, mark, data):
        self.mark.tooltip.value = self._tooltip_value(mark, data)

    @property
    def _has_tooltip(self) -> bool:
        """True if subclass handles mark hover."""
        try:
            return bool(self._tooltip_value())
        except BaseException as err:  # noqa: BLE001
            return not isinstance(err, NotImplementedError)

    @abstractmethod
    def _create_mark(self, **kwargs) -> bq.Mark:
        """bq.Mark from default kwargs.

        Notes
        -----
        Extend on subclass to pass through any additional kwargs.
        """
        has_tooltip = self._has_tooltip
        if has_tooltip:
            kwargs.setdefault("tooltip", w.HTML(value="<p>placeholder</p>"))
            kwargs.setdefault("tooltip_style", STYLE_TOOLTIP)
        kwargs["x"] = self._x_data
        kwargs["y"] = self._get_mark_y_data()

        kwargs["scales"] = self.scales.copy()
        if "y2" in kwargs["scales"]:
            del kwargs["scales"]["y2"]

        mark = self.MarkCls(**kwargs)
        if has_tooltip:
            mark.on_hover(self._hover_handler)
        return mark

    def _create_y2_mark(self, **kwargs) -> bq.Mark:
        """bq.Mark for secondary y-axis from default kwargs.

        Notes
        -----
        Optionally extend on subclass to pass through any additional
        kwargs.
        """
        kwargs["x"] = self._x_data
        kwargs["y"] = self._get_mark_y2_data()

        kwargs["scales"] = self.scales.copy()
        kwargs["scales"]["y"] = kwargs["scales"]["y2"]
        del kwargs["scales"]["y2"]

        return self.MarkY2Cls(**kwargs)

    @abstractmethod
    def _create_figure(self, **kwargs) -> bq.Figure:
        """bq.Figure from default kwargs.

        Notes
        -----
        Extend on subclass to pass through any additional kwargs. Can
        also pass through `title_style` to override default value.
        """
        marks = ([self.mark_y2] if self.mark_y2 is not None else []) + [self.mark]
        kwargs.setdefault("marks", marks)
        kwargs.setdefault("axes", self.axes)
        kwargs.setdefault("background_style", {"fill": "#222222"})
        kwargs.setdefault("title_style", STYLE_CHART_TITLE)
        return bq.Figure(**kwargs)

    def _add_marks(self):
        """Add extra marks to the figure.

        Notes
        -----
        Override on subclass to add extra marks at initalisation.
        `add_marks` should be used to add extra marks.
        """
        return

    # TITLE
    @property
    def title(self) -> str | None:
        """Chart title."""
        return self.figure.title

    @title.setter
    def title(self, title: str):
        self.figure.title = title

    # AXIS PRESENTATION
    @staticmethod
    def _update_axis_color(axis: bq.Axis):
        """Toggle color to get new ticks to take any existing axis color."""
        color = axis.color
        axis.color = "white" if color != "white" else "black"
        axis.color = color

    @abstractmethod
    def update_y_axis_presentation(self):
        """Update y axis presentation.

        Notes
        -----
        Subclass should implement to update y-axis ticks and other axis
        presentation. As a minium subclass should extend to execute
        method as defined on this ABC.
        """
        self._update_axis_color(self.axes[1])

    @abstractmethod
    def update_x_axis_presentation(self):
        """Update x axis presentation.

        Notes
        -----
        Subclass should implement to update x-axis ticks and other axis
        presentation. As a minium subclass should extend to execute
        method as defined on this ABC.
        """
        self._update_axis_color(self.axes[0])

    def update_y2_axis_presentation(self):
        """Update y2 axis presentation.

        Notes
        -----
        Subclass can optionally extend to further update secondary y-axis
        ticks and other axis presentation.
        """
        if len(self.axes) > 2:
            self._update_axis_color(self.axes[2])

    def update_presentation(self):
        """Update axis presentation."""
        self.update_y_axis_presentation()
        self.update_x_axis_presentation()
        self.update_y2_axis_presentation()

    # DATA UPDATE
    @hold_mark_update
    def _update_data(
        self, data: pd.DataFrame | pd.Series | None, data_y2: list[float | int] | None
    ):
        """Update chart data.

        Parameters
        ----------
        data
            Data with which to update the principle mark. Pass as None if
            only updating the secondary mark.

        data_y2
            Data with which to update the secondary mark. Pass as None if
            only updating the principle mark.
        """
        if data is not None:
            self.data = data
            self.mark.x = self._x_data
            self.mark.y = self._get_mark_y_data()
        if data_y2 is not None:
            assert self.data_y2 is not None
            data_ = data if data is not None else self.data
            assert len(data_y2) == len(data_)
            self.mark_y2.x = self._x_data
            self.mark_y2.y = self._get_mark_y_data()
        self.update_presentation()

    # MIXIN METHODS
    def update(
        self,
        data: pd.DataFrame | pd.Series,
        data_y2: list[float | int | list[float | int]] | None = None,
        title: str | None = None,
    ):
        """Create new chart from existing mark and figure.

        Parameters
        ----------
        data
            As 'data' parameter passed to constructor.

        data_y2
            As 'data_y2' parameter passed to constructor. Can only be
            passed if 'data_y2' was passed to constructor.

        title
            New chart title. If not passed will retain any existing title.
        """
        self._update_data(data, data_y2)
        if title:
            self.title = title

    def add_marks(
        self, marks: list[bq.Mark], group: AddedMarkKeys, under: bool = False
    ):
        """Add marks to the chart.

        Parameters
        ----------
        marks
            Marks to add.

        group
            Name of group to which to add marks. This can be subsequently
            be passed to `remove_added_marks` to remove the added marks.

        under
            True: place marks under existing marks.
            False: place marks on top of existing marks.
        """
        self._added_marks[group].extend(marks)
        if under:
            self.figure.marks = marks + list(self.figure.marks)
        else:
            self.figure.marks = list(self.figure.marks) + marks

    @property
    def added_marks(self) -> AddedMarks:
        """Marks that have been added to the chart.

        See Also
        --------
        add_marks : Add marks to the chart.
        added_marks_groups : Names of groups of added marks.
        remove_added_marks : Remove from the chart all marks in a group.
        """
        return self._added_marks

    def remove_added_marks(self, group: AddedMarkKeys):
        """Remove all marks in `group`."""
        marks = self._added_marks[group]
        for m in marks:
            m.close()
        self.figure.marks = [m for m in self.figure.marks if m not in marks]
        del self._added_marks[group]

    def remove_added_marks_all(self):
        """Remove all added marks."""
        for group in self.added_marks:
            self.remove_added_marks(group)

    def opacate_added_marks(
        self, groups: AddedMarkKeys | Sequence[AddedMarkKeys], opacity: float = 1.0
    ):
        """Set opacity of added marks by group."""
        if not isinstance(groups, Sequence):
            groups = [groups]
        for group in groups:
            marks = self.added_marks[group]
            for m in marks:
                m.opacities = [opacity]

    def _set_added_marks_visibility(
        self, visible: bool, groups: AddedMarkKeys | Sequence[AddedMarkKeys]
    ):
        """Set visibility of added marks by group."""
        if not isinstance(groups, Sequence):
            groups = [groups]
        for group in groups:
            marks = self.added_marks[group]
            for m in marks:
                m.visible = visible

    hide_added_marks = partialmethod(_set_added_marks_visibility, False)  # noqa: FBT003

    def show_added_marks(self, groups: AddedMarkKeys | Sequence[AddedMarkKeys]):
        """Show added marks, by group."""
        self._set_added_marks_visibility(True, groups)  # noqa: FBT003
        self.opacate_added_marks(groups)

    def hide_added_marks_all(self):
        """Make all added marks invisible."""
        for marks in self.added_marks.values():
            for m in marks:
                m.visible = False

    def show_added_marks_all(self, opacity: float = 1.0):
        for marks in self.added_marks.values():
            for m in marks:
                m.visible = True
                m.opacities = [opacity]

    def hide_y2(self):
        """Hide any mark plotted against data_y2 and any y2 axis."""
        if self._no_y2_plot:
            return
        self.mark_y2.visible = False
        self.axes[2].visible = False

    def show_y2(self):
        """Show any mark plotted against data_y2 and any y2 axis."""
        if self._no_y2_plot:
            return
        self.mark_y2.visible = True
        self.axes[2].visible = True

    def close(self):
        """Close all chart widgets."""
        for marks in self._added_marks.values():
            for m in marks:
                m.close()
        for widget in self._widgets:
            widget.close()

    def delete(self):
        """Delete all chart widgets."""
        self.remove_added_marks_all()
        self.close()
        for _ in range(len(self._widgets)):
            del self._widgets[0]

    def display(self):
        """Display chart."""
        IPython.display.display(self.figure)

    def _cycle_legend(self):
        """Move legend to next position."""
        i = ubq.LEGEND_LOCATIONS.index(self.figure.legend_location)
        try:
            new_location = ubq.LEGEND_LOCATIONS[i + 1]
        except IndexError:
            new_location = ubq.LEGEND_LOCATIONS[0]
        self.figure.legend_location = new_location


class BaseSubsetDD(Base):
    """Base to chart discontinuous dates or selectable subset of.

    x-axis will comprise of discrete dates (as opposed to all dates within
    a period).

    Subset of data can be plot by assigning an interval to plotted_x_ticks.

    Implementation
    --------------
    Class partially implements ABC Base:
        _x_data: pd.DatetimeIndex

        _axes_kwargs():
            Extended to add default kwargs to format x-axis ticks.

        _create_scales():
            Defines x scale. Subclass must extend to add at least 'y'
            scale.

    Subclass should implement all other aspects of ABC. Additionally,
    subclass implmentation should implement the following properties as
    required in accordance with the properties' doc:

        @property
        _update_mark_data_attr_to_reflect_plotted: bool default (False)

        @property
        _update_mark_y2_data_attr_to_reflect_plotted: bool default (True)

        @property
        _plotted_y(self) -> pd.DataFrame

        @property
        _plotted_y2(self) -> pd.DataFrame

    Settable Properties
    -------------------
    plotted_x_ticks: pd.DatetimeIndex.
        Currently plotted x-ticks.

    Properties
    ----------
    date_intervals: pd.DatetimeIndex
        All plottable date intervals.

    plottable_interval: -> pd.Interval
        Interval covering all plottable dates.

    plotted_interval(self) -> pd.Interval:
        Interval of plotted dates.

    plotted_date_intervals -> pd.IntervalIndex:
        Date intervals for plotted dates.

    tick_interval() -> pd.Timedelta:
        Duration covered by each x-axis timestamp.

    max_ticks -> Optional[int]:
        Maximum number of x-axis ticks that can be visible at any time.

    Methods
    -------
    reset_x_ticks():
        Reset plotted x_ticks to all available x_ticks.

    date_intervals_subset():
        Subset of date intervals overlapping with or contained by an interval.

    date_interval_containing() -> pd.Interval | None:
        Interval of date intervals containing a specific date.

    x_ticks_bv() -> np.ndarray:
        Boolean vector indicating x_ticks included to specific dates.

    x_ticks_subset() -> DatetimeIndex:
        x_ticks that fall within an interval.

    Notes
    -----
    THE QUESTION - on changing x_ticks, is it necessary to reassign the
    mark data attributes, i.e. mark.x, mark.y, mark.color etc...?

    The short answer - it depends on the mark.

    The longer answer...

    For most marks, changing the x-scale domain without changing the
    corresponding mark's data attributes has some kind of unwanted result,
    with effects ranging from merely annoying to outright unusable (for
    example, change the first date of the x-scale domain corresponding with
    a Lines mark and the plot no longer renders). In almost all cases these
    effects can be avoided by updating the mark data attributes at the same
    time as updating the x-scale domain. The trade-off is that the
    continual reassigning of mark data attributes results in a more jerky
    chart update when radidly changing the plotted x_ticks, via a slider
    for example.

    What to do?

    `BaseSubsetDD` sets a handler `_x_domain_chg_handler` to handle any
    change to the x-scale domain. The handler:
        Always calls `update_presentation` to update the axis
        presentation to reflect the new data.

        Optionally calls `_set_mark_to_plotted` and
        `_set_mark_y2_to_plotted` to reassign the mark's and mark_y2's data
        attributes (.x, .y, .color etc) to represent only the plotted data.
        This reassigning of the marks' data attributes, to reflect the
        plotted data, is NOT the default implementation for the principle
        mark, rather it has to be explicitly requested by the subclass
        subclass by way of overridng the property
        `_update_mark_data_attr_to_reflect_plotted` to return True.
        However, the default implmentation for mark_y2 IS to update the
        mark attribute values in this way to reflect the plotted dates. The
        subclass should override the
        `_update_mark_y2_data_attr_to_reflect_plotted` to return False in
        the event that this behaviour is not required.

    Worth noting, the BaseSubsetDD implementation is such that:
        Data for the primary mark is held in `data`, of which:
            y data available from `_y_data`
            x data exposed via `x_ticks`

        Data for the mark y2 mark is held in `data_y2`, of which:
            y data available from `data_y2` which is defined to allign
            with x data exposed via `x_ticks`

        PLOTTED data properties simply take a slice of the corresopnding
        ALL data property, with that slice reflecting the current x-scale
        domain. The plotted date properties are:
            plotted x data available from `plotted_x_ticks`

            plotted y data available from `_plotted_y` (or any exposed
            public method).

            plotted y2 data available from `_plotted_y2` (or any exposed
            public method).
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None = None,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        display: bool = False,
        data_y2: list[float | int] | None = None,
    ):
        """Create chart.

        Parameters
        ----------
        data
            Chart data, with rows indexed with a pd.IntervalIndex or, if
            data for daily prices, a pd.DatetimeIndex.

        title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show all.

        max_ticks
            Limit on number of x_ticks to include to x-axis. NB If passed
            together with `visible_x_ticks` and `visible_x_ticks` is
            longer than max_ticks then `visible_x_ticks` will be curtailed.

        display
            True to display chart.

        data_y2
            Values to be plotted against any secondary y axis.
        """
        self._max_ticks = max_ticks
        super().__init__(data, title, display=False, data_y2=data_y2)
        self.scales["x"].observe(self._x_domain_chg_handler, ["domain"])
        self.plotted_x_ticks = self._get_plot_interval(visible_x_ticks)
        if display:
            self.display()

    # DATES
    @property
    def _x_data(self) -> pd.DatetimeIndex:
        return self.data.index.left

    @property
    def x_ticks(self) -> pd.DatetimeIndex:
        """All plottable x-ticks."""
        return self._x_data

    @property
    def date_intervals(self) -> pd.IntervalIndex:
        """All plottable date intervals."""
        return self.data.index

    @property
    def plottable_interval(self) -> pd.Interval:
        """Interval covering all plottable dates. Closed left."""
        left = self.date_intervals[0].left
        right = self.date_intervals[-1].right
        return pd.Interval(left, right, closed="left")

    @lru_cache  # noqa: B019
    def _x_ticks_s(self) -> pd.Series:
        """X ticks as pd.Series."""
        return pd.Series(self.x_ticks)

    @lru_cache  # noqa: B019
    def _x_ticks_posix_raw(self) -> np.ndarray:
        return ubq.dates_to_posix(self.x_ticks)

    @lru_cache  # noqa: B019
    def _x_ticks_posix_l(self) -> list[np.int64]:
        return list(self._x_ticks_posix_raw())

    @property
    def _x_ticks_posix(self) -> list[np.int64]:
        return self._x_ticks_posix_l()

    def date_intervals_subset(
        self, interval: pd.Interval, overlap: bool = False
    ) -> pd.IntervalIndex:
        """Subset of date intervals covered by an interval.

        Parameters
        ----------
        interval
            Interval of date intervals to be returned as subset.

        overlap
            True to include all date intervals that overlap with interval.

            If False will only return date intervals that are full
            contained within interval.
        """
        return upd.intervals_subset(self.date_intervals, interval, overlap)

    def date_interval_containing(self, date: pd.Timestamp) -> pd.Interval | None:
        """Interval of date intervals containing a specific date."""
        ii = self.date_intervals[self.date_intervals.contains(date)]
        if len(ii) == 1:
            return ii[0]
        if ii.empty:
            return None
        msg = "Multiple date intervals contain" + str(date) + ":" + str(ii)
        raise ValueError(msg)

    def x_ticks_bv(self, dates: pd.DatetimeIndex | pd.Interval) -> np.ndarray:
        """Boolean vector indicating x_ticks included within specific dates.

        Parameters
        ----------
        dates
            Dates to check if included to x_ticks.

            If pd.DatetimeIndex passed then returned array will have True
            values for any x_tick present in DatetimeIndex.

            If pd.Interval passed then returned array will have True values
            for any x_tick that falls inside the interval.
        """
        if isinstance(dates, pd.DatetimeIndex):
            return np.isin(self.x_ticks, dates)
        return self._x_ticks_s().apply(lambda x: x in dates)

    def x_ticks_subset(self, interval: pd.Interval) -> pd.DatetimeIndex:
        """x_ticks that fall within an interval."""
        bv = self.x_ticks_bv(interval)
        return self.x_ticks[bv]

    def reset_x_ticks(self):
        """Reset plotted ticks to all available x_ticks."""
        self.scales["x"].domain = self._x_ticks_posix

    @property
    def max_ticks(self) -> int | None:
        """Maximum number of x-axis ticks that can be visible at any time."""
        return self._max_ticks

    @property
    def tick_interval(self) -> pd.Timedelta:
        """Duration covered by each x-axis timestamp."""
        freq_counts = self.date_intervals.length.value_counts()
        most_freq = max(freq_counts)
        return freq_counts[freq_counts == most_freq].index[0]

    # DOMAIN
    @property
    def _domain(self) -> list[np.int64]:
        return self.scales["x"].domain

    @property
    def _domain_start(self) -> np.int64:
        return self._domain[0]

    @property
    def _domain_end(self) -> np.int64:
        return self._domain[-1]

    @property
    def _domain_slice(self) -> slice:
        """Slice of indices of `dates` that currently comprise `_domain`."""
        search = np.array([self._domain_start, self._domain_end])
        a = np.searchsorted(self._x_ticks_posix_raw(), search)
        return slice(a[0], a[1] + 1)

    @property
    def _domain_bv(self) -> np.ndarray:
        """Boolean vector indciating dates that currently comprise domain."""
        return np.isin(self._x_ticks_posix_raw(), self._domain)

    # PLOTTED
    @property
    def plotted_x_ticks(self) -> pd.DatetimeIndex:
        return self.x_ticks[self._domain_bv]

    @plotted_x_ticks.setter
    def plotted_x_ticks(self, interval: pd.Interval):
        bv = self.x_ticks_bv(interval)
        self.scales["x"].domain = list(self._x_ticks_posix_raw()[bv])

    @property
    def plotted_date_intervals(self) -> pd.IntervalIndex:
        """Date intervals for plotted dates."""
        bv = self.x_ticks_bv(self.plotted_x_ticks)
        return self.date_intervals[bv]

    @property
    def plotted_interval(self) -> pd.Interval:
        """Interval of plotted dates."""
        left = self.plotted_date_intervals[0].left
        right = self.plotted_date_intervals[-1].right
        return pd.Interval(left, right, closed="left")

    def _get_plot_interval(
        self, visible_x_ticks: pd.Interval | None = None
    ) -> pd.Interval:
        """Get interval of x_ticks to plot, curtailed to any limit.

        Parameters
        ----------
        visible_x_ticks
            x_ticks to plot. Will be curtailed to any `self.max_ticks`
            limit. If not passed interval will be assued as currently
            plotted x_ticks.
        """
        pi = visible_x_ticks if visible_x_ticks is not None else self.plotted_interval

        if self.max_ticks is not None:
            x_ticks = self.x_ticks_subset(pi)
            if len(x_ticks) > self.max_ticks:
                pi = pd.Interval(x_ticks[-self.max_ticks], x_ticks[-1], closed="both")
        return pi

    # DATA
    def _clear_cache(self):
        """Clear value of cached property-like functions."""
        for f in ["_x_ticks_posix_raw", "_x_ticks_posix_l", "_x_ticks_s"]:
            getattr(self, f).cache_clear()

    @property
    def data(self) -> pd.DataFrame | pd.Series:
        """Chart data."""
        return super().data

    @data.setter
    def data(self, data: pd.DataFrame | pd.Series):
        self._clear_cache()
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.IntervalIndex.from_arrays(
                data.index, data.index + pd.Timedelta("1D"), closed="left"
            )
        super(BaseSubsetDD, BaseSubsetDD).data.__set__(self, data)

    # Y
    @property
    def _plotted_y(self) -> pd.DataFrame:
        """y_data as currently shown on plot.

        Notes
        -----
        Optionally concrete on subclass.
        """
        return self._y_data[self._domain_bv]

    @property
    def _plotted_y2(self) -> list[float | int | list[float | int]] | None:
        """y2_data as currently shown on plot."""
        if self._no_y2_plot:
            return None
        if isinstance(self.data_y2[0], list):
            return [lst[self._domain_slice] for lst in self.data_y2]
        return self.data_y2[self._domain_slice]

    # X TICK
    @property
    def _x_axis_tick_format(self):
        interval = self.plotted_interval.length
        if interval > pd.Timedelta(days=365):
            return "%y-%b-%d"
        if interval > pd.Timedelta(days=28):
            return "%b-%d"
        if interval > pd.Timedelta(days=1):
            if self.x_ticks.resolution == "minute":
                return "%d %H%M"
            return "%d"
        return "%H%M"

    def _set_x_tick_format(self):
        fmt = self._x_axis_tick_format
        # only set if changed
        if self.axes[0].tick_format != fmt:
            self.axes[0].tick_format = fmt

    # CHART CREATION
    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        dflt_axes_kwargs = {"x": {"tick_format": self._x_axis_tick_format}}
        axes_kwargs = self._add_axes_kwargs(dflt_axes_kwargs, axes_kwargs)
        return super()._axes_kwargs(axes_kwargs, **general_kwargs)

    def _create_scales(self) -> dict[ubq.ScaleKeys, bq.Scale]:
        """Create scales.

        Notes
        -----
        Subclass should extend method to add at least a 'y' scale.
        """
        return {"x": bq.OrdinalScale(domain=self._x_ticks_posix)}

    # UPDATE CHART
    @property
    def _update_mark_data_attr_to_reflect_plotted(self):
        """Does mark data need to be reassigned to relect plotted data.

        Notes
        -----
        Subclass can override to return True if require mark data
        attributes to be updated to always reflect plotted data.

        In event overriden to return True:
            Property documentation should note reason for requiring this
            behaviour.

            `_get_mark_y_plotted_data()` should be reviewed and overriden
            or extended in event that, as defined, it does not return an
            object that can be assigned to Mark.y to represent y data over
            plotted x_ticks.

            `_set_mark_to_plotted()` should be extended in the event the
            mark has any data attribute, other than 'x' and 'y', that also
            requires being set to reflect the plotted value.
        """
        return False

    @property
    def _update_mark_y2_data_attr_to_reflect_plotted(self):
        """Does mark_y2 data need to be reassigned to relect plotted data.

        Notes
        -----
        Subclass can override to return False if do NOT require mark_y2
        data attributes to be updated to always reflect plotted data. NB
        there is no need to override this property in the event that the
        class does provide for a mark_y2 mark.

        If relying on this property returning True then:
            `_get_mark_y_plotted_data()` should be reviewed and overriden
            or extended in event that, as defined, it does not return an
            object that can be assigned to `mark_y2.y` to represent y data
            over plotted x_ticks.

            `_set_mark_to_plotted()` should be extended in the event the
            mark has any data attribute, other than 'x' and 'y', that also
            requires being set to reflect the plotted value.
        """
        return True

    def _set_mark_x_to_plotted(self):
        self.mark.x = self.plotted_x_ticks

    def _set_mark_y2_x_to_plotted(self):
        self.mark_y2.x = self.plotted_x_ticks

    def _get_mark_y_plotted_data(self, multiple_symbols: bool = False):
        """Get y data for plotted dates.

        If `_update_mark_data_attr_to_reflect_plotted` is True (which
        requires subclass to explictely override) then this method should
        be reviewed / extended / overriden to ensure that it returns
        _y_data for the plotted dates.

        Method as defined will return y_data for plotted dates if
        `get_mark_y_data` returns a pd.DataFrame, pd.Series or a list where
        each element represents one y data point.

        Also, if `get_mark_y_data` returns a list of lists where each list
        represents a different symbol, and each element of each list
        represents one y data point, then subclass can extend this method
        to simply pass through `multiple_symbols` as True.
        """
        if not self._update_mark_data_attr_to_reflect_plotted:
            raise NotImplementedError

        mark_y_data = self._get_mark_y_data()
        if multiple_symbols:
            return [lst[self._domain_slice] for lst in mark_y_data]
        try:  # if pandas object
            plotted = mark_y_data[self._domain_bv]
        except AttributeError:  # if list
            plotted = mark_y_data[self._domain_slice]
        return plotted

    def _get_mark_y2_plotted_data(
        self, multiple_symbols: bool = False
    ) -> list[float | int | list[float | int]]:
        """Get y2 data for plotted dates.

        If `_update_mark_y2_data_attr_to_reflect_plotted` is True (the
        default) then this method should be reviewed / extended / overriden
        to ensure that it returns y2 data for the plotted dates.

        Method as defined will return y2 data for plotted dates if
        `get_mark_y2_data` returns a list where each element represents one
        y2 data point.

        Also, if `get_mark_y2_data` returns a list of lists where each list
        represents a different symbol, and each element of each list
        represents one y2 data point, then subclass can extend this method
        to simply pass through `multiple_symbols` as True.
        """
        if not self._update_mark_y2_data_attr_to_reflect_plotted:
            raise NotImplementedError

        mark_y2_data = self._get_mark_y2_data()
        if multiple_symbols:
            return [lst[self._domain_slice] for lst in mark_y2_data]
        return mark_y2_data[self._domain_slice]

    def _set_mark_y_to_plotted(self):
        self.mark.y = self._get_mark_y_plotted_data()

    def _set_mark_y2_y_to_plotted(self):
        self.mark_y2.y = self._get_mark_y2_plotted_data()

    def _set_mark_to_plotted(self):
        self._set_mark_x_to_plotted()
        self._set_mark_y_to_plotted()

    def _set_mark_y2_to_plotted(self):
        self._set_mark_y2_x_to_plotted()
        self._set_mark_y2_y_to_plotted()

    @hold_mark_update
    def _x_domain_chg_handler(self, event):  # noqa: ARG002
        if self._update_mark_data_attr_to_reflect_plotted:
            self._set_mark_to_plotted()
        if self._has_y2_plot and self._update_mark_y2_data_attr_to_reflect_plotted:
            self._set_mark_y2_to_plotted()
        self.update_presentation()

    def update_x_axis_presentation(self):
        self._set_x_tick_format()
        super().update_x_axis_presentation()

    @hold_mark_update
    def _update_data(  # noqa: C901
        self,
        data: pd.DataFrame | pd.Series | None,
        data_y2: list[float | int | list[float | int]] | None = None,
        visible_x_ticks: pd.Interval | None = None,
    ):
        """Update data.

        Parameters
        ----------
        data
            Data with which to update the principle mark. Pass as None if
            only updating the secondary mark with data_y2.

        data_y2
            Data with which to update the secondary mark. Pass as None if
            only updating the principle mark.

        visible_x_ticks
            x_ticks to plot post update. If not passed interval to plot
            will be assued as currently plotted x_ticks.

        Notes
        -----
        Overrides inherted method:
            Updates data according to whether mark attributes are set to
            reflect plotted dates.

            Provides for setting plotted dates.
        """
        if data is None and data_y2 is None:
            return

        if data is not None:
            self.data = data
            if not self._update_mark_data_attr_to_reflect_plotted:
                self.mark.x = self._x_data
                self.mark.y = self._get_mark_y_data()

        if data_y2 is not None:
            assert self._has_y2_plot
            data_ = data if data is not None else self.data
            if isinstance(data_y2[0], list):
                for lst in data_y2:
                    assert len(lst) == len(data_)
            else:
                assert len(data_y2) == len(data_)
            self.data_y2 = data_y2

            if not self._update_mark_y2_data_attr_to_reflect_plotted:
                self.mark_y2.x = self._x_data
                self.mark_y2.y = self._get_mark_y2_data()
            elif data is None:
                # shortcut out, only updating data_y2 and setting to plotted
                self._set_mark_y2_to_plotted()
                self.update_presentation()
                return

        # mark attributes will be updated to reflect plotted
        # and presentation will be updated all as part of setting
        # plotted_x_ticks implementation (via self._x_domain_chg_handler)
        plot_interval = self._get_plot_interval(visible_x_ticks)
        if plot_interval.length < self.tick_interval:
            subset = self.date_intervals_subset(plot_interval, overlap=True)
            plot_interval = upd.interval_of_intervals(subset, closed="left")
        prior_plotted_x_ticks = self.plotted_x_ticks.copy()
        self.plotted_x_ticks = plot_interval
        if prior_plotted_x_ticks.equals(self.plotted_x_ticks):
            # trigger handler manually as will not have fired if value unchanged
            self._x_domain_chg_handler(event=None)

    def update(
        self,
        data: pd.DataFrame | pd.Series,
        data_y2: list[float | int | list[float | int]] | None = None,
        title: str | None = None,
        visible_x_ticks: pd.Interval | None = None,
    ):
        """Update chart with new data.

        Title will remain as existing if `title` not passed.

        Parameters otherwise as constructor.
        """
        self._update_data(data, data_y2, visible_x_ticks)
        if title:
            self.title = title


class BasePrice(BaseSubsetDD):
    """Base class for price charts.

    Concretes `BaseSubsetDD` with y-axis defined for prices. Provides price
    data to be plotted against a linear or log scale.

    Implementation
    --------------
    Class partially implements ABC Base:
        _x_data: pd.DatetimeIndex

        _create_scales():
            Fully defined.

        _axes_kwargs():
            Extended to add default kwargs to this level. Subclasses
            should extend further to customise as required.

        _create_figure():
            Extended to add kwargs. Subclasses can extend to further
            customise.

    Subclass should implement all other aspects of ABC. Additionally,
    subclasses should implement the following:

        _get_mark_y_plotted_data():
            Override or extend in event method as defined on `BaseSubsetDD`
            does not return an object that can be assigned to Mark.y to
            represent y data over plotted dates.


    Settable Properties
    -------------------
    prices: -> pd.DataFrame:
        Prices for all plottable dates.


    Properties
    ----------
    plotted_prices: -> pd.DataFrame:
        Prices for currently plotted dates.

    high_plotted_price:
        High price of currently plotted data.

    low_plotted_price:
        Low price of currently plotted data.

    plotted_y2: -> pd.DataFrame:
        Values for any secondary mark plotted against y2 axes.

    high_plotted_y2:
        High price of any secondary mark plotted against y2 axes.

    low_plotted_y2:
        Low price of currently plotted data.

    logscale: -> bool
        True if prices plotted against a log scale.

    y_tick_increment: -> Optional[float]
        If logscale, pct increase between each successive y-axis label.

    Methods
    -------
    set_drawdown_presentation():
        Set presentation for a y2 mark that represents drawdown.
    """

    LOGSCALE_NUM_TICKS = 8

    # %age to extend scale by beyond max and min price plotted
    Y_AXIS_EXCESS = 0.05

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = False,
        data_y2: list[float | int | list[float | int]] | None = None,
    ):
        """Create chart.

        Parameters
        ----------
        data
            Chart data, with rows indexed with a pd.IntervalIndex.

         title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show
                all.

        max_ticks
            Limit on number of dates to include to x-axis. NB If passed
            together with `visible_x_ticks` and `visible_x_ticks` is longer
            than max_ticks then `visible_x_ticks` will be curtailed.

        log_scale
            True to plot prices against a log scale

        display
            True to display created chart.

        data_y2
            Values to be plotted against any secondary y axis.
        """
        self._log_scale_init = log_scale
        if isinstance(data, DataFrame):
            data.columns = data.columns.str.lower()
        super().__init__(data, title, visible_x_ticks, max_ticks, display, data_y2)

    @property
    def prices(self) -> pd.DataFrame:
        return self._y_data

    @prices.setter
    def prices(self, data: pd.DataFrame | pd.Series):
        """Set new prices for existing dates (to rebase for example).

        NB: This method should not be used if wish to simultaneously set
        values for a secondary y axis, in which case the `update` method
        must be used.
        """
        assert data.index.equals(self.data.index)
        self._update_data(data)

    @property
    def plotted_prices(self) -> pd.DataFrame:
        """Prices as currently shown on plot."""
        return self._plotted_y

    @property
    def high_plotted_price(self) -> float:
        return self.plotted_prices.values.max()

    @property
    def low_plotted_price(self) -> float:
        return self.plotted_prices.values.min()

    @property
    def plotted_y2(self) -> list[float | int | list[float | int]]:
        """Prices as currently shown on plot."""
        return self._plotted_y2

    @property
    def high_plotted_y2(self) -> float:
        if isinstance(self.plotted_y2[0], list):
            return max(itertools.chain.from_iterable(self.plotted_y2))
        return max(self.plotted_y2)

    @property
    def low_plotted_y2(self) -> float:
        if isinstance(self.plotted_y2[0], list):
            return min(itertools.chain.from_iterable(self.plotted_y2))
        return min(self.plotted_y2)

    def _create_scales(self) -> dict[ubq.ScaleKeys, bq.Scale]:
        scales = super()._create_scales()
        logscale = self._log_scale_init
        scales["y"] = bq.LogScale() if logscale else bq.LinearScale()
        if self._has_y2_plot:
            scales["y2"] = bq.LinearScale()
        return scales

    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        # For x, considered:
        #    label_offset': '2.5em'
        #    label: 'Date'
        # For y
        #    tick_format may not be appropriate for all, i.e. might be
        #        better to set according to y value, i.e. if under 1 then
        #        maybe only 4 significant places?
        #    label_offset': '3em'
        #    label: 'Price'
        dflt_axes_kwargs = {"x": {"num_ticks": 6}, "y": {"tick_format": ".6r"}}
        axes_kwargs = self._add_axes_kwargs(dflt_axes_kwargs, axes_kwargs)
        return super()._axes_kwargs(axes_kwargs, **general_kwargs)

    def _create_figure(self, **kwargs) -> bq.Figure:
        kwargs.setdefault("interaction", None)
        return super()._create_figure(**kwargs)

    @property
    def logscale(self) -> bool:
        """True if prices plotted against a log scale."""
        return isinstance(self.scales["y"], bq.LogScale)

    def _update_y_scale(self):
        rnge = self.high_plotted_price - self.low_plotted_price
        excess = rnge * self.Y_AXIS_EXCESS
        if self.logscale:
            # prevent negative minimum value for log
            self.scales["y"].min = self.low_plotted_price * 0.995
        else:
            self.scales["y"].min = self.low_plotted_price - excess
        self.scales["y"].max = self.high_plotted_price + excess

        if "y2" in self.scales:
            self.scales["y2"].max = self.high_plotted_y2
            self.scales["y2"].min = self.low_plotted_y2

    def _y_log_tick_increment(self) -> float | None:
        """Factor by which to raise each successive y-axis label."""
        num_increments = self.LOGSCALE_NUM_TICKS - 1
        high_low_ratio = self.high_plotted_price / self.low_plotted_price
        return high_low_ratio ** (1 / num_increments)

    @property
    def y_tick_increment(self) -> float | None:
        """Percenage increase between each successive y-axis label.

        Only when y-axis is a logscale, otherwise None.

        Percentage returned as float.
        """
        if not self.logscale:
            return None
        return round(self._y_log_tick_increment() - 1, 3)

    def update_y_axis_presentation(self):
        # If log scale, define labels at regular percentage intervals.
        if self.logscale:
            tick_increment = self._y_log_tick_increment()
            tick_values = [self.low_plotted_price * tick_increment**i for i in range(8)]
            self.axes[1].tick_values = tick_values
        super().update_y_axis_presentation()
        self._update_y_scale()

    def set_drawdown_presentation(self):
        """Set presentation for when data_y2 represents drawdown."""
        if self._no_y2_plot:
            return
        self.mark_y2.fill = "top"
        # following mark serves only to obscure the drawdown chart fill which otherwise
        # extends to positive values beyond 0
        bg_color = self.figure.background_style["fill"]
        scales = {"y": self.scales["y2"], "x": self.scales["x"]}
        ln = bq.Lines(
            x=[self._x_data[0], self._x_data[-1]],
            y=[0, 0],
            scales=scales,
            fill="top",
            fill_colors=[bg_color],
            fill_opacities=[1.0],
            colors=[bg_color],
            opacities=[0.0],
        )
        self.add_marks([ln], Groups.PERSIST, under=False)

        def update_block_line(event):
            ln.x = [event.new[0], event.new[-1]]

        self.mark_y2.observe(update_block_line, ["x"])


class Line(BasePrice):
    """Line chart for single financial instrument.

    Properties and Methods as covered by documentation of inherited classes.
    """

    def __init__(
        self,
        prices: pd.DataFrame | pd.Series,
        title: str | None = None,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = False,
        data_y2: list[float | int] | None = None,
    ):
        """Create chart.

        Parameters
        ----------
        prices
            Either series of close data or DataFrame with single column
            of close data. In either case, prices.index should be a
            pd.IntervalIndex.

        title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show all.

        max_ticks
            Limit on number of dates to include to x-axis. NB If passed
            together with `visible_x_ticks` and `visible_x_ticks` is longer
            than max_ticks then +visible_x_ticks+ will be curtailed.

        log_scale
            True to plot prices against a log scale

        display
            True to display created chart.

        data_y2
            Values to be plotted against any secondary y axis.
        """
        super().__init__(
            prices, title, visible_x_ticks, max_ticks, log_scale, display, data_y2
        )
        assert isinstance(self.data, Series) or len(self.data.columns) == 1

    @property
    def _y_data(self) -> Series:
        if isinstance(self.data, Series):
            return self.data
        return self.data.iloc[:, 0]  # single df column as Series

    def _get_mark_y_data(self) -> Series:
        return self._y_data

    @property
    def MarkCls(self) -> type[bq.Mark]:
        return bq.Lines

    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        dflt_axes_kwargs = {"y": {"grid_color": "#666666"}}
        if self._has_y2_plot:
            dflt_axes_kwargs["y2"] = {
                "grid_color": "SteelBlue",
                "grid_lines": "dashed",
            }
        axes_kwargs = self._add_axes_kwargs(dflt_axes_kwargs, axes_kwargs)
        return super()._axes_kwargs(axes_kwargs, **general_kwargs)

    def _create_mark(self, **__) -> bq.Lines:
        return super()._create_mark(
            colors=["DarkOrange"],
            stroke_width=1.2,
            fill="bottom",
            fill_colors=["Orange"],
            fill_opacities=[0.35],
        )

    def _create_y2_mark(self, **__) -> bq.Lines:
        # assumes require to represent drawdown
        return super()._create_y2_mark(
            colors=["LightSkyBlue"],
            stroke_width=0.7,
            fill="top",
            fill_colors=["PowderBlue"],
            fill_opacities=[0.15],
        )

    @property
    def _update_mark_data_attr_to_reflect_plotted(self):
        """Does mark data need to be reassigned to relect plotted data.

        Notes
        -----
        If mark data attributes are not updated then plot line fails to
        render if the first date is changed. If last date is changed then
        plot does render although any fill is inverted whilst the figure
        updates. Ugly.
        """
        return True


class MultLine(BasePrice):
    """Line Chart for multiple financial instruments.

    Properties
    ----------
    labels: -> list[str]
        Labels for all line plots.

    num_lines: -> int
        Number of line plots.

    rebase_date() -> pd.Timestamp:
        Date to which prices are rebased.

    Methods
    -------
    opacity():
        Set lines opacity.

    opaque():
        Make all lines fully opaque.

    fade_lines():
        Fade all lines.

    highlight_line()
        Highlight one line plot.

    show_one_y2_line():
        Show only one of any y2 lines.

    show_all_y2_lines():
        Show all y2 lines.

    cycle_legend():
        Move legend to next location.

    rebase(self, date: pd.Timestamp):
        Rebase prices to a specific date.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale=True,
        display=False,
        data_y2: list[list[float | int]] | None = None,
    ):
        """Create chart.

        Parameters
        ----------
        prices
            One column per symbol to be plotted, column name as symbol,
            values as close prices. prices.index as pd.IntervalIndex.

        title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show all.

        max_ticks
            Limit on number of dates to include to x-axis. NB If passed
            together with `visible_x_ticks` and `visible_x_ticks` is longer
            than max_ticks then `visible_x_ticks` will be curtailed.

        log_scale
            True to plot prices against a log scale

        display
            True to display created chart.

        data_y2
            Values to be plotted against any secondary y axis.
        """
        prices = upd.rebase_to_row(prices, 0)
        super().__init__(
            prices, title, visible_x_ticks, max_ticks, log_scale, display, data_y2
        )

    @property
    def labels(self) -> list[str]:
        """Labels of each line plot."""
        return list(self.data.columns.str.upper())

    def rebase(self, date: pd.Timestamp):
        """Rebase prices to a specific date."""
        i = self.x_ticks.get_loc(date)
        self.prices = upd.rebase_to_row(self.prices, i)

    @property
    def rebase_date(self) -> pd.Timestamp:
        """Date to which prices are currently rebased."""
        bv = self.prices.apply(lambda x: all(x == 100), axis=1)
        return self.prices[bv].index[0]

    def reset_x_ticks(self):
        self.rebase(self.prices.index[0].left)
        super().reset_x_ticks()

    @property
    def _y_data(self) -> pd.DataFrame:
        return self.data

    def _get_mark_y_data(self) -> list[list]:
        return upd.tolists(self._y_data)

    def _get_mark_y_plotted_data(self):
        return super()._get_mark_y_plotted_data(multiple_symbols=True)

    def _get_mark_y2_plotted_data(self):
        return super()._get_mark_y2_plotted_data(multiple_symbols=True)

    @property
    def MarkCls(self) -> type[bq.Mark]:
        return bq.Lines

    def _create_mark(self, **_) -> bq.Lines:
        return super()._create_mark(
            colors=ubq.COLORS_DARK_8,
            labels=self.labels,
            stroke_width=1,
            display_legend=True,
        )

    @property
    def _fill_opacity_dflt(self) -> float:
        return 0.15

    def _create_y2_mark(self, **_) -> bq.Lines:
        return super()._create_y2_mark(
            colors=ubq.COLORS_DARK_8,
            labels=self.labels,
            stroke_width=0.4,
            fill="none",  # client should assign any fill directly to `.mark_y2.fill`
            fill_colors=ubq.COLORS_DARK_8,
            fill_opacities=[self._fill_opacity_dflt] * self.num_lines,
        )

    def _create_figure(self, **_) -> bq.Figure:
        return super()._create_figure(legend_location="top-left")

    @property
    def _update_mark_data_attr_to_reflect_plotted(self):
        """Does mark data need to be reassigned to relect plotted data.

        Notes
        -----
        If mark data attributes are not updated then plot line fails to
        render if the first date is changed. NB If last date is changed
        then plot does update as expected.
        """
        return True

    def update(
        self,
        data: pd.DataFrame | pd.Series | None,
        data_y2: list[float | int | list[float | int]] | None = None,
        title: str | None = None,
        visible_x_ticks: pd.Interval | None = None,
    ):
        """Update chart with new data.

        Exisiting title will remain if `title` not otherwise passed.

        Other parameters as for constructor.
        """
        if data is not None:
            data = upd.rebase_to_row(data, 0)
        super().update(data, data_y2, title, visible_x_ticks)

    @property
    def num_lines(self) -> int:
        """Number of line plots."""
        return len(self.mark.y)

    def opacity(self, value: float):
        """Set all y1 lines to a certain opacity."""
        self.mark.opacities = [value] * self.num_lines

    def show_one_y2_line(self, index: int):
        """Show only one of any y2 lines.

        See Also
        --------
        show_all_y2_lines
        """
        if self._no_y2_plot:
            return
        self.show_y2()
        fill_opacities = [0.0] * self.num_lines
        fill_opacities[index] = self._fill_opacity_dflt
        self.mark_y2.fill_opacities = fill_opacities

        opacities = [0.0] * self.num_lines
        opacities[index] = 1.0
        self.mark_y2.opacities = opacities

    def show_all_y2_lines(self):
        """Show all y2 lines.

        See Also
        --------
        show_one_y2_line
        """
        if self._no_y2_plot:
            return
        self.show_y2()
        self.mark_y2.fill_opacities = [self._fill_opacity_dflt] * self.num_lines
        self.mark_y2.opacities = [1.0] * self.num_lines

    def opaque(self, y2: bool = True):
        """Make all y1 lines fully opaque.

        Parameters
        ----------
        y2 : bool
            Determines whether all y2 lines should be shown at default
            opacities.
        """
        self.opacity(1.0)
        if y2:
            self.show_all_y2_lines()

    def fade_lines(self):
        """Fade all y1 lines."""
        self.opacity(0.5)

    def highlight_line(self, index: int, hide_y2: bool = True):
        """Highlight one line by fading all others.

        Parameters
        ----------
        index : int
            Index position of line to highlight, as given, for example, by
            position of y co-ordinates in `mark.y` list or position of
            label in `labels`.

        hide_y2 : bool
            True to hide all of any y2 lines except that corresponding with
            `index`.
        """
        self.fade_lines()
        # self.mark.opacities[index] = 1.0 won't work as for change to be
        # registered requires changing the reference of the assigned list.
        # believe comes back to nature of traitlets implementation.
        opacities = copy(self.mark.opacities)
        opacities[index] = 1.0
        self.mark.opacities = opacities

        if hide_y2:
            self.show_one_y2_line(index)

    def cycle_legend(self):
        """Move legend to next location."""
        self._cycle_legend()


class OHLC(BasePrice):
    """OHLC Chart for single financial instrument.

    Properties and Methods as covered by documentation of inherited classes.
    """

    from_breaks = pd.IntervalIndex.from_breaks
    STYLE = pd.DataFrame(
        {"stroke_width": [1.0, 0.8, 0.5, 0.3, 0.1]},
        index=from_breaks([0, 50, 70, 180, 548, 365 * 100], closed="right"),
    )

    def __init__(
        self,
        prices: pd.DataFrame,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale=True,
        display=False,
        data_y2: list[float | int] | None = None,
    ):
        """Create OHLC chart.

        Parameters
        ----------
        prices
            Price data as `pd.DataFrame` with rows indexed with
            `pd.IntervalIndex`.

            Columns must include 'open', 'high', 'low' and either 'close'
            or 'adjclose' (if has both then 'close' will be used).

        title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show all.

        max_ticks
            Limit on number of dates to include to x-axis. NB If passed
            together with +visible_x_ticks+ and `visible_x_ticks` is longer
            than max_ticks then `visible_x_ticks` will be curtailed.

        log_scale
            True to plot prices against a log scale.

        display
            True to display created chart.

        data_y2
            Values to be plotted against any secondary y axis.
        """
        super().__init__(
            prices, title, visible_x_ticks, max_ticks, log_scale, display, data_y2
        )

    @property
    def _y_data(self) -> pd.DataFrame:
        return self.data[["open", "high", "low", "close"]]

    def _get_mark_y_data(self):
        return self._y_data.values.tolist()

    @property
    def MarkCls(self) -> type[bq.Mark]:
        return bq.OHLC

    def _tooltip_value(self, mark: bq.OHLC, event: dict) -> str:
        i = event["data"]["index"]
        row = self.data.iloc[i]
        color = mark.colors[0] if row.close >= row.open else mark.colors[1]
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>From: " + formatter_datetime(row.name.left)
        s += f"<br>To: {formatter_datetime(row.name.right)}"
        for line in ["open", "high", "low", "close"]:
            v = getattr(row, line)
            s += "<br>" + line.capitalize() + ": " + FORMATTERS[line](v)
        s += "</p>"
        return s

    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        dflt_axes_kwargs = {"y": {"grid_color": "#666666"}}
        if self._has_y2_plot:
            dflt_axes_kwargs["y2"] = {"grid_color": "SteelBlue", "grid_lines": "dashed"}
        axes_kwargs = self._add_axes_kwargs(dflt_axes_kwargs, axes_kwargs)
        return super()._axes_kwargs(axes_kwargs, **general_kwargs)

    def _create_mark(self, **kwargs) -> bq.OHLC:
        kwargs.setdefault("stroke", "white")
        kwargs.setdefault("stroke_width", 0.5)
        kwargs.setdefault("colors", ["SkyBlue", "DarkOrange"])
        return super()._create_mark(format="ohlc", marker="candle", **kwargs)

    def _create_y2_mark(self, **__) -> bq.Lines:
        return super()._create_y2_mark(
            colors=["LightSkyBlue"],
            stroke_width=0.7,
            fill="top",
            fill_colors=["PowderBlue"],
            fill_opacities=[0.15],
        )

    def _create_figure(self, **kwargs) -> bq.Figure:
        kwargs.setdefault("padding_x", 0.005)
        return super()._create_figure(**kwargs)

    def _update_stroke_width(self):
        mask = self.STYLE.index.contains(len(self.plotted_x_ticks))
        stroke_width = self.STYLE.loc[mask, "stroke_width"].values[0]
        self.mark.stroke_width = stroke_width

    def update_presentation(self):
        self._update_stroke_width()
        super().update_presentation()


Direction = Literal["horizontal", "vertical"]


class _PctChgBarBase(BaseSubsetDD):
    """Concretes Base for Bar Chart displaying precentage changes.

    Properties
    ----------
    dates: pd.DatetimeIndex
        Dates as x-axis ticks.

    pct_chgs: pd.DataFrame
        Percent change for each period.

    Properties (settable)
    ---------------------
    direction: Literal['horizontal', 'vertical']
        Bars direction.
    """

    TICK_FORMAT_PCT = "0.1%"

    def __init__(
        self,
        data: pd.DataFrame,
        title="Change every frequency",
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        direction: Direction = "vertical",
        display=False,
    ):
        """Create chart.

        Parameters
        ----------
        data
            One column per symbol. Column of float values representing
            percentage change over constant frequency. Index as
            `pd.IntervalIndex` with dates representing period (of constant
            frequency) that the corresponding float values offer the
            percentage change over.

        title
            Chart title

        visible_x_ticks
            x_ticks to initially show on x-axis. None to show all.

        max_ticks
            Limit on number of dates to include to x-axis. NB If passed
            together with `visible_x_ticks` and `visible_x_ticks` is longer
            than `max_ticks` then `visible_x_ticks` will be curtailed.

        direction
            Direction of bars.

        display
            True to display created chart.
        """
        self._direction: Direction = direction
        super().__init__(data, title, visible_x_ticks, max_ticks, display=False)
        self._set_direction(direction=direction)
        self.update_presentation()

        if display:
            self.display()

    @property
    def MarkCls(self) -> type[bq.Mark]:
        return bq.Bars

    @property
    def _y_data(self) -> pd.DataFrame:
        return self.data

    def _get_mark_y_data(self):
        return upd.tolists(self._y_data)

    def _get_mark_y_plotted_data(self):
        return super()._get_mark_y_plotted_data(multiple_symbols=True)

    @property
    def pct_chgs(self) -> pd.DataFrame:
        return self._y_data

    @property
    def plotted_pct_chgs(self) -> pd.DataFrame:
        return self._plotted_y

    # CHART CREATION
    def _create_scales(self) -> dict[str, bq.Scale]:
        scales = super()._create_scales()
        scales["y"] = bq.LinearScale()
        return scales

    def _axes_kwargs(
        self, axes_kwargs: AxesKwargs | None = None, **general_kwargs
    ) -> AxesKwargs:
        dflt_axes_kwargs = {
            "x": {"tick_style": {"text-anchor": "end"}},
            "y": {"tick_format": self._y_tick_format},
        }
        axes_kwargs = self._add_axes_kwargs(dflt_axes_kwargs, axes_kwargs)
        return super()._axes_kwargs(axes_kwargs, **general_kwargs)

    def _create_mark(self, **kwargs) -> bq.Mark:
        """Define on subclass to pass through any further customisation."""
        kwargs.setdefault("padding", 0.25)
        return super()._create_mark(**kwargs)

    def _create_figure(self, **kwargs) -> bq.Figure:
        """Define on subclass to pass through any further customisation."""
        kwargs.setdefault("padding_x", 0.01)
        kwargs.setdefault("padding_y", 0.01)
        return super()._create_figure(**kwargs)

    # PRESENTATION
    @property
    def _y_tick_format(self):
        if self._y_range > 0.2:
            return "0.0%"
        return self.TICK_FORMAT_PCT

    def update_y_axis_presentation(self):
        # required to concrete the abstract base method.
        super().update_y_axis_presentation()

    def update_x_axis_presentation(self):
        if self.direction == "vertical":
            self.axes[0].tick_rotate = 270
        else:
            self.axes[0].tick_rotate = 0
        n = 15
        self.axes[0].num_ticks = n if len(self.plotted_x_ticks) > n else None
        super().update_x_axis_presentation()

    # DIRECTION
    def _set_x_axis_direction(self, direction: Direction):
        axis = self.axes[0]
        axis.orientation = direction

    def _set_mark_for_direction(self, direction: Direction):
        self.mark.orientation = direction
        self.mark.align = "right" if direction == "vertical" else "left"

    def _set_figure_for_direction(self, direction: Direction):
        self.figure.padding_y = 0.05 if direction == "vertical" else 0
        self.figure.padding_x = 0 if direction == "vertical" else 0.05

    def _set_direction(self, direction: Direction):
        x_direction = "horizontal" if direction == "vertical" else "vertical"
        self._set_x_axis_direction(x_direction)
        self.axes[1].orientation = direction
        self._set_mark_for_direction(direction)
        self._set_figure_for_direction(direction)

    @property
    def direction(self) -> Direction:
        """Bars direction."""
        return self._direction


class PctChgBar(_PctChgBarBase):
    """Bar Chart displaying precentage changes."""

    COLOR_SCALE = ["crimson", "white", "darkgreen"]

    def _create_scales(self) -> dict[str, bq.Scale]:
        scales = super()._create_scales()
        scales["color"] = bq.ColorScale(
            colors=self.COLOR_SCALE, mid=0, min=self._y_min, max=self._y_max
        )
        return scales

    def _axes_kwargs(self) -> AxesKwargs:
        axes_kwargs = {"color": {"tick_format": self.TICK_FORMAT_PCT, "num_ticks": 8}}
        return super()._axes_kwargs(axes_kwargs)

    def _tooltip_value(self, mark, data):
        i = data["data"]["index"]
        y = data["data"]["y"]
        color_i = 0 if y < 0 else -1
        color = mark.scales["color"].colors[color_i]
        style = tooltip_html_style(color=color, line_height=1.3)
        s = f"<p {style}>Date: " + formatter_datetime(self.plotted_x_ticks[i])
        s += "<br>Chg: " + FORMATTERS["pct_chg"](y) + "</p>"
        return s

    def _create_mark(self, **_) -> bq.Mark:
        kwargs = {
            "color": self.pct_chgs,
            "label_display": True,
            "label_display_format": self.TICK_FORMAT_PCT,
            "label_font_style": {"fill": COLOR_CHART_TEXT, "font-size": "12px"},
        }
        return super()._create_mark(**kwargs)

    def _create_figure(self, **kwargs) -> bq.Figure:
        kwargs.setdefault("padding_x", 0.05)
        kwargs.setdefault("padding_y", 0.05)
        return super()._create_figure(**kwargs)

    @property
    def _update_mark_data_attr_to_reflect_plotted(self):
        """Does mark data need to be reassigned to relect plotted data.

        Notes
        -----
        If mark data attributes are not updated then when bars are
        orientated vertically there's an artefact at the right of the
        chart, a bar that seems to be a composite of all others.
        """
        return True

    def _set_mark_color_to_plotted(self):
        self.mark.color = self._get_mark_y_plotted_data()

    def _set_mark_to_plotted(self):
        super()._set_mark_to_plotted()
        self._set_mark_color_to_plotted()

    def update_color_axis_presentation(self):
        self._update_axis_color(self.axes[2])

    def update_presentation(self):
        """Update axis presentation."""
        super().update_presentation()
        self.update_color_axis_presentation()

    def _set_color_axis_direction(self, direction: Direction):
        self.axes[2].side = "right" if direction == "vertical" else "bottom"

    def _set_mark_for_direction(self, direction: Direction):
        super()._set_mark_for_direction(direction)
        offset = -14 if direction == "vertical" else 34
        self.mark.label_display_vertical_offset = offset

    def _set_figure_for_direction(self, direction: Direction):
        if direction == "vertical":
            self.figure.fig_margin = {"top": 70, "bottom": 60, "left": 60, "right": 80}
        else:
            self.figure.fig_margin = {"top": 70, "bottom": 80, "left": 60, "right": 50}
        super()._set_figure_for_direction(direction)

    def _set_direction(self, direction: Direction):
        self._set_color_axis_direction(direction)
        super()._set_direction(direction)


class PctChgBarMult(_PctChgBarBase):
    """Bar Chart displaying precentage changes of multiple instruments.

    Methods
    -------
    cycle_legend():
        Move legend to next location.

    Notes (including known BUGS)
    ----------------------------
    If mark type is changed to 'grouped' thereafter there's an artefact of
    a composite bar that appears on the far right of the chart (NB only
    appears if bars are alligned vertically). NB occurs from first instance
    that type set to grouped and there's not getting rid of it once it's
    there.

    When mark type is 'grouped' dates on x-axis will not update when dates
    changed. Will 'unblock' if change type 'stacked' and then change dates
    again.

    NB Can lose the above bugs if change implementation to update mark data
    attributes to relfect plotted dates, however, then the bugs are worse:
        If mark type is set to 'grouped' thereafter the figure fails to
        update. Redisplaying the figure shows an empty chart. This is the
        case regardless of whether the chart type is changed back to
        'stacked' before changing the plotted dates. It appears that after
        updating the mark.x and /or mark.y values the figure fails to
        recognise those, or from then on any other, changes.

    In short, the choosen implementation causes the lesser undesirable
    bugs.
    """

    def _tooltip_value(self, mark, data):
        i = data["data"]["index"]
        ci = data["data"]["colorIndex"]
        row = self.data.iloc[i]
        s = (
            "<p"
            + tooltip_html_style(color="white")
            + ">Date: "
            + formatter_datetime(row.name.left)
        )
        for i, tup in enumerate(row.items()):
            symbol, y = tup
            symbol_style = tooltip_html_style(color=mark.colors[i])
            symbol_span = f"<span {symbol_style}>{symbol}: </span>"
            chg = FORMATTERS["pct_chg"](y)
            chg_color = "crimson" if y < 0 else "darkgreen"
            chg_style = tooltip_html_style(color=chg_color)
            chg_span = f"<span {chg_style}>{chg}</span>"
            ss = f"{symbol_span}{chg_span}"
            weight = "bold" if i == ci else "normal"
            div_style = f'style="line-height: 1.3; font-weight: {weight}"'
            ss = f"<div {div_style}>{ss}</div>"
            s += ss
        s += "</p>"
        return s

    def _create_mark(self, **_) -> bq.Mark:
        kwargs = {
            "colors": bq.CATEGORY10,
            "labels": list(self.data.columns),
            "display_legend": True,
        }
        return super()._create_mark(**kwargs)

    def _create_figure(self, **kwargs) -> bq.Figure:
        kwargs.setdefault("legend_location", "top-right")
        return super()._create_figure(**kwargs)

    def cycle_legend(self):
        """Move legend to next location."""
        self._cycle_legend()


class OHLCCaseBase(OHLC, ChartSupportsCasesGui):
    """Base for analysis over OHLC for single financial instrument.

    Base class to overlay a OHLC chart with marks representing analysis.
    Analysis can comprise multiple 'cases', where a case could represent,
    for example, a trend or a position. Base provides for displaying all
    classes collectively or focusing on a single 'current' case and
    navigating forwards or backwards between consecutive cases.

    Subclass Implementation
    -----------------------
    The following methods have a default implemntation which a subclass can
    extend or override as required.

        focus_case
            Focus on a given `case`.

        handle_new_case
            Handler for when a new case has been selected.

        add_case_marks
            Add marks for a specific case.

        reset_marks
            Reset added marks to show all cases.

        deselect current case
            Current case to None, show all cases.

        select_case
            Select a specific case.

    The following methods can be optionally implemented by the subclass
    if requried:

        update_trend_mark
            Update added marks to reflect the plotted data in order to
            avoid unwanted artefacts.

    Parameters
    ----------
    As for OHLC class, except:

    cases
        Analysis cases to be displayed on the chart.

    handler_click_case
        Client-defined handler to be called when a new case is selected
        after having clicked a specific case.
        Should have signature:
            f(case: CaseSupportsChartAnaly, mark: bq.Scatter, event: dict):

        NOTE this is not the same as the 'handle_new_case' method. On
        an event being triggered the 'handle_new_case' method as defined
        on the subclass will be executed first, followed by
        `handler_click_case` if passed.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        cases: CasesSupportsChartAnaly,
        title: str,
        visible_x_ticks: pd.Interval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = False,
        handler_click_case: Callable | None = None,
        data_y2: list[float | int] | None = None,
    ):
        self._client_handler_new_case = handler_click_case
        self.cases = cases
        self._current_case: CaseSupportsChartAnaly | None = None
        super().__init__(
            prices, title, visible_x_ticks, max_ticks, log_scale, display, data_y2
        )

        if max_ticks is not None or visible_x_ticks is not None:
            self.update_trend_mark()

    def create_scatter(
        self,
        x: pd.DatetimeIndex | list[pd.Timestamp],
        y: pd.Series | list[float],
        colors: str | list[str],
        marker: str,
        hover_handler: Callable | None,
        click_handler: Callable | None = None,
        *,
        opacities: list[int] | None = None,
    ) -> bq.Scatter:
        """Convenience method to create a scatter mark.

        Parameters
        ----------
        x
            `pd.DatetimeIndex` or list of pd.Timestamp` representing x
            values of scatter points.

        y
            y values of scatter points.

        colors
            Scatter point color or colors.

            Pass as str to apply same color to all points, for exmaple
            "skyblue".

            Pass as list[str] to apply different colors to each point.
            If length of list is shorter than length of `x` then colors in
            list will be repeatedly cycled through.

        marker
            Scatter point marker, for example "circle".

        hover_handler
            Callback to call if a scatter point is hovered over.

        click_handler
            Callback to call if a scatter point is LMB clicked.

        opacities
            Scatter opacities. Length of int of length equal or less than
            `x`. Each index represents opacity of scatter point at
            corresponding index of `x`. If length is less than that length
            of x then pattern will repeat, e.g. passing as [1] will result
            in all points being fully opaque. Defaults to [1].
        """
        if opacities is None:
            opacities = [1]

        if isinstance(colors, str):
            colors = [colors]
        mark = bq.Scatter(
            scales=self.scales,
            colors=colors,
            x=x,
            y=y,
            marker=marker,
            opacities=opacities,
            tooltip=w.HTML(value="<p>placeholder</p>"),
            tooltip_style=STYLE_TOOLTIP,
        )
        if hover_handler is not None:
            mark.on_hover(hover_handler)
        if click_handler is not None:
            mark.on_element_click(click_handler)
        return mark

    def _create_mark(self, **_) -> bq.OHLC:
        # dampen OHLC mark being overlaid which cases analysis
        opacities = [0.7] * len(self.prices)
        return super()._create_mark(opacities=opacities)

    def _add_scatter(
        self,
        x: pd.DatetimeIndex | list[pd.Timestamp],
        y: pd.Series | list[float],
        colors: str | list[str],
        marker: str,
        hover_handler: Callable | None,
        click_handler: Callable | None = None,
        *,
        opacities: list[int] | None = None,
        group: str | Literal[Groups.CASES_SCATTERS] = Groups.CASES_SCATTERS,
    ) -> bq.Scatter:
        """Add a scatter mark to a `group`."""
        mark = self.create_scatter(
            x, y, colors, marker, hover_handler, click_handler, opacities=opacities
        )
        self.add_marks([mark], group)
        return mark

    def hide_cases(self):
        """Hide marks identifying multiple cases.

        Subclass can extend / override as required.
        """
        added_groups = set(self.added_marks.keys())
        for group in CASES_GROUPS & added_groups:
            self.hide_added_marks(group)

    def show_cases(self):
        """Show marks identifying multiple cases.

        Subclass can extend / override as required.
        """
        added_groups = set(self.added_marks.keys())
        for group in CASES_GROUPS & added_groups:
            self.show_added_marks(group)

    @property
    def current_case(self) -> CaseSupportsChartAnaly | None:
        """Current selected case."""
        return self._current_case

    def remove_case_marks(self):
        """Remove any marks added under `Groups.CASE`."""
        if Groups.CASE in self.added_marks:
            self.remove_added_marks(Groups.CASE)

    @staticmethod
    def show_only_one_scatter_point(index: int, scatter: bq.Scatter):
        """Show only the point of a given `scatter` at a given `index`."""
        arr = np.array([1 if i == index else 0 for i in range(len(scatter.x))])
        scatter.opacities = arr

    def show_only_one_cases_scatter_point(
        self, index: int, exclude_scatters: list[int] | None = None
    ):
        """Show only one point of each scatter added to CASES_SCATTERS.

        Has no effect if no marks are added under CASE_SCATTERS.

        Parameters
        ----------
        index
            Index of scatter point to show.

        exclude_scatters
            Any indices of scatter marks under CASES_SCATTERS that should
            be excluded. Visibility of all points of these scatter marks
            will not be altered.
        """
        for i, scatter in enumerate(self.added_marks.get(Groups.CASES_SCATTERS, [])):
            if exclude_scatters is not None and i in exclude_scatters:
                continue
            self.show_only_one_scatter_point(index, scatter)

    def show_only_one_other_mark(self, index: int, group: AddedMarkKeys):
        """Show only one mark in a group of added marks.

        Parameters
        ----------
        index
            Index of mark to show.

        group
            Group of added marks from which to show one mark.
        """
        marks = self.added_marks[group]
        arr = np.array([1 if i == index else 0 for i in range(len(marks))])
        for mark, opacity in zip(marks, arr, strict=True):
            mark.opacities = [opacity]

    def show_only_one_cases_other_mark(self, index: int):
        """Show only one mark in each of marks added under CASES_OTHERS_*.

        Has no effect if no marks are added under CASE_OTHERS.

        Parameters
        ----------
        index
            Index of mark(s) to show.
        """
        groups = set(self.added_marks.keys()) & CASES_OTHER
        for group in groups:
            self.show_only_one_other_mark(index, group)

    def _add_case_marks(self, case: CaseSupportsChartAnaly, mark: bq.Scatter):
        """Add marks for a specific case.

        Subclass can implement to add marks under the default
        implementation of `focus_current_case`.
        """

    def focus_case(self, case: CaseSupportsChartAnaly, mark: bq.Scatter):
        """Focus on a give `case`.

        Subclass can override this default implementation.
        """
        self.remove_case_marks()
        idx = self.cases.get_index(case)
        self.show_only_one_cases_scatter_point(idx)
        self.show_only_one_cases_other_mark(idx)
        self._add_case_marks(case, mark)

    def handle_new_case(
        self,
        case: CaseSupportsChartAnaly,
        mark: bq.Scatter,
        event: dict,  # noqa: ARG002
    ):
        """Handles new case having been selected.

        Subclass can extend or override as required.
        """
        self.focus_case(case, mark)

    def _handler_click_case(self, mark: bq.Scatter, event: dict):
        """Handler for clicking a scatter representing a case."""
        self._current_case = case = self.cases.event_to_case(mark, event)
        self.handle_new_case(case, mark, event)
        if self._client_handler_new_case is not None:
            self._client_handler_new_case(case, mark, event)

    def update_trend_mark(self, *_, **__):
        """Update trend mark to reflect plotted x ticks.

        Notes
        -----
        Like for the OHLC class, the OHLC mark is not updated to reflect
        changes to the plotted data
        (`_update_mark_data_attr_to_reflect_plotted` is False for the
        class) as the nature of the mark (discrete renders rather than a
        continuous line) does not result in any side-effects when the
        visible domain is shorter than that the mark data. The same is true
        (more or less) for the Scatter marks. However, to avoid unwanted
        artefacts other types of mark may require updating to reflect the
        plotted data. For example, a FlexLine will otherwise double back
        over the render as it tries to plot the 'next point after the
        visible range' to the start of the x-axis.

        This method serves as a handler which should be called by a client
        whenever the plotted dates are changed. To handle everything within this
        class had originally overriden the `self._x_domain_chg_handler` method
        which is invoked whenever the x scales domain is changed (i.e. whenever
        the plotted dates are changed), however, and regardless of whether
        holding off the sync to the frontend of not, aftefacts persisted
        for the FlexLine and other rendering issues manifested.

        In short, considered preferable to provide this handler that the
        subclass can customise if/as required to offer a specific solution
        for the marks added.
        """

    def reset_marks(self):
        """Reset added marks to show all cases."""
        self.remove_case_marks()
        self.show_added_marks_all()

    def deselect_current_case(self):
        """Reset added marks to show all cases.

        Removes any marks for current case. Makes all other added marks
        visible.
        """
        self.reset_marks()
        self._current_case = None

    def _simulate_click_case(self, index: int, scatter_index: int = 0):
        """Simulate clicking an element of scatter representing cases.

        Subclass can override or pass through args as required.

        Parameters
        ----------
        index
            Index of scatter point to be clicked within the `bq.Scatter`
            mark represented by `scatter_index`.

        scatter_index
            Index of `bq.Scatter` mark, within the added marks under
            `Groups.CASES_SCATTERS`, which represents cases. This will be
            the scatter with the points that can be clicked to select a
            specific case.
        """
        event = {"data": {"index": index}}
        scatter = self.added_marks[Groups.CASES_SCATTERS][scatter_index]
        self._handler_click_case(scatter, event)

    def select_case(self, case: CaseSupportsChartAnaly):
        """Select a specific case.

        Notes
        -----
        This default implementation ASSUMES that the scatter mark
        representing a case is the first scatter mark added to
        Groups.CASES_SCATTERS and that this scatter mark represents all
        cases in the same order as `self.cases`.

        Subclass can override as required to select a spedific case.
        """
        i = self.cases.get_index(case)
        self._simulate_click_case(i)

    def get_next_case(self) -> CaseSupportsChartAnaly:
        """Get case that follows the current selected case.

        If no case is currently selected then returns first case.
        """
        case = self.current_case
        i = 0 if case is None else self.cases.get_index(case) + 1
        if i == len(self.cases.cases):
            if TYPE_CHECKING:
                assert case is not None
            return case  # current case is last case
        return self.cases.cases[i]

    def select_next_case(self):
        """Select case that follows the current selected case.

        If no case is currently selected then selects first case.
        """
        case = self.get_next_case()
        if case is self.current_case:
            return  # current case if last case and already selected
        self.select_case(case)

    def get_previous_case(self) -> CaseSupportsChartAnaly:
        """Get case immediately prior to the current selected case.

        If no case is currently selected then returns last case.
        """
        case = self.current_case
        if case is not None:
            i = self.cases.get_index(case) - 1
            if i == -1:
                return case  # current case is first case
        else:
            i = -1
        return self.cases.cases[i]

    def select_previous_case(self):
        """Select case prior to the current selected cae.

        If no case is currently selected then selects last case.
        """
        case = self.get_previous_case()
        if case is self.current_case:
            return  # current case if first case and already selected
        self.select_case(case)
