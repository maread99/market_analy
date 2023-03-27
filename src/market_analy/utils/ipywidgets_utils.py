"""Utility functions and classes for ipywidgets.

Decorators
----------
throttle:
    Reduces number of calls to callbacks.

Classes
--------
NamedChildren:
    Add dictionary-like functionality to a ipywidgets container.

BoxNC:
    Box with NamedChildren.

ButtonIcon:
    Button displaying only icon.

IconRow:
    Row of pre-defined ButtonIcon objects.

DateRangeSlider:
    Double handled slider to select date ranges.
"""

from __future__ import annotations

from collections.abc import Callable
import asyncio
from time import time

import ipywidgets as w  # type: ignore[import]
import pandas as pd


class Timer:
    """Timer.

    Ref:
    https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#debouncing
    """

    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """Decorator to postpone callback.

    Postpones callback until `wait` seconds have elapsed during which the
    value remains unchanged.

    Ref:
    https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#debouncing
    """

    def decorator(fn):
        timer = None
        timer2 = None

        def debounced(*args, **kwargs):
            nonlocal timer, timer2

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            if timer2 is not None:
                timer2.cancel()
            timer = Timer(wait, call_it)
            timer.start()
            timer2 = Timer(wait * 2, call_it)
            timer2.start()

        return debounced

    return decorator


def throttle(wait) -> Callable:
    """Decorator to limit callbacks.

    Limits decorated callback to being called no more than once every
    `wait` period.

    Ref:
    https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#throttling
    """

    def decorator(fn: Callable) -> Callable:
        time_of_last_call = 0
        scheduled, timer = False, None
        new_args, new_kwargs = None, None

        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer

            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False

            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()

        return throttled

    return decorator


class NamedChildren:
    """Mixin adds dictionary-like functionality to an ipywidgets container.

    Provides ipywidgets container with following functionality:
        Assign a name to each child widget.
        Access child widget with [] notation,
            e.g. container['child_name'].
        Assign new child widget with [] notation,
            e.g. container['child_name'] = child_widget
        Delete child widget with [] notation,
            e.g. del container['child_name']
        Use len(container) to return number of child widgets
        Use keyword 'in' to evaluate if a name is the name of a child widget,
            e.g. 'name' in container
        Use iter(container) to return iterator of child widget names
        Use container.pop('child_name') to remove and return widget
            named 'child_name'

    Implementation
    --------------
    By example:

        class BoxNamed(w.Box, NamedChildren):

            def __init__(self, children, names, **kwargs):
                super().__init__(children, **kwargs)
                NamedChildren.__init__(self, children, names)
    """

    def __init__(self, children: list[w.Widget], names: list):
        """Constructor.

        Parameters
        ----------
        children
            ipywidgets passed, independently(!), to container at time of
            instantiation.

        names
            Names to be associated with children. Must be of same length
            as children.
        """
        assert len(names) == len(children)
        assert w.Widget in type(self).__mro__
        self._mapping = dict(zip(names, children))

    @property
    def mapping(self) -> dict[str, w.Widget]:
        return self._mapping

    @property
    def names(self) -> list[str]:
        return list(self.mapping.keys())

    def __getitem__(self, name: str):
        return self.mapping[name]

    def __setitem__(self, name: str, widget: w.Widget):
        self._mapping[name] = widget
        self.children += [widget]

    def __delitem__(self, name: str):
        widget = self[name]
        self.mapping.pop(name)
        self.children = [c for c in self.children if c != widget]

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name: str):
        return name in self.names

    def pop(self, name: str) -> w.Widget:
        widget = self[name]
        self.__delitem__(name)
        return widget


class BoxNC(w.Box, NamedChildren):
    """`w.Box` with NamedChildren Mixin.

    Parameters
    ----------
    children: List
        As `w.Box`

    names: List
        As NamedChildren

    **kwargs:
        passed to `w.Box`.
    """

    def __init__(self, children: list[w.Widget], names: list, **kwargs):
        super().__init__(children, **kwargs)
        NamedChildren.__init__(self, children, names)


class ButtonIcon(w.Button):
    """Button displaying only icon.

    By default button sized a little larger than icon image.

    Class methods
    -------------
    The following are convenience methods to produce `ButtonIcon` objects
    for a specific purpose.

        ButtonIcon.refresh(): refresh button

        ButtonIcon.close(): close button, displaying the 'x'

        ButtonIcon.max_dates(): 'max_dates' button, displaying the double
        headed horizontal arrow.

        ButtonIcon.resize_window(): 'arrows-alt' button.
    """

    @classmethod
    def refresh(
        cls, icon: str = "refresh", color: str = "darkorange", **kwargs
    ) -> w.Button:
        return cls(icon, color, **kwargs)

    @classmethod
    def close_(
        cls, icon: str = "close", color: str = "firebrick", **kwargs
    ) -> w.Button:
        return cls(icon, color, **kwargs)

    @classmethod
    def max_dates(
        cls, icon: str = "arrows-h", color: str = "dodgerblue", **kwargs
    ) -> w.Button:
        return cls(icon, color, **kwargs)

    @classmethod
    def resize_window(
        cls, icon: str = "arrows-alt", color: str = "seagreen", **kwargs
    ) -> w.Button:
        return cls(icon, color, **kwargs)

    def __init__(self, icon: str, color: str | None = None, **kwargs):
        """Constructor.

        Parameters
        ----------
        icon
            Name of font awesome icon to display on button. See:
                https://fontawesome.com/v4.7.0/icons/

        color
            Button color
        """
        layout = kwargs.pop("layout", w.Layout())
        for attr in ["width", "height"]:
            if getattr(layout, attr) is None:
                setattr(layout, attr, "35px")
        super().__init__(icon=icon, layout=layout, **kwargs)

        self._color: str | None
        self.color = color

    @property
    def color(self) -> str | None:
        """Button color."""
        return self._color

    @color.setter
    def color(self, color: str | None):
        self._color = color
        self.style.button_color = self._color


class IconRow(BoxNC):
    """Box of pre-defined `ButtonIcon` objects arranged in a row.

    Available ButtonIcon objects are those offered by ButtonIcon class
    methods.

    Each ButtonIcon accessible via [] notation with key as the name of the
    ButtonIcon class method that created the ButtonIcon, for example if the
    icon row includes the 'close' ButtonIcon then this can be accessed by
    icon_row['close'].

    Incorporates functionality of `NamedChildren` Mixin:

        Custom ButtonIcon's can be added by simple assignment, for example:
            icon_row['name'] = button_icon.

        Any ButtonIcon can be removed with del icon_row['name'].
    """

    dflt_layout = w.Layout(align_self="flex-start", height="30px", grid_gap="10px")

    def __init__(
        self,
        icon_button_names: list[str],
        handlers: list[Callable] | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        icon_button_names
            List of icon_buttons, each defined as the name of the a
            ButtonIcon class method. For exmaple:
                icon_buttons = ['refresh', 'close', 'max_dates']

        handlers
            Handlers to be assigned to ButtonIcon objects, where handler
            will be called on ButtonIcon being clicked by user. Must have
            same length as icon_buttons. Alternatively, can set handlers
            subseqeuntly, by example:
                icon_row['refresh'].on_click(callback)

        **kwargs:
            Passed to `w.Box`.
        """
        assert handlers is None or len(handlers) == len(icon_button_names)
        children = []
        for i, name in enumerate(icon_button_names):
            func = getattr(ButtonIcon, name)
            icon_button = func()
            if handlers is not None:
                icon_button.on_click(handlers[i])
            children.append(icon_button)

        kwargs.setdefault("layout", self.dflt_layout)
        super().__init__(children, names=icon_button_names, **kwargs)


class DateRangeSlider(w.Box):
    """Double handled slider to select date range.

    Instantiates `wBox` containing:
        *SelectionRangeSlider*

    Widget is exposed and many properties of the SelectionRangeSlider
    are directly accessible via the DateRangeSlider object.

    Attributes
    ----------
    slider: SelectionRangeWidget

    Notes
    -----
    This class originally included a 'magnet' icon which provided for
    moving the handles together so that that slider moved as a block.
    ipywidgets subseqeuntly included this functionality natively to the
    `SelectionRangeSlider` and so the functionality was stripped out of
    this class. This legacy is why the class inherits from w.Box (which
    also housed the 'magnet' icon button). Suspect that if wanted to the
    class could now be simplified by inheriting directly from the
    `SelectionRangeSlider`
    """

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        label_format: str | None = None,
        handle_color="skyblue",
        layout: dict | None = None,
        initial_selection: pd.Interval | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        dates
            Dates to comprise range from which interval can be selected.
                Index can be continuous or discontinuous.

        label_format
            Format in which to display selected dates.

        handle_color
            Handle color.

        layout
            Dictionary of kwargs from which to instantiate a Layout for the
            object (i.e. Box).

        initial_selection
            Initial interval to select. If not passed all dates will be
            selected.
        """
        self._passed_label_format = label_format
        self._style = w.SliderStyle(
            description_width="initial", handle_color=handle_color
        )

        slider_layout = w.Layout(width="100%")
        self.slider = w.SelectionRangeSlider(
            options=self._options(dates),
            description="Dates",
            style=self._style,
            continuous_update=True,
            layout=slider_layout,
        )
        if initial_selection is not None:
            self.interval = initial_selection
        else:
            self.select_all()

        layout = layout if layout is not None else {}
        layout.setdefault("display", "flex")
        layout.setdefault("flex_flow", "row")
        layout.setdefault("width", "100%")
        layout.setdefault("justify_content", "flex-start")
        super().__init__([self.slider], layout=w.Layout(**layout))

    def _label_format(self, dti: pd.DatetimeIndex):
        if self._passed_label_format is not None:
            fmt = self._passed_label_format
        else:
            if all(dti == dti.normalize()):
                fmt = "%y/%m/%d"
            elif dti[-1] - dti[0] > pd.Timedelta(days=28):
                fmt = "%m/%d/%H:%M"
            else:
                fmt = "%d/%H:%M"
        return " " + fmt + " "

    def _options(self, dates: pd.DatetimeIndex):
        values = dates
        labels = dates.strftime(self._label_format(dates))
        return list(zip(labels, values))

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates defining range from which interval can be selected."""
        labels, dates = zip(*self.slider.options)
        return pd.DatetimeIndex(dates)

    @dates.setter
    def dates(self, dates: pd.DatetimeIndex):
        self.slider.options = self._options(dates)
        self.slider.index = (0, len(dates) - 1)

    @property
    def index(self) -> tuple[int, int]:
        """Indices of start and end of selected interval."""
        return self.slider.index

    @index.setter
    def index(self, indices: tuple[int, int]):
        self.slider.index = indices

    @property
    def interval(self) -> pd.Interval:
        """Selected interval."""
        left, right = self.slider.value
        return pd.Interval(left, right, closed="both")

    @interval.setter
    def interval(self, interval: pd.Interval):
        dti = self.dates[pd.Series(self.dates).apply(lambda x: x in interval)]
        i_start = self.dates.get_loc(dti[0])
        i_end = self.dates.get_loc(dti[-1])
        self.index = (i_start, i_end)

    @property
    def extent(self) -> pd.Interval:
        """Interval covered by slider."""
        return pd.Interval(self.dates[0], self.dates[-1], closed="both")

    @property
    def selection(self) -> pd.DatetimeIndex:
        """All dates of selected range, inclusive of start and end."""
        indices = self.slider.index
        dates = self.dates[indices[0] : indices[1] + 1]
        return pd.DatetimeIndex(dates)

    def select_all(self):
        """Select all date range."""
        self.index = (0, len(self.dates) - 1)

    @property
    def all_dates_selected(self) -> bool:
        """True if all dates are currently selected."""
        return len(self.selection) == len(self.dates)

    @property
    def handle_color(self) -> str:
        return self._style.handle_color

    @handle_color.setter
    def handle_color(self, color: str):
        self._style.handle_color = color
