"""GUIs to visualise and analyse price data.

Classes
-------
Base(metaclass=ABCMeta):
    ABC defining implementation and base functionality.

DateSliderMixin():
    Include wu.DateRangeSlider to interactively select plot dates.

BasePrice(Base):
    Base implementation for GUIs incorporating a price chart.

ChartLine(Base):
    Line price chart GUI for single financial instrument.

ChartMultLine(Base):
    Line price chart GUI for comparing multiple financial instruments.

ChartOHLC(Base):
    OHLC chart GUI for single financial instrument.

PctChg(Base, DateSliderMixin):
    Bar chart GUI of precentage changes of single instrument.

PctChgMult(PctChg):
    Bar chart GUI of precentage changes of multiple instruments.

Notes
-----
Module might benefit from a more compositional approach, as opposed to
hierarchical, to creating the gui interfaces. For example:

    Selector
        To select periods.

    Crosshairs
        To administer crosshairs.

    TabsControl
        UI to create crosshairs and undertake analysis.

    DataSlider
        To scross through plot dates.

Before diving in making changes, would need to have a look at the current
implementation and get clear how to best implement a compositional
approach.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal

import bqplot as bq
from bqplot.interacts import Selector
import exchange_calendars as xcals
import IPython
import ipyvuetify as v
import ipywidgets as w
import market_prices as mp
from market_prices.intervals import TDInterval, PTInterval, to_ptinterval, ONE_DAY
import pandas as pd

from market_analy import charts, gui_parts, analysis as ma_analysis
from market_analy.utils.bq_utils import Crosshairs, FastIntervalSelectorDD
from market_analy.utils.dict_utils import set_kwargs_from_dflt
import market_analy.utils.ipyvuetify_utils as vu
import market_analy.utils.ipywidgets_utils as wu
import market_analy.utils.pandas_utils as upd

from .cases import CasesSupportsChartAnaly, CaseSupportsChartAnaly

# CONSTANTS
HIGHLIGHT_COLOR = "lightyellow"


class Base(metaclass=ABCMeta):
    """ABC for creating GUI.

    ABC defines:
        Implementation to create GUI
        Functionality via Mixin methods

    Notes
    -----
    Subclasses must implement the following abstract properties and methods:

    **Abstact Properties**

        ChartCls:
            charts module class, for example charts.Line.

        SelectorCls -> type[Selector] | None:
            Selector class or None if selector not required.

        _gui_box_contents -> list[w.Widget]:
            List of widgets that comprise gui.

    **Abstact Methods**

        _create_gui_parts(self):
            Method must be extended* to define all gui parts, except chart,
            with each part assigned to an instance attribute that can be
            included to `_gui_box_contents`.
            * Extension must execute method as defined on this abstract
            base class.

    Subclasses can optionally extend the following methods:

        _create_selector():
            Subclass can extend by way of passing through kwargs which will
            in turn be passed to `SelectorCls`.

        _create_gui_box():
            Subclass can extend in order to pass through kwargs which will
            in turn be passed to box `Layout` constructor.

    Subclasses should NOT override or extend the following methods:

        _create_gui()

    It is not intended that subclasses override or extend the following
        methods:

        _create_chart()

    Mixin Properties
    ----------------
    slctd_interval -> pd.Interval | None:
        Selected interval.

    Mixin Methods
    -------------
    close():
        Close all widgets that comprise gui.

    delete():
        Close and delete all widgets that comprise gui.

    display():
        Display gui.

    _show_loading_overlay():
        Show overlay indicating app is loading.

    _hide_loading_overlay():
        Hide overlay indicating app is loading.

    _popup():
        Overlay application and display popup dialog.

    _close_popup():
        Hide any popup dialog.

    _resize_chart():
        Stretch chart to fill all available horizontal space.

    _update_chart():
        Update chart with new data.

    _cycle_legend_handler():
        Handle a request to cycle chart legend to next position.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None = None,
        display=True,
        **kwargs,
    ):
        """Create GUI.

        Parameters
        ----------
        data
            Data to be passed to constructor of chart class returned by
            `ChartCls`.

        title
            Chart title.

        display
            True to display created GUI.

        **kwargs
            All excess kwargs will be passed to constructor of chart class
            (i.e. class as returned by `ChartCls`).
        """
        self.chart = self._create_chart(data, title=title, **kwargs)
        # Attributes set by --_create_gui--.
        # NB Furthertype[charts.Base] attributes will be set by subclass via _create_gui_parts
        self._widgets: list[w.Widget]
        self.slctr: Selector
        self._gui: w.Box

        self._create_gui()

        if display:
            self.display()

    @property
    @abstractmethod
    def ChartCls(self):
        """A class of charts module. For example, charts.Line

        Notes
        -----
        Instance of this class will instantiated by `_create_mark`.
        """

    def _create_chart(
        self, data: pd.DataFrame | pd.Series, title: str | None, **kwargs
    ) -> charts.Base:
        """Create chart.

        Notes
        -----
        Not intended that subclass extends.

        `kwargs` receive kwargs as collected by constructor.
        """
        c = self.ChartCls(data, title=title, display=False, **kwargs)
        c.figure.layout.margin = "-20px -10px -30px 0"
        return c

    @property
    @abstractmethod
    def SelectorCls(self) -> type[Selector] | None:
        """Selector class.

        Notes
        -----
        Subclass must implement. Return None if selector not required."""
        return None

    @property
    def slctd_interval(self) -> pd.Interval | None:
        """Selected interval. None if no selector or no selection."""
        if self.SelectorCls is None or self.slctr.interval is None:
            return None
        else:
            left = self.slctr.interval[0]
            right_tick = self.slctr.interval[1]
            last_interval = self.chart.date_interval_containing(right_tick)
            return pd.Interval(left, last_interval.right, closed="left")

    def _create_selector(self, **kwargs) -> type[Selector] | None:
        """Create selector.

        Notes
        -----
        Subclass can optionally extend to pass through +**kwargs+."""
        if self.SelectorCls is None:
            return None
        kwargs.setdefault("color", "crimson")
        kwargs.setdefault("enable", False)
        kwargs["marks"] = [self.chart.mark]
        kwargs["scale"] = self.chart.scales["x"]
        kwargs["figure"] = self.chart.figure
        return self.SelectorCls(**kwargs)

    def _create_gui(self):
        """Create gui.

        Notes
        -----
        Subclass should not overwrite or extend this method."""
        self._widgets = []
        w.Widget.on_widget_constructed(lambda w: self._widgets.append(w))
        self.slctr: Selector = self._create_selector()
        self._create_gui_parts()
        self._gui: v.App = self._create_gui_box()
        w.Widget.on_widget_constructed(None)

    def _show_loading_overlay(self):
        """Show loading overlay over gui box."""
        self._loading_overlay.value = True

    def _hide_loading_overlay(self):
        """Hide loading overlay over gui box."""
        self._loading_overlay.value = False

    def _resize_chart(self):
        """Stretch chart to fill all available horizontal space."""
        old = self.chart.figure.layout.align_self
        new = "stretch" if old in [None, "flex-end"] else "flex-end"
        self.chart.figure.layout.align_self = new

    def _popup(self, title: str | None = None, text: str | None = None):
        """Overlay application and display popup dialog."""
        self._dialog.popup(title, text)

    def _close_popup(self):
        """Hide any popup dialog."""
        self._dialog.close_dialog()

    @abstractmethod
    def _create_gui_parts(self):
        """Create gui parts.

        Notes
        -----
        Subclass must extend to define all parts to contribute to gui,
        assigning each to a suitable instance attribute. Subclass must
        also execute method as defined on this abstract class.
        """
        self._loading_overlay: v.Overlay = gui_parts.loading_overlay()
        self._dialog: v.Overlay = gui_parts.Dialog()

    @property
    @abstractmethod
    def _gui_box_contents(self) -> list[w.Widget]:
        """Gui box contents

        Notes
        -----
        Subclass must implement to return list of widgets that comprise
        gui, in insertion order, ordered top to bottom. Elements should
        include `self.chart.figure`.
        """

    def _create_gui_box(self, **kwargs) -> v.App:
        """Create gui box.

        Notes
        -----
        Subclass may extend to pass through kwargs for Layout."""
        base_contents = [self._dialog, self._loading_overlay]
        contents = self._gui_box_contents + base_contents
        dflt_kwargs = {
            "d_flex": True,
            "column": True,
            "justify_center": True,
            "align_stretch": True,
            "class_": "pa-3 " + gui_parts.BG,
        }
        kwargs = set_kwargs_from_dflt(kwargs, dflt_kwargs)
        return v.App(children=contents, **kwargs)

    # MIXIN METHODS

    def close(self):
        """Closes all gui widgets."""
        self.chart.close()
        for widget in self._widgets:
            widget.close()

    def delete(self):
        """Delete all gui widgets."""
        self.close()
        self.chart.delete()
        for _ in range(len(self._widgets)):
            del self._widgets[0]

    def display(self):
        """Display gui."""
        IPython.display.display(self._gui)

    def _update_chart(self, data: pd.DataFrame | pd.Series, title: str | None = None):
        """Update chart with new data."""
        self.chart.update(data, title)

    def _cycle_legend_handler(
        self,
        widget: w.Widget | None = None,
        event: str | None = None,
        data: dict | None = None,
    ):
        """Handle a request to cycle chart legend to next position."""
        self.chart._cycle_legend()


class BaseVariableDates(Base):
    """Interactively select plot dates via `wu.DateRangeSlider`.

    Implementation
    --------------
    `ChartCls` must inherit from `charts.BaseSubsetDD`

    To include date slider to gui:
        include following line to `_create_gui_parts` method:
            `self.date_slider: wu.DateRangeSlider = self._create_date_slider()`
        include self.data_slider within return of `_gui_box_contents()`.

    Added properties
    ----------------
    max_ticks -> int | None
        Maximum number of x-axis ticks that will be shown by default.

    _tick_interval():
        Duration covered by each chart x-tick.

    Methods of note
    ---------------
    _set_slider_limits_to_plotted_x_ticks:
        Set date slider limits to extents of currently plotted x_ticks.

    _max_x_ticks:
        Set slider to show all plottable x_ticks.

    _create_icon_row_top:
        Create top icon row for charts with variable dates.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None = None,
        display=True,
        max_ticks: int | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        data
            Data to be passed to constructor of chart class returned by
            `ChartCls`.

        title
            Chart title.

        max_ticks
            Maximum number of x-axis ticks that will shown by default
            (client can choose to show more via slider). None for no limit.
            (passed through to chart class).

        display
            True to display created GUI.

        **kwargs
            All excess kwargs will be passed to constructor of chart class
            (i.e. class as returned by `ChartCls`).
        """
        assert issubclass(self.ChartCls, charts.BaseSubsetDD)
        super().__init__(data, title, display, max_ticks=max_ticks, **kwargs)
        self.date_slider: wu.DateRangeSlider

    @property
    def _tick_interval(self) -> pd.Timedelta:
        """Duration covered by each x-axis timestamp."""
        return self.chart.tick_interval

    def _create_date_slider(self, **kwargs) -> wu.DateRangeSlider:
        """Create date slider.

        Notes
        -----
        Subclass can customise `wu.DateRangeSlider` by passing through
        **kwargs.
        """
        kwargs.setdefault("handle_color", HIGHLIGHT_COLOR)
        kwargs.setdefault("layout", {"margin": "0 0 15px 0"})
        ds = wu.DateRangeSlider(dates=self.chart.x_ticks, **kwargs)
        ds.interval = self.chart.plotted_interval
        ds.slider.observe(self._set_chart_x_ticks_to_slider, ["index"])
        return ds

    @property
    def _date_slider_extent(self) -> pd.Interval:
        """Interval covered by date slider.

        NB unlike `_slider.extent` includes duration of interval covered
        by last tick.
        """
        ticks_extent = self.date_slider.extent
        right = self.chart.date_interval_containing(ticks_extent.right).right
        return pd.Interval(ticks_extent.left, right, closed="left")

    @property
    def _slider(self):
        return self.date_slider.slider

    def _set_chart_x_ticks_to_slider(self, event=None):
        self.chart.plotted_x_ticks = self.date_slider.interval

    @contextmanager
    def _handler_disabled(self):
        """Undertake operation within context of disabled handler.

        Undertake an operation with context of
        `_set_chart_x_ticks_to_slider` handler being disabled.
        """
        self._slider.unobserve(self._set_chart_x_ticks_to_slider, ["index"])
        yield
        self._slider.observe(self._set_chart_x_ticks_to_slider, ["index"])

    def _set_slider_limits_to_plotted_x_ticks(self):
        """Set date slider extents to reflect currently plotted x_ticks."""
        with self._handler_disabled():
            self.date_slider.dates = self.chart.plotted_x_ticks

    def _set_slider_limits_to_all_plottable_x_ticks(self):
        """Set date slider extents to reflect all plottable x_ticks."""
        with self._handler_disabled():
            self.date_slider.dates = self.chart.x_ticks

    def _update_slider_dates(self, price_interval: pd.Timedelta):
        """Update slider dates to reflect chart x_ticks. Extents unchanged.

        price_interval
            Interval by which to extend extent beyond final timestamp, to
            cover all period represented by the final timestamp.
        """
        extent = self.date_slider.extent
        extent = pd.Interval(extent.left, extent.right + price_interval, closed="left")
        ticks = self.chart.x_ticks_subset(extent)
        with self._handler_disabled():
            self.date_slider.dates = ticks

    def _set_slider_to_plotted_x_ticks(self):
        """Set slider handles to plotted x_ticks."""
        with self._handler_disabled():
            self.date_slider.interval = self.chart.plotted_interval

    # UPDATE CHART
    def _max_x_ticks(self):
        """Show data for all plottable x_ticks."""
        self.chart.reset_x_ticks()
        self._set_slider_limits_to_all_plottable_x_ticks()

    def _update_chart(
        self,
        data: pd.DataFrame | pd.Series,
        title: str | None = None,
        visible_x_ticks: pd.Interval | None = None,
        reset_slider=False,
    ):
        """Update chart with new data.

        Overrides inherited method.

        reset_slider : bool, default: False
            True to set slider extents to reflect all plottable x-ticks.
            False to retain existing extents to extent possible.
        """
        prior_tick_interval = self._tick_interval
        self.chart.update(data, title, visible_x_ticks)
        if reset_slider:
            self._set_slider_limits_to_all_plottable_x_ticks()
        else:
            self._update_slider_dates(prior_tick_interval)
        self._set_slider_to_plotted_x_ticks()

    # TOP ICON ROW
    @property
    def _icon_row_top_handlers(self) -> list[Callable]:
        return [self._max_x_ticks, self._resize_chart, self.close]

    def _icon_row_top_handler_funnel(self, widget):
        """Funnel for events triggered by clicking an icon row button."""
        handlers = self._icon_row_top_handlers
        i = self._icon_row_top.children.index(widget)
        handlers[i]()

    def _create_icon_row_top(self) -> gui_parts.IconRowTop:
        num_handlers = len(self._icon_row_top_handlers)
        funnel = [self._icon_row_top_handler_funnel] * num_handlers
        return gui_parts.IconRowTop(funnel)


class BasePrice(BaseVariableDates):
    """Base implementation for GUIs incorporating a price chart.

    GUI comprises:

    IconRow: Icons to undertake chart operations.
    w.ToggleButtons: Toggle buttons to select interval (optional)
    Figure: plot
    wu.DateRangeSlider: slider to interactively set chart dates
    HtmlOutput: Housing for html output.

    TabsControl: cursor and selector tabs.
        Cursor tab:
            + and - toggle buttons to add or delete crosshairs to the plot.
            lightbulb toggle button to make all crosshairs fully opaque or
                semi-transparent.
            trash button to delete all crosshairs.

        Selector tab. When selected, clicking chart produces a selector
        which end-user can use to define a selection of dates. Tab
        includes:
            zoom button to zoom to current selection, or currently displayed
                dates if no selection made.
            up / down arrows to evaluate maximum advance / decline over
                selected period, or period comprising all plotted dates if
                no selection made.

    Implementation
    --------------
    BaseVariableDates is substantially implemented by the base class.
    Subclasses are left with implementing the following properties:
        ChartCls -> type[charts.BasePrice]

        _chart_title -> str

    In addition, subclasses can optionally extend:

        Inherited methods as described by Base documentation.

        The following class attributes introduced by this class:
            _HAS_INTERVAL_SELECTOR : bool, default: True
                Set to False on subclass to not include the
                `w.ToggleButtons` that provides for changing tick interval.

        The following properties introduced by this class:

            _prices_kwargs -> dict:
                Immutable chart kwargs. Subclass should extend to override
                defaults and/or add immutable kwargs specific to
                `ChartCls`.

    Methods
    -------
    add_crosshair():
        Add crosshair to figure.

    Properties
    ----------
    crosshairs: list
        Crosshair objects associated with figure.
    """

    TICK_INTERVALS = [
        "3M",
        "1M",
        "5D",
        "1D",
        "4H",
        "1H",
        "30T",
        "15T",
        "5T",
        "2T",
        "1T",
    ]
    TICK_INTERVALS_PT = [to_ptinterval(ti) for ti in TICK_INTERVALS]
    _HAS_INTERVAL_SELECTOR = True

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval | None = None,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        chart_kwargs: dict | None = None,
        **kwargs,
    ):
        """Create GUI.

        Parameters
        ----------
        analysis
            Analysis instance representing instruments to be plotted.

        interval
            Interval covered by an x-axis tick, as 'interval` parameter
            described by `help(analysis.prices.get)`.

        max_ticks
            Maximum number of x-axis ticks that will shown by default (client
            can choose to show more via slider). None for no limit.

        log_scale
            True to plot prices against a log scale. False to plot prices
            against a linear scale.

        display
            True to display created GUI.

        chart_kwargs
            Any kwargs to pass on to the chart class.

        **kwargs
            Period for which to plot prices. Passed as period parameters as
            described by market-prices documentation for 'PricesCls.get'
            method where 'PricesCls' is the class that was passed to
            'PricesCls' parameter of `mkt_anlaysis.Analysis` to intantiate
            `analysis`.
        """
        assert issubclass(self.ChartCls, charts.BasePrice)
        chart_kwargs = {} if chart_kwargs is None else chart_kwargs
        ptinterval = None if interval is None else to_ptinterval(interval)
        self._initial_prices: pd.DataFrame | pd.Series  # set by _set_initial_prices
        self._initial_price_params_non_period_: dict  # set via _set_initial_prices
        prices = self._set_initial_prices(analysis, ptinterval, kwargs)
        self._analysis = analysis
        super().__init__(
            data=prices,
            title=self._chart_title,
            max_ticks=max_ticks,
            display=display,
            log_scale=log_scale,
            **chart_kwargs,
        )

    @property
    def SelectorCls(self) -> type[FastIntervalSelectorDD]:
        return FastIntervalSelectorDD

    @property
    def _chart_title(self) -> str:
        """Chart title.

        Notes
        -----
        Implement on subclass to return chart title.
        """
        raise NotImplementedError()

    @property
    def _prices_kwargs(self) -> dict:
        """Fixed kwargs `market_prices.PricesBase.get`.

        Subclass can extend to override any default values or/and extend to
        introduce further immutable kwargs specific to how `ChartCls`
        requires price data.
        """
        return {}

    def _set_initial_prices(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval | None,
        period_parameters: dict,
    ) -> pd.DataFrame | pd.Series:
        """Set initial prices data.

        Parameters
        ----------
        analysis
            'analysis' parameter as received by constructor.
        interval
            'interval' parameter as received by constructor.

        period_parameters
            **kwargs receieved by constructor.

        Notes
        -----
        Initial prices are evaluated once, when this method is first
        called. Subsequent calls will return a copy of the initial prices.

        The method is implemented in this way in order that prices can be
        requested by a base class earlier than they would otherwise be
        called by this `BasePrice` class. For example, price data may be
        required by a subclass to undertake analysis, such as evaluting
        trends or positions, which is then be displayed visually over a
        chart of the underlying data.
        """
        if hasattr(self, "_initial_prices"):
            return self._initial_prices.copy()

        params = period_parameters.copy()
        params["composite"] = False
        params["interval"] = interval
        prices = analysis.prices.get(**params, **self._prices_kwargs)
        self._initial_prices = prices
        self._set_initial_price_params_non_period(params)
        return self._initial_prices.copy()

    def _set_initial_price_params_non_period(self, initial_price_params: dict):
        """Set initial price params excluding those that define period and interval."""
        exclude = [
            "interval",
            "start",
            "end",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
        ]
        params = {k: v for k, v in initial_price_params.items() if k not in exclude}
        self._initial_price_params_non_period_ = params

    @property
    def _initial_price_params_non_period(self) -> dict:
        return self._initial_price_params_non_period_

    def _get_prices(self, **kwargs):
        """Get prices from analysis."""
        return self._analysis.prices.get(**kwargs, **self._prices_kwargs)

    # TOP ICON ROW
    @property
    def _icon_row_top_handlers(self) -> list[Callable]:
        return [self._max_x_ticks, self._reset, self._resize_chart, self.close]

    # INTERVAL SELECTOR
    def _interval_selector_handler(self, change):
        self._show_loading_overlay()
        try:
            self._change_tick_interval(change["old"], change["new"])
        except ValueError as err:
            self._hide_loading_overlay()
            self._popup("Interval unavailable", err.args[0])
            self._interval_selector.set_value_unobserved(change["old"])
        else:
            self._hide_loading_overlay()

    def _create_intrvl_slctr(self) -> w.ToggleButtons:
        left = self.chart.plottable_interval.left
        right = self.chart.plottable_interval.right
        intervals = zip(self.TICK_INTERVALS, self.TICK_INTERVALS_PT)
        labels = []
        for label, pt in intervals:
            to = (
                right if isinstance(pt, TDInterval) else pt.as_offset_ms.rollback(right)
            )
            if left + pt <= to:
                labels.append(label)
        return gui_parts.IntervalSelector(
            labels, self._tick_interval, self._interval_selector_handler
        )

    # TAB CONTROL
    def _cursor_tab_reset(self):
        self.slctr.disable()

    def _selector_tab_reset(self):
        self.slctr.enable()

    def _tabs_handler(self, widget, event, data):
        if not data:
            self._cursor_tab_reset()
        elif data == 1:
            self._selector_tab_reset()

    def _trash_handler(self, widget, event, data):
        self._close_crosshairs()
        self.html_output.clear()

    def _create_tabs_control(self):
        tc = gui_parts.TabsControl()
        tc.but_lightbulb.handler_on_selecting = self.crosshairs.opaque
        tc.but_lightbulb.handler_on_deselecting = self.crosshairs.fade
        tc.but_trash.on_event("click", self._trash_handler)
        tc.but_zoom.on_event("click", self._zoom_handler)
        tc.but_arrow_up.on_event("click", self._max_adv_handler)
        tc.but_arrow_down.on_event("click", self._max_dec_handler)
        tc.on_event("change", self._tabs_handler)
        return tc

    # HTML OUTPUT
    def _create_html_output(self):
        return gui_parts.HtmlOutput()

    @property
    def html_output(self) -> gui_parts.HtmlOutput:
        return self._html_output

    # GUI BOX
    def _create_gui_parts(self):
        super()._create_gui_parts()
        self._crosshairs = Crosshairs(self.chart.figure)
        self._set_mark_handlers()
        self._icon_row_top: gui_parts.IconRowTop = self._create_icon_row_top()

        if (
            not self._HAS_INTERVAL_SELECTOR
            or self.chart.tick_interval not in self.TICK_INTERVALS_PT
        ):
            self._interval_selector: w.ToggleButtons | None = None
        else:
            self._interval_selector = self._create_intrvl_slctr()

        self.date_slider: wu.DateRangeSlider = self._create_date_slider()
        self._html_output: w.HTML = self._create_html_output()
        self.tabs_control: gui_parts.TabsControl = self._create_tabs_control()

    @property
    def _gui_box_contents(self) -> list[w.Widget]:
        contents = [self._icon_row_top]
        if self._interval_selector is not None:
            contents += [self._interval_selector]
        contents += [
            self.chart.figure,
            self.date_slider,
            self.tabs_control,
            self.html_output,
        ]
        return contents

    # CROSSHAIRS
    @property
    def crosshairs(self) -> Crosshairs:
        return self._crosshairs

    def _close_crosshairs(self):
        self.crosshairs.close()
        self.tabs_control.cursor_toggle.deselect()

    def _ch_delete_onclick_handler(self, crosshair):
        if self.tabs_control.cursor_toggle.selected == "minus":
            crosshair.close()
            if not self.crosshairs:
                self.tabs_control.cursor_toggle.deselect()

    def _ch_standout_onclick_handler(self, crosshair):
        if self.tabs_control.cursor_toggle.selected != "minus":
            self.crosshairs.fade()
            crosshair.opacity = 1.0

    def _ch_chgside_onclick_handler(self, crosshair):
        crosshair.change_side()

    def add_crosshair(
        self,
        draggable: bool = True,
        delible: bool = True,
        standout: bool = True,
        chgside: bool = True,
        **kwargs,
    ) -> Crosshairs.CrosshairsCh:
        """Add crosshair to plot.

        draggable
            True to allow end-sure to drag crosshairs via the cross.

        delible
            True to allow end-user to delete crosshair (in accordance with
            `_ch_delete_onclick_handler`)

        standout
            True to, on-click, make full opaque and all other crosshairs
            semi-transparent.

        chgside: bool
            True to, on-click, change side of crosshair's labels.

        **kwargs:
            passed to Crosshair.
        """
        ch = self.crosshairs.add(draggable=draggable, **kwargs)
        reg = ch.cross.on_element_click
        if delible:
            reg(lambda _, _e: self._ch_delete_onclick_handler(ch))
        if standout:
            reg(lambda _, _e: self._ch_standout_onclick_handler(ch))
        if chgside:
            reg(lambda _, _e: self._ch_chgside_onclick_handler(ch))
        return ch

    def _add_crosshair_handler(self, mark, event):
        if self.tabs_control.cursor_toggle.selected == "plus":
            self.add_crosshair()

    def _set_mark_handlers(self):
        self.chart.mark.on_background_click(self._add_crosshair_handler)

    # ZOOM and UNZOOM
    def _zoom_to_selection(self):
        """Zoom to selection

        If no selection then zooms to x_ticks currently shown on plot.
        """
        if self.slctr.has_selection:
            self.chart.plotted_x_ticks = self.slctd_interval
        self._set_slider_limits_to_plotted_x_ticks()

    def _zoom_handler(self, widget, event, data):
        self._zoom_to_selection()
        self.slctr.reset()

    @property
    def _operation_interval(self):
        """Interval over which operation should act.

        User-selected interval or currently plotted x_ticks otherwise.
        """
        if self.slctd_interval is not None:
            return self.slctd_interval
        else:
            return self.chart.plotted_interval

    # MAX ADV/DEC
    def _get_max_chg(
        self, direction: Literal["max_adv", "max_dec"], style=False, **kwargs
    ):
        interval = self._operation_interval
        method = getattr(self._analysis, direction)
        return method(
            interval=self._tick_interval,
            start=interval.left,
            end=interval.right,
            style=style,
            **kwargs,
        )

    def _max_chg_html(self, method_name: Literal["max_adv", "max_dec"]):
        styler = self._get_max_chg(method_name, style=True)
        self.html_output.display(styler.to_html(), add_padding=4)

    def _max_chg_crosshair(
        self,
        x,
        y,
        color,
        side: Literal["lesser", "greater"],
        y_offset: int | None = None,
    ):
        label_kwargs = {"y_offset": y_offset} if y_offset is not None else {}
        self.add_crosshair(
            x=x,
            y=y,
            color=color,
            draggable=False,
            line_kwargs={"side": side},
            label_kwargs=label_kwargs,
        )

    def _max_chg_crosshairs(
        self,
        row: pd.Series,
        advance: bool,
        colors: tuple[str, str],
        y_offset: int | None = None,
    ):
        """Add crosshairs at either extreme of a defined movement.

        Adds crosshairs at the start and end of a movement represented by a
        `row` of a `pd.DataFrame` returned by `Analysis.max_adv` or
        `Analysis.max_dec`.

        Parameters
        ----------
        advance
            True if movement represents a maximum advance
            False if movements represents a maximum decline.
        """
        start = row.start if row.start.tz is None else row.start.tz_localize(None)
        y = row.low if advance else row.high
        self._max_chg_crosshair(
            x=start, y=y, color=colors[0], side="lesser", y_offset=y_offset
        )
        end = row.end if row.end.tz is None else row.end.tz_localize(None)
        y = row.high if advance else row.low
        self._max_chg_crosshair(
            x=end, y=y, color=colors[1], side="greater", y_offset=y_offset
        )
        # ugly but necessary. When the crosshairs are drawn
        # figure.interaction is set to None (don't know why). The selector is
        # thereby undrawn although retains its selection. This line ensures that
        # 1) selection reflects what's seen on screen (i.e. that there isn't a
        # selection) and 2) that a new selector remains available to the user
        # should they click on the figure.
        # Might prefer to 'fix' this by redrawing the selector after the
        # crosshairs have been added. I originally tried to decorate the
        # _max_adv and _max_dec with a context manager which stored the
        # value of .selected, forced slctr.reset() and after adding the
        # crossharirs reassigning the stored selection to .selected. However,
        # reassigning FastIntervalSelector.selected does not trigger a redrawing
        # of the  selector rectangle to reflect the newly assigned selection.
        # with the exception of assigning None, which removes is (NB noted to
        # bqplot/issues.ipynb).
        self.slctr.reset()

    def _max_adv_crosshairs(
        self, colors=("olivedrab", "limegreen"), y_offset: int | None = None
    ):
        """Add crosshairs at either extreme of a defined advance.

        Adds crosshairs at the start and end of a movement represented by a
        `row` of a `pd.DataFrame` returned by `Analysis.max_adv`.
        """
        row = self._get_max_chg("max_adv").iloc[0]
        self._max_chg_crosshairs(
            row=row, advance=True, colors=colors, y_offset=y_offset
        )

    def _max_dec_crosshairs(
        self, colors=("orangered", "crimson"), y_offset: int | None = -25
    ):
        """Add crosshairs at either extreme of a defined decline.

        Adds crosshairs at the start and end of a movement represented by a
        `row` of a `pd.DataFrame` returned by `Analysis.max_dec`.
        """
        row = self._get_max_chg("max_dec").iloc[0]
        self._max_chg_crosshairs(
            row=row, advance=False, colors=colors, y_offset=y_offset
        )

    def _is_max_chg_available(self, method_name: Literal["max_adv", "max_dec"]) -> bool:
        if self.chart.tick_interval > ONE_DAY:
            msg = (
                f"The {method_name} function is unavailable for intervals > 1 day"
                " as the underlying data does not provide for sufficient precision."
            )
            self._popup("Function unavailable", msg)
            return False
        return True

    def _max_adv_handler(self, widget, event, data):
        if not self._is_max_chg_available("max_adv"):
            return
        self._max_chg_html("max_adv")
        self._max_adv_crosshairs()

    def _max_dec_handler(self, widget, event, data):
        if not self._is_max_chg_available("max_dec"):
            return
        self._max_chg_html("max_dec")
        self._max_dec_crosshairs()

    # UPDATE CHART
    def _clear_user_activity(self):
        self._close_crosshairs()
        self.html_output.clear()

    def _verify_tick_interval_valid(
        self,
        old: PTInterval,
        new: PTInterval,
        plotted_interval: pd.IntervalIndex,
        cal: xcals.ExchangeCalendar,
    ):
        """Raise ValueError if new interval greater than currently plotted range.

        Raises `ValueError` if either side of one interval at `new` would
        lie outside of `plotted_interval`.
        """
        ERROR_MSG = (
            f"Interval '{new.as_pdfreq}' unavailable: the currently plotted"
            f" range is shorther than the requested interval."
        )
        left, right = plotted_interval.left, plotted_interval.right
        if new.is_intraday:
            if old.is_intraday and (right - left < new):
                raise ValueError(ERROR_MSG)
            return  # if new intraday and old > intraday then all ok

        if old.is_intraday:
            if left in cal.opens.values:
                left = cal.minute_to_session(left)
            else:
                left = cal.minute_to_future_session(left, 1)
            right = cal.minute_to_past_session(right, 1)

        if new.is_monthly:
            right = new.as_offset_ms.rollback(right)
            if right - new < left:
                raise ValueError(ERROR_MSG)
            return

        assert new.is_daily
        if not old.is_monthly:
            if old.is_daily:
                right -= ONE_DAY  # `plotted_interval.right` is last session + one day
            if cal.session_offset(left, new.days - 1) > right:
                raise ValueError(ERROR_MSG)

    def _change_tick_interval(self, old: PTInterval, new: PTInterval):
        """Change interval of price data.

        old
            Existing price interval.

        new
            Price interval to change the chart to.
        """
        cal = self._analysis.prices.calendar_default
        self._verify_tick_interval_valid(old, new, self.chart.plotted_interval, cal)

        # evaluate new price data
        start = self.chart.plottable_interval.left
        if not new.is_intraday:
            bi = self._analysis.prices.bi_daily
        else:
            for bi in reversed(self._analysis.prices.bis_intraday):
                if new == bi or not new % bi:
                    break
        date_limit = self._analysis.prices.limits_sessions[bi][0]
        if date_limit is not None and date_limit > start:
            msg = (
                f"Prices for interval '{new.as_pdfreq}' are only available from"
                f" '{mp.helpers.fts(date_limit)}' although the earliest date that"
                f" can be plotted on the chart implies that require data"
                f" from '{mp.helpers.fts(start)}'."
            )
            raise ValueError(msg)

        end = self.chart.plottable_interval.right
        if old.is_daily:
            end -= ONE_DAY
        interval = new.as_pdfreq[:-1] if new.is_monthly else new
        try:
            prices = self._get_prices(
                interval=interval,
                start=start,
                end=end,
                **self._initial_price_params_non_period,
            )
        except Exception as err:
            raise ValueError(str(err)) from None
        if prices.isna().any(axis=None):
            symbols = [col for col in prices.columns if prices[col].isna().any()]
            msg = (
                f"Prices for interval '{new.as_pdfreq}' are not available over the"
                f" current plottable dates as no price is available over this peroid"
                f" for the following symbols: {symbols}."
            )
            raise ValueError(msg)

        # evaluate visible_x_ticks range
        left = self.chart.plotted_interval.left
        right = self.chart.plotted_interval.right
        if old.is_intraday and not new.is_intraday:
            tz = self._analysis.prices.tz_default
            if left in cal.opens.values:
                left = cal.minute_to_session(left.tz_localize(tz))
            else:
                left = cal.minute_to_future_session(left.tz_localize(tz), 1)
            right = cal.minute_to_past_session(right.tz_localize(tz), 1) + ONE_DAY
        elif not old.is_intraday and new.is_intraday:
            right -= ONE_DAY  # plotted interval right is last session + one day
            if old.is_monthly:
                left = cal.date_to_session(left, "next")
                right = cal.date_to_session(right, "previous")
            left = cal.opens[left].tz_localize(None)
            right = cal.closes[right].tz_localize(None)
        elif old.is_intraday and new.is_intraday and new > old:
            # dont' show any data to the right of the right of the last shown bar
            right -= new

        vxts = pd.Interval(left, right, closed="left")

        self._clear_user_activity()
        self._update_chart(prices, reset_slider=True, visible_x_ticks=vxts)

    def _reset_chart(self):
        """Reset initial chart."""
        prices = self._initial_prices.copy()
        if isinstance(prices.index, pd.IntervalIndex):
            left = prices.index[0].left.tz_localize(None)
            right = prices.index[-1].right.tz_localize(None)
        else:
            left, right = prices.index[0], prices.index[-1] + ONE_DAY
        vxts = pd.Interval(left, right, closed="left")
        self._update_chart(prices, visible_x_ticks=vxts, reset_slider=True)
        self._interval_selector.set_value_unobserved(self._tick_interval)

    def _reset(self):
        self.slctr.disable()
        self._clear_user_activity()
        self.tabs_control.reset()
        self._reset_chart()


class GuiLine(BasePrice):
    """GUI to display and interact with a Line Chart."""

    @property
    def ChartCls(self) -> type[charts.Line]:
        return charts.Line

    @property
    def _chart_title(self) -> str:
        return self._analysis.symbol

    @property
    def _prices_kwargs(self) -> dict:
        d = super()._prices_kwargs
        d["lose_single_symbol"] = True
        d["close_only"] = True
        return d


class GuiMultLine(BasePrice):
    """GUI to display and interact with a Multiple Line Chart."""

    def __init__(
        self,
        analysis: ma_analysis.Compare,
        interval: mp.intervals.RowInterval | None = None,
        rebase_on_zoom: bool = True,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        **kwargs,
    ):
        """Create GUI.

        Parameters
        ----------
        analysis
            Compare object representing instruments to plot.

        interval
            Interval covered by an x-axis tick, as 'interval` parameter
            described by `help(analysis.prices.get)`.

        rebase_on_zoom
            if True will rebase prices following zoom.

        max_ticks
            Maximum number of x-axis ticks that will shown by default
            (client can choose to show more via slider). None for no limit.

        log_scale
            True to plot prices against a log scale. False to plot prices
            against a linear scale.

        display
            True to display created GUI.

        **kwargs
            Period for which to plot prices. Passed as period parameters as
            described by market-prices documentation for 'PricesCls.get'
            method where 'PricesCls' is the class that was passed to
            'PricesCls' parameter of `mkt_anlaysis.Compare` to intantiate
            `analysis`.
        """
        self._rebase_on_zoom = rebase_on_zoom
        self._labels: list[str]  # set by --_set_initial_prices--
        super().__init__(analysis, interval, max_ticks, log_scale, display, **kwargs)

    @property
    def ChartCls(self) -> type[charts.MultLine]:
        return charts.MultLine

    @property
    def _chart_title(self) -> str:
        return "Rebased comparison"

    @property
    def _prices_kwargs(self) -> dict:
        d = super()._prices_kwargs
        d["fill"] = "both"
        d["close_only"] = True
        return d

    def _set_initial_prices(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval | None,
        period_parameters: dict,
    ) -> pd.DataFrame | pd.Series:
        prices = super()._set_initial_prices(analysis, interval, period_parameters)
        self._labels = list(prices.columns)
        return prices

    def _create_rebase_button(self) -> vu.IconBut:
        but = gui_parts.rebase_but(class_="ml-5 mr-5")
        but.on_event("click", self.rebase_to_first_plotted_x_tick)
        return but

    def _create_legend_button(self) -> vu.IconBut:
        but = gui_parts.legend_but(class_=" ")  # class_='mr-5'
        but.on_event("click", self._cycle_legend_handler)
        return but

    def _create_controls_box(self) -> v.Layout:
        return v.Layout(
            children=[
                self._tabs_control_container,
                self._but_rebase.tt,
                self._but_legend.tt,
            ],
            d_flex=True,
            flex_row=True,
            align_center=True,
            justify_center=True,
        )

    def _create_gui_parts(self):
        super()._create_gui_parts()
        # As tabs_control serves as a container itself, in order to avoid the
        # tabs control growing within its parent container (_controls_box)
        # necessary to set class_ to 'flex-grow-0'. However this trick doesn't
        # work by including 'flex-grow-0' directly to tabs_control, rather
        # necessary to place it within its own container.
        self._tabs_control_container = v.Layout(
            children=[self.tabs_control], class_="flex-grow-0"
        )
        self._but_rebase: vu.IconBut = self._create_rebase_button()
        self._but_legend: vu.IconBut = self._create_legend_button()
        self._controls_box = w.HBox = self._create_controls_box()

    @property
    def _gui_box_contents(self) -> list[w.Widget]:
        """Contents of gui box.

        Notes
        -----
        Subclass extends method to replace tab control with controls box
        that includes legend cycle button.
        """
        contents = super()._gui_box_contents
        index = contents.index(self.tabs_control)
        contents[index] = self._controls_box
        return contents

    def _rebase_plot_prices(self, date: pd.Timestamp):
        """Rebase plot prices such that price 100 on a given `date`.

        Closes all crosshairs which would not relate to rebased prices and,
        for same reason, any html output.
        """
        self._crosshairs.close()
        self.html_output.clear()
        self.chart.rebase(date)

    def _zoom_to_selection(self):
        """Zoom to selection

        If no selection then to x_ticks currently shown on plot.
        """
        if self.slctr.has_selection:
            self.chart.plotted_x_ticks = self.slctd_interval
        else:
            self._set_chart_x_ticks_to_slider()
        self._set_slider_limits_to_plotted_x_ticks()
        if self._rebase_on_zoom:
            self._rebase_plot_prices(self.chart.plotted_interval.left)

    def rebase_to_first_plotted_x_tick(self, widget=None, event=None, data=None):
        self._rebase_plot_prices(self.chart.plotted_interval.left)

    def _get_max_chg(
        self, direction: Literal["max_adv", "max_dec"], style=False, **kwargs
    ):
        """Get max change.

        Notes
        -----
        PITA

        Max change has to account for day high and low. Other classes
        simply call the associated `._analysis` method (max_adv or
        max_dec). This isn't an option for ChartMultLine as the plotted
        prices have been rebased and hence do not correspond with the
        non-rebased prices that `_analysis` get and uses to calculate the
        changes. Accordingly this method:
            Gets price data from analysis, to include high and low columns,
            and rebases prices so that the close columns reflect the
            plotted data and high and low columns are rebased
            proportionally (a check is whether the %age diff between the
            close and the high or low on any day of the rebased data is the
            same as the %age differences for the same day of the original
            data).

            Takes subset of these prices over the period to be operated on.

            Gets maximum change by passing this subset of the rebased data
            to `._analysis._max_chg_compare`.
        """
        interval = self._tick_interval
        prices = self._analysis.prices.get(
            interval=interval,
            include=self._labels,
            start=self.chart.x_ticks[0],
            end=self.chart.x_ticks[-1],
            fill="both",
        )
        if isinstance(prices.index, pd.IntervalIndex):
            prices.index = upd.interval_index_new_tz(prices.index, None)
        else:
            prices.index = mp.utils.pandas_utils.get_interval_index(prices.index, "1D")
        for symbol in self._labels:
            prices[symbol] = upd.rebase_to_cell(
                prices[symbol], row=self.chart.rebase_date, col="close"
            )
        prices = prices[prices.index.overlaps(self._operation_interval)]
        start = prices.index[0].left
        end = prices.index[-1].right
        if interval == ONE_DAY:
            prices.index = prices.index.left
            end = prices.index[-1]
        direction_ = "advance" if direction == "max_adv" else "decline"
        return self._analysis.max_chg_compare(
            direction_, prices, style, start=start, end=end
        )

    def _max_chg_crosshairs(self, advance: bool):
        """Add crosshairs at either extreme of a defined movement.

        Adds crosshairs at the start and end of a movement represented by
        the row of a `pd.DataFrame` returned by `Compare.max_chg_compare`.

        Parameters
        ----------
        advance
            True if movement represents a maximum advance
            False if it represents a maximum decline.
        """
        direction: Literal["max_adv", "max_dec"] = "max_adv" if advance else "max_dec"
        max_chgs = self._get_max_chg(direction)
        for i, label in enumerate(self.chart.mark.labels):
            offset = -10 if not i else -20
            color = self.chart.mark.colors[i]
            row = max_chgs.loc[label]
            super()._max_chg_crosshairs(
                row=row, advance=advance, colors=(color, color), y_offset=offset
            )

    def _max_adv_crosshairs(self):
        self._max_chg_crosshairs(advance=True)

    def _max_dec_crosshairs(self):
        self._max_chg_crosshairs(advance=False)

    def _legend_click_handler(self, mark, event):
        """Highlight line associated with legend label clicked.

        Also highlights any crosshairs of same color as that line - There
        is no check that crosshairs of the same color are associated in any
        way with the line.

        If line associated clicked legend label is already the only
        highlighted line then makes all lines opaque (i.e. toggle action).
        """
        index = event["data"]["index"]
        opacities = self.chart.mark.opacities
        if opacities and opacities[index] == 1 and opacities.count(1.0) == 1:
            self.chart.opaque()
        else:
            self.chart.highlight_line(index)
            self.crosshairs.highlight_color(self.chart.mark.colors[index])

    def _set_mark_handlers(self):
        super()._set_mark_handlers()
        self.chart.mark.on_legend_click(self._legend_click_handler)

    def _reset(self):
        super()._reset()
        self.chart.opaque()


class GuiOHLC(BasePrice):
    """GUI to display and interact with a OHLC Chart.

    Properties (in addition to inhertied)
    ----------
    last_selected -> pd.Series:
        Price data row corresponding with most recently clicked bar.
    """

    @property
    def ChartCls(self) -> type[charts.OHLC]:
        return charts.OHLC

    @property
    def _chart_title(self) -> str:
        return self._analysis.symbol

    @property
    def _prices_kwargs(self) -> dict:
        d = super()._prices_kwargs
        d["lose_single_symbol"] = True
        return d

    def _create_selector(self, **kwargs) -> type[Selector] | None:
        """Create selector.

        Notes
        -----
        Overrides inherited method to avoid passing mark. Seems that
        passing mark prevents selector even being able to register
        selection against scale.

        **kwargs parameter only to maintain compatibility with base class.
        """
        kw_args = {
            "color": "white",
            "enable": False,
            "scale": self.chart.scales["x"],
            "figure": self.chart.figure,
        }
        return self.SelectorCls(**kw_args)

    def _add_crosshair_handler(self, mark: bq.OHLC, event: dict):
        if self.tabs_control.cursor_toggle.selected != "plus":
            return
        index = event["data"]["index"]
        x = self.chart.mark.x[index]

        # NOTE suspect this part will be refactored out to somewhere more general.
        higher, lower = False, False
        i = 0
        data = self.chart.mark.y
        while higher and lower or (not higher and not lower):
            i += 1
            higher = data[index][1] > data[index - i][1]
            lower = data[index][2] < data[index - i][2]
            if i == 100:
                raise StopIteration("breaking infinite loop.")
        y = data[index][1] if higher else data[index][2]

        self.add_crosshair(x=x, y=y, existing_mark=mark)

    def _display_mark_data(self, mark: bq.OHLC, event: dict):
        s = self.chart._tooltip_value(mark, event)
        self.html_output.display(s)

    @property
    def last_selected(self) -> pd.Series:
        """Price data row corresponding with most recently clicked mark."""
        return self._last_selected

    def _selected_mark_handler(self, mark: bq.OHLC, event: dict):
        self._display_mark_data(mark, event)
        i = event["data"]["index"]
        self._last_selected = self.chart.data.iloc[i]

    def _mark_handler(self, mark: bq.OHLC, event: dict):
        if self.tabs_control.cursor_toggle.selected == "plus":
            self._add_crosshair_handler(mark, event)
        else:
            self._selected_mark_handler(mark, event)

    def _set_mark_handlers(self):
        self.chart.mark.on_element_click(self._mark_handler)


class GuiPctChg(BaseVariableDates):
    """GUI for bar chart showing precentage changes of single instrument.

    GUI comprises:

        Close button: Icon to close gui.
        Figure: chart
        wu.DateRangeSlider: If there are more than 10 data points then a
            slider is included to provide for interactively selecting date
            range.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        title: str | None = None,
        display: bool = True,
        max_ticks: int | None = 10,
        direction: Literal["horizontal", "vertical"] = "vertical",
    ):
        """Create GUI.

        Parameters
        ----------
        data
            One column per symbol. Column of float values representing
            percentage change over constant frequency. Index as
            `pd.IntervalIndex` with dates representing period (of constant
            frequency) that the corresponding float values offer the
            percentage change over.

        title
            Chart title.

        display
            True to display created GUI.

        max_ticks
            Maximum number of x-axis ticks that will shown by default
            (client can choose to show more via slider). None for no limit.

        direction
            Direction of bars.
        """
        super().__init__(data, title, display, max_ticks, direction=direction)

    @property
    def ChartCls(self) -> type[charts.PctChgBar]:
        return charts.PctChgBar

    @property
    def SelectorCls(self) -> type[Selector] | None:
        return None

    def _create_date_slider(self, **_):
        return super()._create_date_slider(layout={"margin": "30px 0 15px 0"})

    def _create_gui_parts(self):
        super()._create_gui_parts()
        self._icon_row_top = self._create_icon_row_top()
        if len(self.chart.x_ticks) > self.chart.max_ticks:
            self.date_slider: wu.DateRangeSlider = self._create_date_slider()

    @property
    def _gui_box_contents(self) -> list[w.Widget]:
        contents = [self._icon_row_top, self.chart.figure]
        if hasattr(self, "date_slider"):
            contents += [self.date_slider]
        return contents


class GuiPctChgMult(GuiPctChg):
    """GUI for bar chart showing precentage changes of multiple instruments.

    GUI comprises:

        Close button: Icon to close gui.
        Figure: chart
        wu.DateRangeSlider: If there are more than 10 data points then a
            slider is offered provide for interactively selecting date
            range.
        PctChgIconRowMult: Provides for:
            Selecting chart type, either stacked or group
            Cycling legend location

    Notes
    -----
    See `charts.PctChgBarMult` documentation for known bugs concerning
    grouped bar chart.
    """

    @property
    def ChartCls(self) -> type[charts.PctChgBarMult]:
        return charts.PctChgBarMult

    def _create_icon_row(self):
        return gui_parts.PctChgIconRowMult()

    def _chart_type_handler(self, widget, event, data):
        self.chart.mark.type = self._icon_row.bar_type_tog.selected

    def _set_icon_row_handlers(self):
        handlers = [self._chart_type_handler] * 2
        self._icon_row.bar_type_tog.handlers_on_selecting = handlers
        self._icon_row.legend_cycle_but.on_event("click", self._cycle_legend_handler)

    def _create_gui_parts(self):
        super()._create_gui_parts()
        self._icon_row = self._create_icon_row()
        self._set_icon_row_handlers()

    @property
    def _gui_box_contents(self) -> list[w.Widget]:
        return super()._gui_box_contents + [self._icon_row]


class GuiOHLCCaseBase(GuiOHLC):
    """Base for analysis over OHLC for single financial instrument.

    Base class to create a gui with an OHLC chart with overlaid analysis.
    Analysis can comprise multiple 'cases', where a case could represent,
    for example, a trend or a position. gui provides user controls to
    display all classes collectively or focus on a single 'current' case
    and to navigate forwards or backwards between consecutive cases.

    Parameters
    ----------
    As base class, except:

    cases
        Cases of analysis to be displayed.

    narrow_view
        When displaying a case in 'narrow' view, the number of bars
        that should be shown before the bar representing the case's start
        and after the bar representing the case's conclusion.

    wide_view
        When displaying a case in 'wide' view, the number of bars
        that should be shown before the bar representing the case's start
        and after the bar representing the case's conclusion.

    Notes
    -----
    Subclass implementation
    -----------------------
    In addition to base clases implementation requirements, subclasses can
    optionally extend or override the following methods as required:
        _gui_handler_click_case
            Gui level handler for clicking a specific case. Alternatively
            a handler can be passed within `chart_kwargs` with the key
            'handler_click_case'.
    """

    def __init__(
        self,
        analysis: ma_analysis.Analysis,
        interval: mp.intervals.RowInterval,
        cases: CasesSupportsChartAnaly,
        max_ticks: int | None = None,
        log_scale: bool = True,
        display: bool = True,
        narrow_view: int = 10,
        wide_view: int = 10,
        chart_kwargs: dict | None = None,
        **kwargs,
    ):
        self.cases = cases
        self._narrow_view = narrow_view
        self._wide_view = wide_view
        if chart_kwargs is None:
            chart_kwargs = {}
        chart_kwargs.setdefault("cases", self.cases)
        chart_kwargs.setdefault("handler_click_case", self._gui_handler_click_case)
        super().__init__(
            analysis, interval, max_ticks, log_scale, display, chart_kwargs, **kwargs
        )

    @property
    def ChartCls(self) -> type[charts.OHLCCaseBase]:
        return charts.OHLCCaseBase

    @property
    def current_case(self) -> CaseSupportsChartAnaly | None:
        """Current selected case.

        None if no case is currently selected.
        """
        return self.chart.current_case

    def _gui_handler_click_case(
        self, case: CaseSupportsChartAnaly, mark: bq.Scatter, event: dict
    ):
        """Gui level handler for clicking a specific case.

        Lightens 'show all scatters' button to indicate option available.
        Displays tooltip to html output.

        Subclass can extend or override as required.
        """
        self.cases_controls_container.lighten_single_case()
        self.cases_controls_container.but_show_all.darken()
        html = self.cases.get_case_html(case)
        self.html_output.display(html)

    def _show_all_but_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_light:
            self.chart.hide_cases()
            but.darken()
            return

        if self.current_case is not None:
            self.chart.deselect_current_case()
            self.cases_controls_container.darken_single_case()
        else:
            self.chart.show_cases()
        but.lighten()

    def _select_next_case_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_dark:
            return
        self.chart.select_next_case()

    def _select_prev_case_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_dark:
            return
        self.chart.select_previous_case()

    def _set_slider_to_current_case(self, bars: int):
        """Set slider to focus on current selected case.

        Parameters
        ----------
        bars
            Number of bars to view prior to case start and following
            case's conclusion.
        """
        if self.current_case is None:
            return
        index = self.cases.data.index
        start = max(index.get_loc(self.current_case._start) - bars, 0)
        end = self.current_case._end
        if end is None:
            stop = len(index) - 1
        else:
            stop = min(index.get_loc(end) + bars, len(index) - 1)
        self.date_slider.interval = pd.Interval(index[start], index[stop], "both")

    def _narrow_view_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_dark:
            return
        self._set_slider_to_current_case(self._narrow_view)

    def _wide_view_handler(self, but: vu.IconBut, event: str, data: dict):
        if but.is_dark:
            return
        self._set_slider_to_current_case(self._wide_view)

    def _create_cases_controls_container(self) -> v.Layout:
        controls = gui_parts.CaseControls()
        controls.but_show_all.on_event("click", self._show_all_but_handler)
        controls.but_next.on_event("click", self._select_next_case_handler)
        controls.but_prev.on_event("click", self._select_prev_case_handler)
        controls.but_narrow.on_event("click", self._narrow_view_handler)
        controls.but_wide.on_event("click", self._wide_view_handler)
        return controls

    def _create_controls_container(self) -> v.Layout:
        self._tabs_control_container = v.Layout(
            children=[self.tabs_control], class_="d-flex justify-end mr-2"
        )
        self.cases_controls_container = self._create_cases_controls_container()
        return v.Layout(
            children=[self._tabs_control_container, self.cases_controls_container],
            class_="d-flex align-center justify-center",
        )

    def _create_gui_parts(self):
        super()._create_gui_parts()
        self._controls_container = self._create_controls_container()

    @property
    def _gui_box_contents(self) -> list[w.Widget]:
        contents = [
            self._icon_row_top,
            self.chart.figure,
            self.date_slider,
            self._controls_container,
            self.html_output,
        ]
        return contents

    def _create_date_slider(self, **kwargs):
        ds = super()._create_date_slider(**kwargs)
        ds.slider.observe(self.chart.update_trend_mark, ["index"])
        return ds

    @contextmanager
    def _handler_disabled(self):
        """Undertake an operation within context of disabled handler.

        Undertake an operation with context of slider's handlers being
        disabled.

        Notes
        -----
        Messily overrides method defined on `BaseVariableDates` subclass to
        include the additional `self.chart.update_trend_mark` handler.
        """
        self._slider.unobserve(self._set_chart_x_ticks_to_slider, ["index"])
        self._slider.unobserve(self.chart.update_trend_mark, ["index"])
        yield
        self._slider.observe(self._set_chart_x_ticks_to_slider, ["index"])
        self._slider.observe(self.chart.update_trend_mark, ["index"])

    def _max_x_ticks(self):
        """Show data for all plottable x-ticks (bars).

        Notes
        -----
        Overrides inherited method to ensure any added marks are also
        updated.
        """
        self.chart.reset_x_ticks()
        self._set_slider_limits_to_all_plottable_x_ticks()
        self.chart.update_trend_mark()
