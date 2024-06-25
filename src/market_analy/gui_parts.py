"""Definitions of widgets to contribute to GUI.

Widgets define the GUI, they do NOT define any event handling.

Functions
---------
loading_overlay() -> v.Overlay:
    Overlay application with transparent sheet with loading symbol.

close_but() -> wu.ButtonIcon:
    Return 'Close' icon button.

legend_but() -> vu.IconButton:
    Return `IconButton` to cycle legend location.

rebase_but() -> vu.IconBut:
    Return `IconButton` to rebase a multiline chart.

Classes
-------
IntervalSelector:
    ToggleButtons to select a date interval.

IconRowTop:
    Row of icons to place above chart with selectable dates.

TabsControl:
    Tabs for price chart operations.

HtmlOutput:
    HTML output box.

BarDirectionToggle:
    Toggle to set bar chart direction.

BarTypeToggle:
    Toggle to set bar chart type.

PctChgIconRow:
    Percent change bar chart options for single instrument.

PctChgIconRowMult:
    Percent change bar chart options for multiple instruments.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import re

import ipywidgets as w
import ipyvuetify as v
from market_prices import intervals

import market_analy.utils.ipywidgets_utils as wu
import market_analy.utils.ipyvuetify_utils as vu
from market_analy.utils.dict_utils import set_kwargs_from_dflt

ICON_DIM = "35px"

TT_KWARGS_DFLT = {"bottom": True, "open_delay": 750, "content_class": "black--text"}

BG = "grey darken-4"


def close_but(handler: Callable | None = None) -> wu.ButtonIcon:
    """Return close icon button. Red with white cross.

    Parameters
    ----------
    handler
        On-click handler. Signature should accommodate following parameters:
            handler_name(widget)
    """
    layout = w.Layout(align_self="flex-start", width=ICON_DIM)
    but = wu.ButtonIcon.close_(layout=layout)
    if handler is not None:
        but.on_click(handler)
    but.tooltip = "Close"
    return but


class IntervalSelector(w.ToggleButtons):
    """`w.ToggleButtons` to select a date interval."""

    def __init__(
        self,
        labels: list[str],
        value: intervals.PTInterval | None = None,
        handler: Callable | None = None,
        layout_kwargs: dict | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        labels
            Button labels. Should be `market-prices` shorthand intervals
            from which PTIntervals can be instantiated, for example:
                ['3mo', '1d', '1h', '30m', '5m']

        value
            Value corresponding with any toggle button to initially select.
            If not passed then first toggle button will be selected.

        handler
            Handler to handle changes to `ToggleButtons` value. Should have
            signature `handler(change: dict)` where 'change' is a
            dictionary including keys 'old' and 'new' which have values as
            old and new price intervals, of type `intervals.PTInterval`.

        layout_kwargs
            Passed to `w.Layout` assigned to `ToggleButtons`.
        """
        values = [intervals.to_ptinterval(lab) for lab in labels]
        tbs_style = w.ToggleButtonsStyle(button_width="40px")
        tooltips = ["Select date interval" for i in range(len(values))]
        layout_kwargs = layout_kwargs if layout_kwargs is not None else {}
        layout_kwargs.setdefault("justify_content", "center")
        layout_kwargs.setdefault("margin", "15px 0 0 0")
        layout = w.Layout(**layout_kwargs)
        super().__init__(
            options=list(zip(labels, values)),
            tooltips=tuple(tooltips),
            style=tbs_style,
            layout=layout,
        )

        if value is not None and value in values:
            self.value = value
        self._handler = handler
        self._observe_handler()

    def _observe_handler(self):
        if self._handler is not None:
            self.observe(self._handler, ["value"])

    def _unobserve_handler(self):
        if self._handler is not None:
            self.unobserve(self._handler, ["value"])

    @contextmanager
    def _handler_disabled(self):
        """Undertake operation within context of handler being disabled."""
        self._unobserve_handler()
        yield
        self._observe_handler()

    def set_value_unobserved(self, value: intervals.PTInterval):
        """Set value without triggering any handler."""
        with self._handler_disabled():
            self.value = value


def loading_overlay() -> v.Overlay:
    """Overlay application with opaque sheet with loading symbol."""
    prog = v.ProgressCircular(indeterminate=True, color="white", size=30)
    return v.Overlay(color="grey", value=False, absolute=True, children=[prog])


class Dialog(vu.Dialog):
    """Dialog box.

    Pass instance as child of component to be overlaid and within which
    dialog box to appear.
    """

    def __init__(self, title: str | None = None, text: str | None = None):
        title = title if title is not None else "title placeholder"
        text = text if text is not None else "text placeholder"
        super().__init__(title, text)


class IconRowTop(wu.IconRow):
    """Row of icons to place above chart with selectable dates.

    Includes following icons:
        'max_dates'
        'refresh' (only if 4 handlers passed to constructor)
        'resize_window'
        'close'

    Constructor accommodates on-click handlers.

    Parameters
    ----------
    handlers
        3 or 4-list of callables to serve as on-click handlers for,
        respectively, `IconButtons` 'max_dates', 'refresh' (only if 4)
        'resize_window' and 'close'.
    """

    def __init__(self, handlers: list[Callable] | None = None, **kwargs):
        assert handlers is None or len(handlers) in [3, 4]
        has_3_handlers = handlers is not None and len(handlers) == 3
        icon_button_names = ["max_dates", "refresh", "resize_window", "close_"]
        if has_3_handlers:
            icon_button_names.remove("refresh")

        super().__init__(icon_button_names, handlers=handlers, **kwargs)

        tooltips = ["Max dates", "Reset", "Resize Chart", "Close"]
        if has_3_handlers:
            tooltips.remove("Reset")
        for i, child in enumerate(self.children):
            child.tooltip = tooltips[i]


class TabsControl(v.Tabs):
    """Tabs widget.

    Includes:
        Cursor tab, one row:
            ToggleIcons[+ - IconBut] // Divider // Lightbulb IconBut
                // Trash IconBut
        Selector tab, one row:
            Zoom IconBut // UpArrow IconBut // DownArrow IconBut

    Attributes
    ----------
    cursor_objs: `vue.Layout`
        Container of widgets that comprise cursor tab items.
    cursor_toggle: `ToggleIcons`
    but_lightbulb: `IconBut`
    self.but_trash: `IconBut`

    slctr_objs: `vue.Layout`
        Container of widgets that comprise selector tab items.
    but_zoom: `IconBut`
    but_arrow_up: `IconBut`
    but_arrow_down: `IconBut`
    """

    _tab_container_class_ = "d-flex justify-center align-center"

    def __init__(self):
        self.cursor_objs: v.Layout
        self.cursor_toggle: vu.ToggleIcons
        self.but_lightbulb: vu.IconBut
        self.but_trash: vu.IconBut
        self._create_cursor_tab_content()

        self.slctr_objs: v.Layout
        self.but_zoom: vu.IconBut
        self.but_arrow_up: vu.IconBut
        self.but_arrow_down: vu.IconBut
        self._create_slctr_tab_content()

        tab_class_ = "text-lowercase font-weight-regular " + BG

        texts = ["Cursor", "Selector"]
        tooltips = ["Crosshair Operations", "Operate on a selected period"]
        tts_kwargs = [{"left": True}, {"right": True}]
        for kwargs in tts_kwargs:
            kwargs["open_delay"] = TT_KWARGS_DFLT["open_delay"]
        tab_tt_list = []
        for text, tt, tt_kwargs in zip(texts, tooltips, tts_kwargs):
            tab_tt = vu.TabTt(
                children=[text], class_=tab_class_, tooltip=tt, tt_kwargs=tt_kwargs
            ).tt
            tab_tt_list.append(tab_tt)

        tab_items = [
            v.TabItem(children=[layout], class_=BG)
            for layout in [self.cursor_objs, self.slctr_objs]
        ]

        tabs_class_ = "d-flex flex-column align-center flex-grow-0 " + BG
        super().__init__(
            v_model=0,
            children=tab_tt_list + tab_items,
            centered=True,
            fixed_tabs=True,
            dark=True,
            slider_size=1,
            height="30px",
            class_=tabs_class_,
        )

    def _create_cursor_tab_content(self):
        colors = ["green", "red"]
        d = {
            "icon_names": ["fa-plus", "fa-minus"],
            "names": ["plus", "minus"],
            "tooltips": ["Add crosshair", "Remove crosshair"],
            "deselected_colors": ["green darken-4", "red darken-4"],
            "selected_colors": colors,
        }

        d["tts_kwargs"] = [
            set_kwargs_from_dflt({"color": c}, TT_KWARGS_DFLT) for c in colors
        ]

        self.cursor_toggle = vu.ToggleIcons(**d)

        divider = vu.IconBut.divider()

        tt = "Toggle crosshairs transparency"
        color = "yellow"
        tt_kwargs = {"color": color}
        set_kwargs_from_dflt(tt_kwargs, TT_KWARGS_DFLT)
        self.but_lightbulb = vu.ToggleIcon(
            "fa-lightbulb-o",
            selected=True,
            deselected_color=color + " darken-4",
            selected_color=color,
            tooltip=tt,
            tt_kwargs=tt_kwargs,
        )

        tt = "Remove all crosshairs"
        color = "red accent-3"
        tt_kwargs = {"color": color}
        set_kwargs_from_dflt(tt_kwargs, TT_KWARGS_DFLT)
        self.but_trash = vu.IconBut(
            "fa-trash", color=color, tooltip=tt, tt_kwargs=tt_kwargs
        )

        cursor_buts = [self.but_lightbulb.tt, self.but_trash.tt]
        children = self.cursor_toggle.contain_me + [divider] + cursor_buts
        self.cursor_objs = v.Layout(
            children=children, class_=self._tab_container_class_
        )

    def _create_slctr_tab_content(self):
        def get_but(icon_name: str, color: str, tooltip: str) -> vu.IconBut:
            class_ = "mx-4 mb-2 mt-4"
            tt_kwargs = {"color": color}
            set_kwargs_from_dflt(tt_kwargs, TT_KWARGS_DFLT)
            return vu.IconBut(
                icon_name,
                color=color,
                class_=class_,
                tooltip=tooltip,
                tt_kwargs=tt_kwargs,
            )

        self.but_zoom = get_but(
            "fa-search-plus", "orange lighten-2", "Zoom to selection"
        )

        tt = "Evalute maximum advance over selected period"
        self.but_arrow_up = get_but("fa-arrow-circle-up", "light-green lighten-1", tt)

        tt = "Evalute maximum decline over selected period"
        self.but_arrow_down = get_but("fa-arrow-circle-down", "red lighten-1", tt)

        buts = [self.but_zoom.tt, self.but_arrow_up.tt, self.but_arrow_down.tt]
        self.slctr_objs = v.Layout(children=buts, class_=self._tab_container_class_)

    def reset(self):
        self.v_model = 0
        self.cursor_toggle.deselect()


class HtmlOutput(w.HBox):
    """HTML Output box with button to close display."""

    def __init__(self):
        self._html = w.HTML()

        self._close_but = wu.ButtonIcon.close_(
            tooltip="Close output", layout=w.Layout(height="30px")
        )
        self._close_but.on_click(self._close_button_handler)
        self._hide_close_button()

        super().__init__(
            children=[self._html, self._close_but],
            layout=w.Layout(
                justify_content="center",
                align_self="center",
                grid_gap="10px",
                margin="20px 0 0 0",
            ),
        )

    def _show_close_button(self):
        self._close_but.layout.visibility = "visible"
        self._close_but.layout.margin = "0 0 0 0"  # "25px 0 0 0"

    def _hide_close_button(self):
        self._close_but.layout.visibility = "hidden"
        self._close_but.layout.margin = "0 0 0 0"

    def display(self, html: str, add_padding: int = 0):
        """Display html.

        Parameters
        ----------
        html
            HTML to be displayed.

        add_padding
            Where html includes a table, add padding in the form of blank
            spaces to the start of the contents of every 'td' and 'tr'
            tabs. Pass `add_padding` as number of spaces to be included.

        Notes
        -----
        add_padding option provided to account for padding style
        information being annoyingly lost, or at least not recognised,
        when a `HtmlOutput` widget is otherwise included to a
        `ipyvuetify.app`.
        """
        if add_padding:

            def repl(mo: re.Match) -> str:
                match = mo.group()
                return f'>{"&nbsp" * add_padding}{match[1:]}'

            regex = re.compile(r"\>.+(?=.*\</t[hd]\>)")
            html = re.sub(regex, repl, html)

        self._html.value = html
        self._show_close_button()

    def clear(self):
        """Clear output."""
        self._html.value = ""
        self._hide_close_button()

    def _close_button_handler(self, widget):
        self.clear()


class _BiToggle(vu.ToggleIcons):
    """Two `IconButtons`, one of which has to be selected."""

    def __init__(
        self,
        icon_names: list[str],
        names: list[str],
        color: str,
        tooltips: list[str],
        **kwargs,
    ):
        d = {
            "icon_names": icon_names,
            "names": names,
            "deselected_colors": ["grey lighten-1"] * 2,
            "selected_colors": [color] * 2,
            "tooltips": tooltips,
            "tts_kwargs": [
                {
                    "bottom": True,
                    "color": color,
                    "open_delay": 750,
                    "content_class": "white--text",
                }
            ]
            * 2,
            "select_by": "name",
            "allow_no_selection": False,
            "initial_selection": names[0],
        }
        super().__init__(**d, **kwargs)


class BarDirectionToggle(_BiToggle):
    """Toggle to set bar chart direction.

    Two `IconButton` toggle to set bar chart direction, vertical or
    horizontal.
    """

    def __init__(self):
        d = {
            "icon_names": ["fa-grip-lines-vertical", "fa-grip-lines"],
            "names": ["vertical", "horizontal"],
            "color": "orange",
            "tooltips": ["vertical bars", "horizontal bars"],
        }
        super().__init__(**d)


class BarTypeToggle(_BiToggle):
    """Toggle to set bar chart type.

    Two `IconButton` toggle to set bar chart type, stacked or
    grouped.
    """

    def __init__(self):
        d = {
            "icon_names": ["mdi-chart-bar-stacked", "mdi-chart-bar"],
            "names": ["stacked", "grouped"],
            "color": "blue",
            "tooltips": ["stacked", "grouped"],
        }
        super().__init__(**d)


def create_icon_but(
    icon_name: str,
    tooltip: str,
    color: str,
    handler: Callable | None = None,
    **kwargs,
) -> vu.IconBut:
    """`IconButton` with tooltip and handler.

    Parameters
    ----------
    icon_name
        Example, 'fa-list-alt'

    tooltip
        Tooltip text.

    color

    handler
        If passed handler will handle button clicks. Handler should have
        signature:
            handler(widget, event, data)

        If not passed then handler can be subsequently set with:
            but.on_event('click', handler)

    **kwargs
        Passed on to `IconBut`. Will not pass on parameters defined by
        method to maintain legend button style.
    """
    kwargs["icon_name"] = icon_name
    kwargs["tooltip"] = tooltip
    kwargs["color"] = color

    dark = kwargs.get("dark", False) and (dark_color := kwargs.get("dark_color", False))
    tt_kwargs = {"color": dark_color if dark else color}
    tt_kwargs = set_kwargs_from_dflt(tt_kwargs, TT_KWARGS_DFLT)
    tt_kwargs = set_kwargs_from_dflt(kwargs.pop("tt_kwargs", {}), tt_kwargs, deep=True)
    but = vu.IconBut(tt_kwargs=tt_kwargs, **kwargs)
    if handler is not None:
        but.on_event("click", handler)
    return but


def legend_but(handler: Callable | None = None, **kwargs) -> vu.IconBut:
    """`IconButton` to cycle legend location.

    Parameters
    ----------
    handler
        If passed handler will handle clicks on the legend button. Handler
        should have signature:
            handler(widget, event, data)

    **kwargs
        Passed on to `IconBut`. Will not pass on parameters defined by
        method to maintain legend button style.
    """
    return create_icon_but(
        icon_name="fa-list-alt",
        tooltip="Cycle legend location",
        color="white",
        handler=handler,
        **kwargs,
    )


def rebase_but(handler: Callable | None = None, **kwargs) -> vu.IconBut:
    """`IconButton` to rebase a multiline chart.

    Parameters
    ----------
    handler
        If passed handler will handle clicks of rebase button. Handler
        should have signature:
            handler(widget, event, data)

    **kwargs
        Passed on to `IconBut`. Will not pass on parameters defined by
        method to maintain legend button style.
    """
    return create_icon_but(
        icon_name="fa-dot-circle-o",
        tooltip="Rebase prices",
        color="orange",
        handler=handler,
        **kwargs,
    )


class PctChgIconRowMult(v.Layout):
    """Percent change bar chart options for multiple instruments."""

    def __init__(self):
        self.bar_type_tog = BarTypeToggle()
        self.legend_cycle_but = legend_but()

        children = [
            v.Layout(children=self.bar_type_tog.contain_me, class_="flex-grow-0"),
            self.legend_cycle_but.tt,
        ]

        super().__init__(children=children, class_=BG, justify_space_around=True)


class CaseControls(v.Layout):
    """Container of controls for selecting and viewing analysis cases.

    Client should directly assign handlers to exposed buttons after
    creating instance.

    Includes vu.IconBut arranged as:

                        but_next  //  but_prev
        but_show_all //
                        but_narrow  //  but_wide

    Subclasses can extend by extending or overriding the `_create_children`
    method.
    """

    _TURN_OFF_COLOR = "#CD853F"

    def __init__(self):
        self.but_show_all: vu.IconBut
        self.but_next: vu.IconBut
        self.but_prev: vu.IconBut
        self.but_narrow: vu.IconBut
        self.but_wide: vu.IconBut
        self._buts: list[vu.IconBut] = []
        self._dark = True

        super().__init__(
            children=self._create_children(),
            class_="d-flex justify-start ml-2",
        )

    def _create_iconbut(
        self,
        icon_name: str,
        tooltip: str,
        color: str,
        class_="mr-2",
        dark: bool | None = None,
    ) -> vu.IconBut:
        if dark is None:
            dark = self._dark
        but = create_icon_but(
            icon_name=icon_name,
            tooltip=tooltip,
            color=color,
            class_=class_,
            dark_color="grey",
            dark=dark,
        )
        self._buts.append(but)
        return but

    def _create_show_all_container(self) -> v.Layout:
        self.but_show_all = self._create_iconbut(
            icon_name="mdi-chart-scatter-plot",
            tooltip="Toggle visibilty of case marks",
            color="yellow",
            class_="flex-grow-0 align-self-center",
            dark=False,
        )
        return v.Layout(
            children=[
                v.Layout(
                    children=[self.but_show_all.tt],
                    class_="d-flex justify-start flex-grow-0 mr-2",
                )
            ],
            class_="d-flex justify-start flex-grow-0",
        )

    def _create_prev_next_container(self) -> v.Layout:
        self.but_prev = self._create_iconbut(
            "mdi-arrow-left-circle", "Select previous case", "#F0E68C"
        )
        self.but_next = self._create_iconbut(
            "mdi-arrow-right-circle", "Select next case", "#FFD700"
        )
        return v.Layout(
            children=[self.but_prev.tt, self.but_next.tt],
            class_="d-flex justify-start mb-2",
        )

    def _create_narrow_wide_view_container(self) -> v.Layout:
        self.but_narrow = self._create_iconbut(
            "mdi-arrow-expand-horizontal",
            "Narrow view around selected case",
            "#BDB76B",
        )
        self.but_wide = self._create_iconbut(
            "mdi-arrow-left-right", "Wide view around selected case", "#BDB76B"
        )
        return v.Layout(
            children=[self.but_narrow.tt, self.but_wide.tt],
            class_="d-flex justify-start",
        )

    def _create_case_container(self) -> v.Layout:
        contain_prev_next = self._create_prev_next_container()
        contain_narrow_wide = self._create_narrow_wide_view_container()
        return v.Layout(
            children=[contain_prev_next, contain_narrow_wide],
            class_="d-flex flex-column justify-center flex-grow-0 ml-2",
        )

    def _create_children(self) -> list[v.Layout]:
        """Create all children to populate top level container."""
        return [
            self._create_show_all_container(),
            self._create_case_container(),
        ]

    @property
    def is_dark_single_case(self) -> bool:
        """Query if icons for single-case actions are dark."""
        return self._dark

    def darken_single_case(self):
        """Darken icon buttons for single-case actions.

        Will not darken any icon waiting to be switched off.
        """
        for but in self._buts:
            if (but.icon_color == self._TURN_OFF_COLOR) or but is self.but_show_all:
                continue
            but.darken()
        self._dark = True

    def lighten_single_case(self):
        """Lighten icon buttons for single-case actions."""
        for but in self._buts:
            if but is self.but_show_all:
                continue
            but.lighten()
        self._dark = False


class TrendControls(CaseControls):
    """Container of controls for selecting and viewing trend movements.

    Client should directly assign handlers to exposed buttons after
    creating instance.

    Includes vu.IconBut arranged as:

                        but_next  //  but_prev
        but_show_all //                           // but_ruler
                        but_narrow  //  but_wide
    """

    _TURN_OFF_COLOR = "#CD853F"

    def __init__(self):
        self.but_ruler: vu.IconBut

        super().__init__()

    def but_ruler_handler(self, but: vu.IconBut, event: str, data: dict):
        """Handler to toggle ruler but color light/red if group not dark."""
        if but.is_dark:
            return
        if but.is_light:
            but.icon_color = self._TURN_OFF_COLOR
            return
        f = but.darken if self.is_dark_single_case else but.lighten
        f()

    def _create_ruler_container(self) -> v.Layout:
        self.but_ruler = self._create_iconbut(
            icon_name="mdi-ruler",
            tooltip="Show ruler with width of trend period",
            color="yellow",
            class_="flex-grow-0 align-self-center",
        )
        self.but_ruler.on_event("click", self.but_ruler_handler)
        return v.Layout(
            children=[self.but_ruler.tt],
            class_="d-flex justify-start flex-grow-1",
        )

    def _create_children(self) -> list[v.Layout]:
        """Create all children to populate top level container."""
        children = super()._create_children()
        children.append(self._create_ruler_container())
        return children
