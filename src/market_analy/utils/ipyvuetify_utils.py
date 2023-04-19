"""Utility functions and classes for ipyvuetify library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import ipyvuetify as vue  # type: ignore[import]

from .list_utils import SelectableList

DARK = "grey darken-4"
LIGHT = "grey lighten-4"


def border_comp(
    component: vue.VuetifyWidget,
    color: str = "orange",
    border_width: str = "5px",
    style: str = "dashed",
):
    """Draw a border around a component.

    Useful to see the limit of the component's layout.

    Remove border with component.style_ = None.

    NOTE: will erase any existing _style attribute.
    """
    style_string = " ".join(["border:", border_width, style, color])
    component.style_ = style_string


def tooltip_decorator(func: Callable) -> Callable:
    """Decorator to optionally add a tooltip to a component.

    Implementation
    --------------
    Add decorator to the constructor of a class that instantiates the
    component to which the tooltip is to be added.

    Add the following parameters to the constructor signature:
        tooltip: str | None = None,
        tt_kwargs: dict | None = None,

    Add the following to the construtor's documentation:

        tooltip
            Component to be displayed as a tooltip, or text (str) for
            tooltip label, for example tooltip = 'this is my tooltip'.

        tt_kwargs
            kwargs to be passed to any vue.Tooltip. Only applicable if
            `tooltip` passed. Exmample:
                tt_kwargs = {'bottom': True, 'color': 'red'}
                kwargs to set position are 'top', 'right', 'bottom', 'left'.
                To see all possible kwargs select v-tooltip at:
                    https://vuetifyjs.com/en/components/api-explorer/

    Add the following to the class documentation:

        NB If `tooltip` is passed then the tooltip (self.tt), NOT the
        component (self), should be passed to the parent component, For
        exmaple:
            vue.Card(children=[tab_tt.tt]).

        Attributes
        ----------
        tt: vue.Tooltip | None
            Associated Tooltip (None if `tooltip` not passed to constructor).

    Optionally, explicitly declare the tooltip in the decorated
    constructor. For example:
        self.tt: vue.Tooltip

    Usage
    -----
    Tooltip will be assigned to component 'tt' attribute.

    Whenever a tooltip is defined the tooltip, NOT the component, should be
    passed the parent component, e.g. vue.Card(children=[component.tt]).
    This is necessary as, rather oddly, vuetify defines the component
    associated with a tooltip as the child of the tooltip's activator
    slot (the tooltip wraps the component) as opposed to the tooltip
    being a child of the component.
    """

    def wrapper(self, *args, **kwargs):
        tooltip = kwargs.get("tooltip", None)
        tt_kwargs = kwargs.get("tt_kwargs", None)

        if tooltip is not None:
            kwargs["v_on"] = "ttip.on"

        func(self, *args, **kwargs)

        if tooltip is not None:
            tt_kwargs = tt_kwargs if tt_kwargs is not None else {}
            tt_kwargs["children"] = [tooltip]
            self.tt = vue.Tooltip(
                v_slots=[{"name": "activator", "variable": "ttip", "children": [self]}],
                **tt_kwargs,
            )
        else:
            self.tt = None

    return wrapper


class IconBut(vue.Btn):
    """Button containing icon and an optional tooltip.

    NB If `tooltip` passed then the tooltip (self.tt), NOT the icon_but
    (self), should be passed to the parent component, For exmaple:
        vue.Card(children=[icon_but.tt]).

    Attributes
    ----------
    tt: vue.Tooltip | None
        Associated Tooltip (None if `tooltip` not passed to constructor).

    Parameters
    ----------
    icon_name
        As string definition used by vuetify to define an icon, e.g.
        'fa-trash'.

    color
        Icon color. As vuetify color definition, for example 'red' or
        'grey lighten-4'. Default 'grey lighten-4'.

    dark_color
        Icon color when darkened. As vuetify color definition, for
        example 'grey' or 'grey darken-4'.

    dark : bool, default: False
        True to initially set icon color to dark.

    tooltip
        Component to be displayed as a tooltip, or text (str) for
        tooltip label, for example `tooltip = 'this is my tooltip'`.

    tt_kwargs
        kwargs to be passed to any vue.Tooltip. Only applicable if
        `tooltip` passed. Exmample:
            tt_kwargs = {'bottom': True, 'color': 'red'}
        kwargs to set position are 'top', 'right', 'bottom', 'left'.
        To see all possible kwargs select v-tooltip at:
            https://vuetifyjs.com/en/components/api-explorer/

    kwargs
        Passed on to `vue.Btn` base class constructor.

        NB If `class_` is not included to kwargs then assigned value
        'ma-2 mt-4'.
    """

    # DEVELOPMENT

    # Currently does not provide for the button to house text.
    # NB As of Nov 20, seems that TEXT WILL ALWAYS BE CONVERTED TO
    # UPPERCASE! Annoying.
    # Could develop to provide for the button to optinally house text
    # either before/after the icon or indeed to house soley text and
    # no vue.Icon, such that the 'icon' is the text. Would require the
    # constructor defining icon light and dark colors and separately
    # text light and text colors (NB when icon=True button text takes
    # the color passed to the button 'color' argument and the icon
    # background colour takes on, by deafult, whatever's underneath).
    # Then extend the--lighten-- and --darken-- methods to also change
    # the text (i.e. button) color.
    # Also NB if don't select icon=True then could create the icon
    # effect with rounded=True and setting the size to no more than
    # the minimum to hold the content. This way could still use color
    # to set the background color. Althoguh would need to investigate
    # how to then set the color! Half of one and six a dozen of the
    # other.

    # Currently does not provide for setting button background color:
    # NB button colour usually set via the color argument although
    # when icon=True this argument sets the text colour rather than
    # the background colour, with the background colour taking on
    # whatever's underneath.
    # Could develp provide for - the background color can be set via
    # class_ attribute. Probably could implement by defining ++class_++
    # to not include color, with color defined separately, have a
    # but_color option property that can be set via a setter property,
    # then dynamically evaluate class_ and assign to class_ on any
    # changes to class_ or color (will probably involve defining
    # properties here which in turn set properties on the underlying
    # class via super()).

    icon_dim = "35px"
    def_class_ = "ma-2 mt-4"

    @classmethod
    def divider(
        cls, color: str = "grey darken-1", vertical: bool = True, **kwargs
    ) -> vue.Divider:
        """Return Divider object that aligns with default icon buttons."""
        style_ = (" ").join(
            ["max-height:", IconBut.icon_dim + ";", "height:", IconBut.icon_dim]
        )
        kwargs.setdefault("class_", cls.def_class_)
        kwargs["class_"] += " " + color
        return vue.Divider(style_=style_, vertical=vertical, **kwargs)

    @tooltip_decorator
    def __init__(
        self,
        icon_name: str,
        color=LIGHT,
        dark_color=DARK,
        dark=False,
        tooltip: str | None = None,
        tt_kwargs: dict | None = None,
        **kwargs,
    ):
        self.tt: vue.Tooltip
        self.color_dark = dark_color
        self.color_light = color

        assert "children" not in kwargs
        kwargs.setdefault("height", self.icon_dim)
        kwargs.setdefault("width", self.icon_dim)
        kwargs.setdefault("min_width", self.icon_dim)
        kwargs.setdefault("elevation", 5)
        kwargs.setdefault("class_", self.def_class_)
        kwargs.setdefault("icon", True)

        color = self.color_dark if dark else self.color_light

        super().__init__(
            children=[vue.Icon(children=[icon_name], small=True, color=color)], **kwargs
        )

    @property
    def icon_color(self) -> str:
        return self.children[0].color

    @icon_color.setter
    def icon_color(self, color: str):
        self.children[0].color = color
        self.tt.color = color

    def darken(self):
        """Darken icon color."""
        self.icon_color = self.color_dark

    def lighten(self):
        """Lighten icon color."""
        self.icon_color = self.color_light

    @property
    def is_dark(self) -> bool:
        """Query if icon is dark."""
        return self.icon_color == self.color_dark

    @property
    def is_light(self) -> bool:
        """Query if icon is light."""
        return self.icon_color == self.color_light


class ToggleIcon(IconBut):
    """Selectable `IconBut`.

    Clicking `IconBut` toggles selection. Optionally:
        Define selected and unselected icon colors.
        Set selecting/deselecting handlers.

    Parameters
    ----------
    icon_name
        Name as used by vuetify to define icon, e.g. 'fa-trash'.

    selected
        True to initially select button.

    deselected_color
        Icon color when not selected. As vuetify color definition, for
        example 'grey' or grey darken-4'.

    selected_color
        Icon color when selected. As vuetify color definition, for
        example 'grey' or grey lighten-4'.

    handler_on_selecting
        Callable to be executed on icon button becoming selected. If not
        passed then a handler can be subsequently assigned to
        attribute --handler_on_selecting--.

        Handler should have following signature to accommodate passed
        parameters:
            handler_name(widget, event, data)

    handler_on_deselecting
        Callable to be executed on icon button becoming deselected. If
        not passed then a handler can be subsequently assigned to
        attribute --handler_on_deselecting--.

        Handler should have following signature to accommodate passed
        parameters:
            handler_name(widget, event, data)

    kwargs:
        Passed on to `IconBut` constructor.

    Attributes
    ----------
    handler_on_selecting : Callable | None
        Handler to be executed on icon button being selected. Set to None
        to remove any existing handler.

    handler_on_deselecting : Callable | None
        Handler to be executed on icon button being deselected. Set to None
        to remove any existing handler.

    Settable Properties
    -------------------
    selected: -> bool
        True if icon button selected, False otherwise.
    """

    # DEVELOPMENT NB
    # As covered to IconBut, currently not possible to set the buttons color,
    # rather they will take the color of their parent. If such funcationality
    # proves desirable then would be possible to further develop Iconbut to
    # accommodate this - see Devlopment NB to IconBut class.

    def __init__(
        self,
        icon_name: str,
        selected=False,
        deselected_color=DARK,
        selected_color=LIGHT,
        handler_on_selecting: Callable | None = None,
        handler_on_deselecting: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            icon_name, color=selected_color, dark_color=deselected_color, **kwargs
        )

        self.handler_on_selecting = handler_on_selecting
        self.handler_on_deselecting = handler_on_deselecting
        self.selected = selected
        self.on_event("click", self._selection_handler)

    @property
    def selected(self) -> bool:
        """True if icon button selected, False otherwise."""
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        self._selected = value
        if self._selected:
            self.lighten()
        else:
            self.darken()

    def _toggle_selection(self):
        self.selected = not self.selected

    def _selected_handler(self):
        if self.handler_on_selecting is not None:
            self.handler_on_selecting()

    def _unselected_handler(self):
        if self.handler_on_deselecting is not None:
            self.handler_on_deselecting()

    def _selection_handler(self, widget, event, data):
        self._toggle_selection()
        if self.selected:
            self._selected_handler()
        else:
            self._unselected_handler()


class ToggleIcons(SelectableList):
    """List of IconBut of which EITHER zero or one OR one must be selected.

    Can allow no selection or require one button to always be selected.

    `Iconbut` objects all dark except any currently selected `IconBut`.

    All methods of `SelectableList` available to select an `IconBut`.

    Clicking an `IconBut` will select it. If no selection is permitted, then
    clicking a selected `IconBut` will deselect it.

    Independent tooltips can be added to all or any icon button.

    Usage
    -----
    To house in a parent component, pass `.contain_me` attribute to parent
    component's children attribute. NB If `tooltips` not passed then can
    alternatively pass object (self) directly to parent container's
    children kwarg.

    Attributes
    ----------
    contain_me: List[Union[IconBut, vue.Tooltip]]
        To house `ToggleIcons` in a parent component, pass `.contain_me` to
        parent component's children attribute.

    Notes
    -----
    `ToggleIcons` does NOT employ ToggleIcon objects.
    """

    # DEVELOPMENT
    # As covered to `IconBut`
    # currently text can not accompany/substitue the icon. Could make
    # changes to provide for text.
    # currently not possible to set the buttons background color (as
    # buttons have icon=True which results in color being assigned
    # to the text and border as opposed to fill). As noted to
    # IconBut, could develop to provide for independently setting
    # background colour.

    def __init__(
        self,
        icon_names: list[str | int],
        names: list[str | int] | None = None,
        tooltips: list[str | None] | None = None,
        tts_kwargs: list[dict] | None = None,
        select_by: Literal["element", "name", "index"] = "name",
        allow_no_selection=True,
        initial_selection=None,
        deselected_colors: list[str] | None = None,
        selected_colors: list[str] | None = None,
        handlers_on_selecting: list[Callable | None] | None = None,
        handlers_on_deselecting: list[Callable | None] | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        icon_names
            List of string definitions as used by vuetify to define icons,
            e.g. ['fa-trash', fa-minus, fa-plus].

        names
            List of strings of same length as icon_names. If passed names
            can be used to reference icon buttons via the 'name' methods
            and, if select_by passed as name, default methods. If not
            passed names will take icon_names.

        tooltips
            List of strings to serve as tooltips for each icon button. List
            should have same length as icon_names. Use None to represent
            any icon button is not to have a tooltip.

        tts_kwargs
            (only pass if `tooltips` passed and not relying on default
            tooltip properties for all tooltips)
            List of dictionaries to serve as kwargs for each icon button
            tooltip. List should have same length as icon_names. Use None
            to represent any icon button tooltip which is to use default
            properties. Example for three buttons, the middle one of which
            uses default properties:
                ttw_kwargs = [{'bottom': True, 'color': 'lime'},
                                None,
                                {'right': True,
                                'color': 'white',
                                'content_class': 'black--text'}]

        select_by
            How default selection methods will reference icon buttons.

        allow_no_selection : bool, default: True
            True to allow no selection.
            False to require a button to always be selected.

        initial_selection
            Name of any icon button to be initially selected (or index if
            `select_by` passed as 'index'). Default None if no selection
            permitted, otherwise icon button with index 0.

        deselected_colors
            List of strings representing icon colors when not selected.
            List should have same length as icon_names. Define colors as
            vuetify color definition, for example:
                ['grey darken-4', 'brown darken-4']
            Default all 'grey darken-4'.

        selected_colors
            List of strings representing icon colors when selected. List
            should have same length as icon_names. Define colors as vuetify
            color definition, for example:
                ['grey lighten-4', 'brown lighten-4']
            Default all 'grey lighten-4'.

        handlers_on_selecting
            List of Callables to be executed on icon buttons becoming
            selected. List should have same length as icon_names. Use None
            to represent any icon button that does not have an on selecting
            handler. If not passed then a list of handlers can be
            subsequently assigned to attribute `handlers_on_selecting`.

            Handlers should have following signature to accommodate passed
            parameters:
                handler_name(widget, event, data)

        handlers_on_deselecting
            List of Callables to be executed on icon buttons becoming
            selected. List should have same length as icon_names. Use None
            to represent any icon button that does not have an on
            deselecting handler. If not passed then a list of handlers can
            be subsequently assigned to attribute
            `handlers_on_deselecting`.

            Handlers should have following signature to accommodate passed
            parameters. NB differs from handlers_on_selecting signature:
                handler_name(widget)

        kwargs
            Passed on to IconBut constructor for each instantiated icon
            button.
        """
        self._allow_no_selection = allow_no_selection
        self.handlers_on_selecting = handlers_on_selecting
        self.handlers_on_deselecting = handlers_on_deselecting

        num_icons = len(icon_names)

        names = names if names is not None else icon_names
        tooltips = tooltips if tooltips is not None else [None] * num_icons
        tts_kwargs = tts_kwargs if tts_kwargs is not None else [None] * num_icons
        if deselected_colors is None:
            deselected_colors = ["grey darken-4"] * num_icons
        if selected_colors is None:
            selected_colors = ["grey lighten-4"] * num_icons

        icon_buts = []
        for name, color, d_color, tt, tt_kwargs in zip(
            icon_names, selected_colors, deselected_colors, tooltips, tts_kwargs
        ):
            icon_but = IconBut(
                name,
                tooltip=tt,
                tt_kwargs=tt_kwargs,
                color=color,
                dark_color=d_color,
                dark=True,
                **kwargs,
            )
            icon_but.on_event("click", self._selection_handler)
            icon_buts.append(icon_but)

        if initial_selection is None and not allow_no_selection:
            if select_by == "index":
                initial_selection = 0
            else:
                initial_selection = names[0]

        super().__init__(
            icon_buts,
            names=names,
            select_by=select_by,
            initial_selection=initial_selection,
        )

        contain_me = []
        for icon_but in self:
            if icon_but.tt is not None:
                contain_me.append(icon_but.tt)
            else:
                contain_me.append(icon_but)
        self.contain_me = contain_me

    def _handler(self, handlers, widget, **kwargs):
        if handlers is not None:
            handler = handlers[self.index(widget)]
            if callable(handler):
                handler(widget, **kwargs)

    def _handle_selection(self, widget, event, data):
        self._handler(self.handlers_on_selecting, widget, event=event, data=data)

    def _handle_deselection(self, widget):
        self._handler(self.handlers_on_deselecting, widget)

    def _selection_handler(self, widget, event, data):
        if self._allow_no_selection and self.selected_element == widget:
            self.deselect()
        else:
            self.select_element(widget)
            self._handle_selection(widget, event, data)

    def _add_element_to_selection(self, element):
        super()._add_element_to_selection(element)
        element.lighten()

    def deselect(self):
        was_slctd = self.selected_element
        super().deselect()
        if was_slctd is not None:
            was_slctd.darken()
            self._handle_deselection(widget=was_slctd)


class TabTt(vue.Tab):
    """`vue.Tab` with optional tooltip.

    NB If `tooltip` passed then the tooltip (self.tt), NOT the component
    (self), should be passed to the parent component, For exmaple:
        vue.Card(children=[tab_tt.tt]).

    Attributes
    ----------
    tt: vue.Tooltip | None
        Associated Tooltip (None if `tooltip` not passed to constructor).
    """

    @tooltip_decorator
    def __init__(
        self,
        *args,
        tooltip: str | None = None,
        tt_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tooltip
            Component to be displayed as a tooltip, or text (str) for
            tooltip label, for example tooltip = 'this is my tooltip'.

        tt_kwargs
            kwargs to be passed to any `vue.Tooltip`. Only applicable if
            `tooltip` passed. Exmample:
                tt_kwargs = {'bottom': True, 'color': 'red'}
            kwargs to set position are 'top', 'right', 'bottom', 'left'.
            To see all possible kwargs select v-tooltip at:
                https://vuetifyjs.com/en/components/api-explorer/
        """
        self.tt: vue.Tooltip
        super().__init__(*args, **kwargs)


class Dialog(vue.Overlay):
    """Dialog box.

    NB Does not employ vuetify's Dialog widget. Rather, overlays parent
    with a v-overlay widget which has a v-card as it's child that serves
    as the dialog box. Advantage over v-dialog is that allows to be
    positioned over a specific component rather than occupying full screen.

    Usage
    -----
    Pass instance as child of component to be overlaid and within which
    dialog box to appear.

    Settable Properties
    -------------------
    title
        Dialog box title.

    text
        Dialog box text.
    """

    def __init__(
        self,
        title: str,
        text: str,
        width="450px",
        btn_kwargs: dict | None = None,
        title_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
        card_kwargs: dict | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        title
            Dialog title.

        text
            Dialog text.

        btn_kwargs
            Passed on to close button constructor. Defaults will only be
            overriden if explicitly included to dictionary.

            Defaults:
                'color': 'black'
                'text': True
                'class_': 'ma-2'

        title_kwargs
            Passed on to vue.CardTitle constructor. Defaults will only be
            overriden if explicitly included to dictionary.

            Defaults:
                'style_': 'color: black'
                'class_', 'headline grey lighten-2'

        text_kwargs
            Passed on to vue.CardText constructor. Defaults will only be
            overriden if explicitly included to dictionary.

            Defaults:
                'color': 'black'
                'text': True
                'class_': 'ma-2'

        card_kwargs
            Passed on to vue.Card constructor. Defaults will only be
            overriden if explicitly included to dictionary. NB 'width'
            should be passed independently

            Default:
                'color': 'white'

        width : str, default: '450px'
            Dialog box width.

        kwargs
            Passed on to `vue.Overlay` constructor.

            Defaults:
                'absolute': True
                'color': 'grey'
                'class_': 'd-flex flex-row justify-center'
        """
        kwds = btn_kwargs if btn_kwargs is not None else {}
        self._action = self._create_action(**kwds)
        kwds = title_kwargs if title_kwargs is not None else {}
        self._card_title = self._create_card_title(title, **kwds)
        kwds = text_kwargs if text_kwargs is not None else {}
        self._card_text = self._create_card_text(text, **kwds)
        kwds = card_kwargs if card_kwargs is not None else {}
        kwds["width"] = width
        self._card = self._create_dialog_card(title, text, **kwds)
        kwargs["value"] = False
        kwargs.setdefault("absolute", True)
        kwargs.setdefault("color", "grey")
        kwargs.setdefault("class_", "d-flex flex-row justify-center")
        super().__init__(children=[self._card], **kwargs)

    @property
    def card_title(self) -> vue.CardTitle:
        return self._card_title

    @property
    def card_text(self) -> vue.CardText:
        return self._card_text

    @property
    def card(self) -> vue.Card:
        return self._card

    @property
    def title(self) -> str:
        """Dialog box title. Settable."""
        return self.card_title.children[0]

    @title.setter
    def title(self, value: str):
        self.card_title.children = [value]

    @property
    def text(self) -> str:
        """Dialog box text. Settable."""
        return self.card_text.children[0]

    @text.setter
    def text(self, value: str):
        self.card_text.children = [value]

    def popup(self, title: str | None = None, text: str | None = None):
        """Popup the dialog box.

        title
            New title.

        text
            New text.
        """
        if title is not None:
            self.title = title
        if text is not None:
            self.text = text
        self.value = True

    def close_dialog(self):
        """Close dialog box.

        NB this will not close the connection with the underlying widget,
        for which use `close` method.
        """
        self.value = False

    def _close_handler(self, widget=None, event=None, data=None):
        self.close_dialog()

    def _create_action(self, **kwargs):
        kwargs.setdefault("color", "black")
        kwargs.setdefault("text", True)
        kwargs.setdefault("class_", "ma-2")
        btn = vue.Btn(children=["Close"], **kwargs)
        btn.on_event("click", self._close_handler)
        return vue.CardActions(children=[btn], class_="d-flex flex-row justify-end")

    def _create_card_title(self, title: str, **kwargs):
        kwargs.setdefault("style_", "color: black")
        kwargs.setdefault("class_", "headline grey lighten-2")
        return vue.CardTitle(children=[title], **kwargs)

    def _create_card_text(self, text: str, **kwargs):
        kwargs.setdefault("class_", "mt-2")
        kwargs.setdefault("style_", "color: black")
        return vue.CardText(children=[text], **kwargs)

    def _create_dialog_card(self, title: str, text: str, **kwargs):
        kwargs.setdefault("color", "white")
        contents = [self.card_title, self.card_text, self._action]
        return vue.Card(children=contents, **kwargs)
