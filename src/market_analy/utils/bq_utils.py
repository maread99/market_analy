"""Utility constants, functions and classes for bqplot."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from itertools import cycle
from typing import Literal

import bqplot as bq
import pandas as pd
import numpy as np
from bqplot.interacts import FastIntervalSelector
from traitlets import HasTraits, Any, dlink, link

from market_analy.utils import UTC

# possible keys for dictionary taken by +scales+ parameter of Axes class.
ScaleKeys = Literal["x", "y", "color", "opacity", "size", "rotation", "skew"]

# Colors for use on charts with dark backgrounds.
COLORS_DARK_8 = [
    "yellow",
    "springgreen",
    "hotpink",
    "darkorange",
    "lightskyblue",
    "burlywood",
    "red",
    "mediumorchid",
]

LEGEND_LOCATIONS = [
    "top-left",
    "top",
    "top-right",
    "right",
    "bottom-right",
    "bottom",
    "bottom-left",
    "left",
]


def dates_to_posix(dates: pd.DatetimeIndex) -> np.ndarray:
    """Dates as posix.

    Can pass return (as list) to bqplot OrdinalScale domain.

    Notes
    -----
    Only useful if creating a scale of discontinuous dates. If dates are
    continuous, use DateScale.
    """
    return dates.values.astype(np.int64) // 10**6


def discontinuous_date_to_timestamp(value: np.int64) -> pd.Timestamp:
    """Convert integer representing posix to pd.Timestamp.

    Parameters
    ----------
    value
        Value of an OrdinalScale representing a date which require as
        pd.Timestamp.
    """
    # utc version of fromtimestamp ensures that operation is the inverse
    # of dates_to_posix
    return pd.Timestamp.fromtimestamp(value / 1000, tz=UTC).tz_convert(None)


def format_label(value: Any, label_format: str) -> str:
    """Format `value` for use as a label.

    Parameters
    ----------
    value
        Value as a value of a mark's 'x' or 'y' attribute.

    Returns
    -------
        `value` formatted in accordance with `label_format`.
    """
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.to_datetime(value).strftime(label_format)
    if isinstance(value, (np.integer, np.floating)):
        try:
            f_str = "{:" + label_format + "}"
            return f_str.format(value)
        except ValueError:
            return f_str.format(int(value))
    msg = "format_label does not support value type " + str(type(value))
    raise TypeError(msg)


class _LevelLine(bq.Lines):
    """Create line at specific level along a scale.

    Creates line on `figure` in `direction` at `level` based on either
    `scale`['y'] or `scale`['x'] (as corresponds with `direction`). (Takes
    full scales as dict for convenience.)

    Parameters
    ----------
    draw_to_figure:
        True to draw line to `figure`.

    kwargs:
        Passed to `Lines` constructor.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        direction: Literal["vertical", "horizontal"] = "vertical",
        basis: Literal["figure", "axis"] = "figure",
        start: Any = 0,
        end: Any = 1,
        draw_to_figure=True,
        **kwargs,
    ):
        kwargs.setdefault("colors", ["Yellow"])
        kwargs.setdefault("stroke_width", 1)
        kwargs.setdefault("preserve_domain", {"y": True, "x": True})

        by_fig = basis == "figure"
        if vertical := (direction == "vertical"):
            scales = {"x": scales["x"], "y": figure.scale_y if by_fig else scales["y"]}
        else:
            scales = {"y": scales["y"], "x": figure.scale_x if by_fig else scales["x"]}

        super().__init__(
            x=[level, level] if vertical else [start, end],
            y=[start, end] if vertical else [level, level],
            scales=scales,
            **kwargs,
        )

        if draw_to_figure:
            self._draw_to_figure(figure)

        # NOTES FOR POSSIBLE EXTENDING TO ACCOMODATE MULTIPLE LINES
        # pyplot script for vline...might want to consider using arrays and
        # column_stack if going to accomodate multiple lines
        #     level = np.array(level)
        #     if len(level.shape) == 0:
        #         x = [level, level]
        #         y = [0, 1]
        #     else:
        #         x = np.column_stack([level, level])
        #         y = [[0, 1]] * len(level)

    def _draw_to_figure(self, figure: bq.Figure):
        figure.marks = list(figure.marks) + [self]


class HLevelLine(_LevelLine):
    """Horizontal Line at specific level along a scale."""

    def __init__(
        self,
        level: Any,
        scales: dict,
        figure: bq.Figure,
        basis: Literal["figure", "axis"] = "figure",
        start: Any = 0,
        end: Any = 1,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        level
            level of y-scale at which to draw line.

        scales: dictionary of Scale objects
            Must include key 'y' with value as Scale which `level`
                references.

        figure
            Figure to which line to be drawn.

        basis : Literal["figure", "axis"], default: "figure"
            Basis for defining `start` and `end`.

        start : Any, default: 0
            Value from which line should start.

            If `basis` is "figure" then as a value between 0 and 1
            describing a proportion of figure from left side. For example,
            0.25 would start line at one quarter of the way across the
            figure.

            If `basis` is "axis" then a value described by the x-axis.

        end : Any, default: 1
            Value from which line should end.

            If `basis` is "figure" then as a value between 0 and 1
            describing a proportion of figure from left side. For example,
            0.75 would end line at three quarters of the way across the
            figure.

            If `basis` is "axis" then a value described by the x-axis.

        **kwargs: passed on to bqplot.Line.
        """
        super().__init__(
            level, scales, figure, "horizontal", basis, start, end, **kwargs
        )


class VLevelLine(_LevelLine):
    """Vertical Line at specific level along a scale."""

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        basis: Literal["figure", "axis"] = "figure",
        start: Any = 0,
        end: Any = 1,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        level:
            level of x-scale at which to draw line.

        scales: dictionary of Scale objects
            Must include key 'x' with value as Scale which `level`
                references.

        figure:
            Figure to which line to be drawn.

        basis : Literal["figure", "axis"], default: "figure"
            Basis for defining `start` and `end`.

        start : Any, default: 0
            Value from which line should start.

            If `basis` is "figure" then as a value between 0 and 1
            describing a proportion of figure from bottom side. For
            example, 0.25 would start line at one quarter of the way up
            the figure.

            If `basis` is "axis" then a value described by the y-axis.

        end : Any, default: 1
            Value from which line should end.

            If `basis` is "figure" then as a value between 0 and 1
            describing a proportion of figure from bottom side. For
            example, 0.75 would end line at three quarters of the way up
            the figure.

            If `basis` is "axis" then a value described by the y-axis.

        **kwargs: passed on to bqplot.Line.
        """
        super().__init__(level, scales, figure, "vertical", basis, start, end, **kwargs)


class _LabeledLevelLine(_LevelLine):
    """Labeled Line at specific level along a scale.

    Label indicating level along scale placed alongside axis.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        label_format: str | None = None,
        label_kwargs: dict | None = None,
        direction: Literal["vertical", "horizontal"] = "vertical",
        side: Literal["greater", "lesser"] = "greater",
        draw_to_figure=True,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        All parameters as for _LevelLine save for:

        label_format: str
            Format to apply to label. Can (intended to) accept any format
            string supported by the .tick_format attribute of an Axis
            object. If not passed will display label as value
            corresponding with the line's level along the scale.

        label_kwargs: dict
            Passed to Label constructor.

        side: str. Literal['greater', 'lesser']
            Determines if label is placed on the 'greater' or 'lesser' side
            of the line, relative to the line level.
        """
        super().__init__(
            level, scales, figure, direction, draw_to_figure=False, **kwargs
        )

        self._direction = direction
        self._label_format = label_format

        label_kwargs = label_kwargs if label_kwargs is not None else {}
        label_kwargs.setdefault("colors", ["Yellow"])
        label_kwargs.setdefault("font_weight", "normal")
        label_kwargs.setdefault("preserve_domain", {"y": True, "x": True})
        label_kwargs.setdefault("y_offset", -10)
        label_kwargs.setdefault("x_offset", 3)
        self.label = bq.Label(
            x=self.x[:1],
            y=self.y[:1],
            scales=self.scales,
            text=["placeholder"],
            **label_kwargs,
        )

        self._side: Literal["greater", "lesser"]
        self.side = side

        attr = "x" if self._direction == "vertical" else "y"
        dlink((self, attr), (self.label, attr), lambda v: v[:1])
        dlink((self, attr), (self.label, "text"), self._format_label)

        if draw_to_figure:
            self._draw_to_figure(figure)

    def _set_side_v(self):
        """Set parameters for a label associated with a vertical line."""
        self.label.x_offset = -3 if self.side == "lesser" else 3
        self.label.align = "end" if self.side == "lesser" else "start"

    def _set_side_h(self):
        """Set parameters for a label associated with a horizontal line."""
        self.label.y_offset = 10 if self.side == "lesser" else -10

    @property
    def side(self) -> Literal["greater", "lesser"]:
        return self._side

    @side.setter
    def side(self, value: Literal["greater", "lesser"] = "greater"):
        """Set side of line on which label to be placed.

        Parameters
        ----------
        value: str Literal['greater', 'lesser']
            Set side of line to place label, relative to line's level.
        """
        assert value in ["greater", "lesser"]
        self._side = value
        if self._direction == "vertical":
            self._set_side_v()
        elif self._direction == "horizontal":
            self._set_side_h()

    def change_side(self):
        """Change side of the line that the label is placed."""
        self.side = "greater" if self.side == "lesser" else "lesser"

    def _format_label(self, value: list) -> list:
        """Format `value` for use as a label.

        Converter for dlink between self.x or self.y and label's text.

        Parameters
        ----------
        value
            Either self.x or self.y, i.e. line's level along the scale

        Returns
        -------
            `value` formatted in accordance with `self.label_format`.
        """
        if self._label_format is None:
            return value[:1]
        return [format_label(value[0], self._label_format)]

    def _draw_to_figure(self, figure: bq.Figure):
        figure.marks = list(figure.marks) + [self, self.label]


class HLabeledLevelLine(_LabeledLevelLine):
    """Horizontal Line at specific level along a scale.

    Label indicating level along scale placed alongside axis.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        label_format: str | None = None,
        label_kwargs: dict | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        All parameters as for HLevelLine save for:

        label_format: str
            Format to apply to label. Can (intended to) accept any format
                string supported by the .tick_format attribute of an Axis
                object. If not passed will display label as value
                corresponding with the line's level along the scale.

        label_kwargs: dict
            Passed to `bqplot.Label` constructor.
        """
        super().__init__(
            level,
            scales,
            figure,
            label_format,
            label_kwargs,
            direction="horizontal",
            **kwargs,
        )


class VLabeledLevelLine(_LabeledLevelLine):
    """Vertical Line at specific level along a scale.

    Label indicating level along scale placed alongside axis.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        label_format: str | None = None,
        label_kwargs: dict | None = None,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        All parameters as for VLevelLine save for:

        label_format: str
            Format to apply to label. Can (intended to) accept any format
                string supported by the .tick_format attribute of an Axis
                object. If not passed will display label as value
                corresponding with the line's level along the scale.

        label_kwargs: dict
            Passed to `bqplot.Label` constructor.
        """
        super().__init__(
            level,
            scales,
            figure,
            label_format,
            label_kwargs,
            direction="vertical",
            **kwargs,
        )


class Crosshair(HasTraits):
    """Add draggable crosshair to a figure.

    Requires figure to have an existing Mark drawn to it.

    Features:
        Centre of Crosshair can be dragged around figure (optional)
        Crosshairs extend to both horizontal and vertical axis.
        Labels alongside axes describe level of corresponding crosshair line.

    Attributes
    ----------
    vhair:
        VlabeledLevelLine that forms vertical hair.

    hhair:
        HlabeledLevelLine that forms horizontal hair.

    cross:
        Scatter mark that represents cross.
    """

    _y_center = Any()
    _x_center = Any()

    def __init__(
        self,
        figure: bq.Figure,
        existing_mark: bq.Mark | None = None,
        x=None,
        y=None,
        color="Yellow",
        opacity=1.0,
        line_style="solid",
        line_kwargs: dict | None = None,
        label_kwargs: dict | None = None,
        draggable=True,
        override_figure_interaction=True,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        figure:
            Figure to which line to be drawn.

        existing_mark:
            Mark already drawn to figure and hence included to
            `figure.marks`. Crosshair will read x and y levels against the
            scales that the existing mark is plotted against. Also,
            crosshair will assume horizontal center of chart as coinciding
            with median value of `existing_mark.x`.

        x:
            x level at which to place crosshair. If not passed then will place
            at assumed horizontal center.

        y:
            y level at which to place crosshair. If not passed then will place
            at assumed vertical center.

        line_kwargs: dict
            Passed to `bq.Lines` to create hairs.

        label_kwargs: dict
            Passed to `bq.Label` to create hair labels alongside axis.

        kwargs:
            Passed on to `bqplot.Scatter` to create cross.

        color: str
            Color of all elements of Crosshair. NB will override any
            'colors' argument included to `line_kwargs`, `label_kwargs` or
            `kwargs`.

        opacity: float
            Opacity of all elements of Crosshair. NB will override any
            of the muddle of possible opacities arguments included to
            `line_kwargs`, `label_kwargs` or `kwargs`.

        line_style: str
            Line style of the hairs. NB will override any 'line_style'
            argument included to `line_kwargs`.

        draggable: bool
            True to enable dragging of the crosshair via the scatter mark.

        override_figure_interaction: bool
            True to remove any existing figure interaction in order that
            crosshair will be visible.
        """
        self.figure = figure
        self._color = [color]
        self._opacity = [opacity]
        self._line_style = line_style

        self._existing_mark = (
            existing_mark if existing_mark is not None else figure.marks[0]
        )

        dlink((self._existing_mark, "x"), (self, "_x_center"), self._get_x_center)
        dlink((self._existing_mark, "y"), (self, "_y_center"), self._get_y_center)

        self.scales = self._existing_mark.scales

        line_kwargs = line_kwargs if line_kwargs is not None else {}
        label_kwargs = label_kwargs if label_kwargs is not None else {}
        for kws in [kwargs, line_kwargs, label_kwargs]:
            kws["colors"] = self._color
            kws.setdefault("preserve_domain", {"y": True, "x": True})

        start_x = x if x is not None else self._x_center
        start_y = y if y is not None else self._y_center

        kwargs.setdefault("marker", "cross")
        self.cross = bq.Scatter(
            x=[start_x],
            y=[start_y],
            scales=self.scales,
            enable_move=draggable,
            update_on_move=True,
            **kwargs,
        )

        figure.marks = list(figure.marks) + [self.cross]

        line_kwargs["line_style"] = self._line_style
        if x is not None and "side" not in line_kwargs:
            line_kwargs["side"] = "lesser" if x > self._x_center else "greater"
        self.vhair = VLabeledLevelLine(
            level=start_x,
            scales=self.scales,
            figure=figure,
            label_format=self._x_tick_format,
            label_kwargs=label_kwargs,
            **line_kwargs,
        )
        if y is not None and "side" not in line_kwargs:
            line_kwargs["side"] = "lesser" if y > self._y_center else "greater"
        self.hhair = HLabeledLevelLine(
            level=start_y,
            scales=self.scales,
            figure=figure,
            label_format=self._y_tick_format_python_format,
            label_kwargs=label_kwargs,
            **line_kwargs,
        )

        dlink((self.cross, "y"), (self.hhair, "y"), lambda y: np.concatenate((y, y)))
        dlink((self.cross, "x"), (self.vhair, "x"), lambda x: np.concatenate((x, x)))

        self.cross.observe(self._side_x_handler, ["x"])
        self.cross.observe(self._side_y_handler, ["y"])

        self._components = [
            self.cross,
            self.hhair,
            self.vhair,
            self.hhair.label,
            self.vhair.label,
        ]

        self.opacity = opacity

        if override_figure_interaction:
            figure.interaction = None

    def _tick_format(self, orientation: Literal["horizontal", "vertical"]):
        for axis in self.figure.axes:
            if axis.orientation == orientation:
                return axis.tick_format

    @property
    def _x_tick_format(self):
        return self._tick_format("horizontal")

    @property
    def _y_tick_format(self):
        return self._tick_format("vertical")

    @property
    def _y_tick_format_python_format(self):
        return self._y_tick_format.replace("r", "g")

    def _get_x_center(self, x: list):
        """Mid-point of x axis.

        Parameters
        ----------
        x: list
            Center assumed as coinciding with 'x' attribute of middle mark
            of list of existing (original) marks.
        """
        return x[len(x) // 2]

    def _get_y_center(self, y: list):
        """Mid-point of y axis

        y: list
            Center assumed as coinciding with 'y' attribute of middle mark
            of list of existing marks.
        """
        if isinstance(self._existing_mark, bq.OHLC):
            y = y[:, -1:].flatten()  # close prices
        y_min = y.min()
        return y_min + (y.max() - y_min) / 2

    def _side_x_handler(self, x: dict):
        """Change side of vhair label if vhair has crossed x scale mid-point."""
        x_center = self._x_center
        was_on_right = x["old"][0] > x_center
        now_on_right = x["new"][0] > x_center
        if was_on_right != now_on_right:
            self.vhair.change_side()

    def _side_y_handler(self, y: dict):
        """Change side of hhair label if vhair has crossed y scale mid-point."""
        if (y["old"][0] > self._y_center) != (y["new"][0] > self._y_center):
            self.hhair.change_side()

    @property
    def components(self) -> list:
        """List of all Mark objects that comprise crosshair."""
        return self._components

    @property
    def color(self) -> str:
        """Crosshair color, for example 'yellow'."""
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = [color]
        for mark in self.components:
            mark.colors = self._color

    @property
    def opacity(self) -> list[float]:
        """Crosshair opacity, for example 0.5."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        self._opacity = [opacity]
        self.cross.opacities = self._opacity
        for hair in [self.vhair, self.hhair]:
            hair.opacities = self._opacity
            hair.label.opacities = self._opacity
            hair.label.opacity = self._opacity

    @property
    def line_style(self) -> str:
        """Hairs line style, for example 'dashed'."""
        return self._line_style

    @line_style.setter
    def line_style(self, style: Literal["solid", "dashed", "dotted", "dash_dotted"]):
        self._line_style = style
        for hair in [self.vhair, self.hhair]:
            hair.line_style = self._line_style

    def change_side(self):
        """Change side of labels."""
        self.vhair.change_side()
        self.hhair.change_side()

    def close(self):
        """Remove crosshair from frontend."""
        self.figure.marks = [m for m in self.figure.marks if m not in self._components]
        for comp in self._components:
            comp.close()


class Crosshairs(list):
    """List of crosshairs with collective functionality."""

    class CrosshairsCh(Crosshair):
        """Crosshair that can be included to Crosshairs."""

        def __init__(self, crosshairs, **kwargs):
            """Constructor.

            Parameters
            ----------
            crosshairs: Crosshairs
                Instance of Crosshairs to which crosshair to be added.

            **kwargs:
                Passed to Crosshair contstructor.
            """
            self.crosshairs = crosshairs
            super().__init__(**kwargs)

        def close(self):
            self.crosshairs.remove(self)
            super().close()

    def __init__(self, figure: bq.Figure, colors: Iterable[str] | None = None):
        """Constructor.

        Parameters
        ----------
        figure: Figure
            bqplot Figure to which crosshairs to be added.

        colors: Iterable
            Sequence of colours, as defined by bqplot, to determine default
            colour of consecutive crosshairs. For example, ['red',
            'green', 'blue'] will colour first four added crosshairs (if
            'color' not otherwise passed to --add()--) as red, green,
            blue, red.
        """
        self.figure = figure
        self.colors = cycle(colors) if colors is not None else cycle(COLORS_DARK_8)

    def add(self, **kwargs) -> Crosshair:
        """Add crosshair.

        Parameters
        ----------
        kwargs:
            Passed to `Crosshair` constructor.

        Notes
        -----
        Crosshairs should ONLY be added using this method.
        """
        kwargs.setdefault("color", next(self.colors))
        kwargs["figure"] = self.figure
        ch = self.CrosshairsCh(crosshairs=self, **kwargs)
        self.append(ch)
        return ch

    def close(self):
        """Close all crosshairs.

        NB Close a specific crosshair with the crosshair's .close() method.
        """
        for ch in copy(self):
            ch.close()

    def opacity(self, value: float):
        """Set opacity of all crosshairs."""
        for ch in self:
            ch.opacity = value

    def opaque(self):
        """Make all crosshairs fully opaque."""
        self.opacity(value=1.0)

    def fade(self):
        """Fade all crosshairs."""
        self.opacity(0.5)

    def highlight_color(self, color: str):
        """Highlight all crosshairs of a certain color."""
        self.fade()
        for ch in self:
            if ch.color[0] == color:
                ch.opacity = 1.0

    def bring_to_front(self, crosshair: Crosshair):
        """Bring spicific `crosshair` to front."""
        crosshairs_components = crosshair.components
        for ch in self:
            for component in ch.components:
                if component not in crosshair.components:
                    crosshairs_components.append(component)
        non_crosshair = [m for m in self.figure.marks if m not in crosshairs_components]
        self.figure.marks = non_crosshair + crosshairs_components


class HFixedRule:
    """Add a draggable fixed-length horizontal rule to a figure.

    Features:
        Rule can be dragged via a marker (optional).
        Labeled at extremes to describe x-axis range covered by rule.

    Parameters
    ----------
    level
        level of y-scale at which to draw line.

    scales: dictionary of Scale objects
        Must include key 'y' with value as Scale which `level`
        references and key 'x' with value as Scale within which `start`
        is represented.

    figure
        Figure to which line to be drawn.

    start
        Value from which line should start.

    length
        Length of the rule.

        If `scales["x"]` is a `bq.OrdinalScale` then pass as an int
        representing the number of x-ticks that the rule is to cover.

        If `scales["x"]` is a `bq.LinearScale` then pass as an int or
        float defining the x-axis distance that the rule is to cover.

        NB Not other type of scale is supported.

    color: str
        Color of all rule elements. Color of any particular component
            can be overriden by including `colors` to the corresponding
            kwargs argument, for example `label_kwargs`.

    opacity: float
        Opacity of all elements of Crosshair. Will override any opacity
        defined via kwargs for any specific component.

    draggable: bool
        True to enable dragging of rule via the scatter mark.

    oridinal_values: list[Any] | None
        If `scales["x"]` is `bq.OrdinalScale` then pass axis values as
        a list in same terms `start`. For example, if scale represents
        discontinuous dates then pass a list of the plottable dates.

        If not passed then will attempt to find values in
        `scales["x"].domain`.

    label_kwargs: dict | None
        Passed to Label constructor for each label. The following
        kwargs will be ignored if passed:
            'x', 'y', 'scales', 'text', 'y_offset', 'x_offset', 'align'

    scat_kwargs: dict | None
        Passed to Scatter constructor to create mark via which rule
        can be dragged. Ignored if not `draggable`.

    **kwargs:
        Passed on to bq.Lines to create horizontal line.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: bq.Figure,
        start: Any,
        length: int | float,
        color: str = "yellow",
        opacity: float = 1.0,
        draggable: bool = True,
        ordinal_values: list[Any] | None = None,
        label_kwargs: dict | None = None,
        scat_kwargs: dict | None = None,
        **kwargs,
    ):
        self.figure = figure
        self._scale = scales["x"]
        if not isinstance(self._scale, (bq.OrdinalScale, bq.LinearScale)):
            raise TypeError(f"'x' scale type {type(self._scale)} is not supported.")

        if not isinstance(self._scale, bq.OrdinalScale):
            self._ordinal_values = None
        elif ordinal_values is not None:
            self._ordinal_values = ordinal_values
        else:
            self._ordinal_values = self._scale.domain

        self._length = length
        self._color = [color]
        self._opacity = [opacity]

        label_kwargs = label_kwargs if label_kwargs is not None else {}
        scat_kwargs = scat_kwargs if scat_kwargs is not None else {}
        for kws in [kwargs, scat_kwargs, label_kwargs]:
            kws["colors"] = self._color
            kws["opacities"] = self._opacity
            kws.setdefault("preserve_domain", {"y": True, "x": True})

        kwargs.setdefault("stroke_width", 1)

        end = self._get_end_from_start(start)
        # Create line
        self.line = HLevelLine(
            level, scales, figure, "axis", start, end, draw_to_figure=False, **kwargs
        )

        # Create labels
        self._label_format = None
        for axis in figure.axes:
            if axis.orientation == "horizontal":
                self._label_format = axis.tick_format

        label_y_offset = kwargs["stroke_width"] + 10
        label_x_offset = 5

        label_kwargs.setdefault("opacity", self._opacity)
        label_kwargs.setdefault("font_weight", "normal")
        label_kwargs.setdefault("preserve_domain", {"y": True, "x": True})

        self.label_l = bq.Label(
            x=self.line.x[:1],
            y=self.line.y[:1],
            scales=scales,
            text=["placeholder"],
            y_offset=label_y_offset,
            x_offset=label_x_offset,
            align="start",
            **label_kwargs,
        )
        dlink((self.line, "y"), (self.label_l, "y"), lambda v: v[:1])
        dlink((self.line, "x"), (self.label_l, "x"), lambda v: v[:1])
        dlink((self.line, "x"), (self.label_l, "text"), self._format_label_l)

        self.label_r = bq.Label(
            x=self.line.x[-1:],
            y=self.line.y[:1],
            scales=scales,
            text=["placeholder"],
            y_offset=-label_y_offset,
            x_offset=-label_x_offset,
            align="end",
            **label_kwargs,
        )
        dlink((self.line, "y"), (self.label_r, "y"), lambda v: v[:1])
        dlink((self.line, "x"), (self.label_r, "x"), lambda v: v[-1:])
        dlink((self.line, "x"), (self.label_r, "text"), self._format_label_r)

        self._components = [
            self.line,
            self.label_l,
            self.label_r,
        ]

        if not draggable:
            self.grip_l, self.grip_r = None, None
            self._draw_to_figure(figure)
            return

        scat_kwargs.setdefault("marker", "ellipse")
        self.grip_r = bq.Scatter(
            x=self.line.x[-1:],
            y=self.line.y[:1],
            scales=scales,
            enable_move=True,
            update_on_move=True,
            **scat_kwargs,
        )
        dlink((self.grip_r, "y"), (self.line, "y"), lambda y: np.concatenate((y, y)))
        dlink((self.grip_r, "x"), (self.line, "x"), self._get_line_x_if_move_grip_r)

        self.grip_l = bq.Scatter(
            x=self.line.x[:1],
            y=self.line.y[:1],
            scales=scales,
            enable_move=True,
            update_on_move=True,
            **scat_kwargs,
        )
        dlink((self.grip_l, "x"), (self.line, "x"), self._get_line_x_if_move_grip_l)

        link((self.grip_r, "y"), (self.grip_l, "y"))  # bi-directional link
        r_to_l_lnk = dlink(
            (self.grip_r, "x"), (self.grip_l, "x"), self._get_grip_l_from_grip_r
        )
        l_to_r_lnk = dlink(
            (self.grip_l, "x"), (self.grip_r, "x"), self._get_grip_r_from_grip_l
        )

        def grip_r_on_drag_start_handler(*_):
            try:
                l_to_r_lnk.unlink()
            except ValueError:  # not linked
                pass
            r_to_l_lnk.link()

        def grip_l_on_drag_start_handler(*_):
            try:
                r_to_l_lnk.unlink()
            except ValueError:  # not linked
                pass
            l_to_r_lnk.link()

        self.grip_r.on_drag_start(grip_r_on_drag_start_handler)
        self.grip_l.on_drag_start(grip_l_on_drag_start_handler)

        self._components.append(self.grip_r)
        self._components.append(self.grip_l)

        self._draw_to_figure(figure)

    def _format_label_l(self, value: list) -> list:
        """Format `value` for use as the left label.

        Converter for dlink between self.x and label's text.
        """
        if self._label_format is None:
            return value[:1]
        return [format_label(value[0], self._label_format)]

    def _format_label_r(self, value: list) -> list:
        """Format `value` for use as the right label.

        Converter for dlink between self.x and label's text.
        """
        if self._label_format is None:
            return value[-1:]
        return [format_label(value[-1], self._label_format)]

    def _draw_to_figure(self, figure: bq.Figure):
        figure.marks = list(figure.marks) + self.components

    def _get_end_from_start(self, start: Any) -> Any:
        """Get x value for end of line given start value."""
        if isinstance(self._scale, bq.LinearScale):
            return start + self._length
        i_start = self._ordinal_values.index(start)
        i_end = min(i_start + self._length - 1, len(self._ordinal_values) - 1)
        return self._ordinal_values[i_end]

    def _get_start_from_end(self, end: Any) -> Any:
        """Get x value for start of line given end value."""
        if isinstance(self._scale, bq.LinearScale):
            return end - self._length
        i_end = self._ordinal_values.index(end)
        i_start = max(0, i_end - self._length + 1)
        return self._ordinal_values[i_start]

    def _get_line_x_if_move_grip_r(self, grip_r_x: Any) -> np.ndarray:
        """Return value to set `line.x` to for a given grip_r`."""
        end = grip_r_x[0]
        start = self._get_start_from_end(end)
        return np.array((start, end))

    def _get_line_x_if_move_grip_l(self, grip_l_x: Any) -> np.ndarray:
        """Return value to set `line.x` to for a given grip_l`."""
        start = grip_l_x[0]
        end = self._get_end_from_start(start)
        return np.array((start, end))

    def _get_grip_l_from_grip_r(self, grip_r_x: Any) -> np.ndarray:
        """Return value to set `grip_l.x` to for a given grip_r`."""
        return np.array([self._get_line_x_if_move_grip_r(grip_r_x)[0]])

    def _get_grip_r_from_grip_l(self, grip_l_x: Any) -> np.ndarray:
        """Return value to set `grip_r.x` to for a given grip_l`."""
        return np.array([self._get_line_x_if_move_grip_l(grip_l_x)[-1]])

    @property
    def components(self) -> list:
        """List of all Mark objects that comprise rule."""
        return self._components

    @property
    def color(self) -> str:
        """Rule color, for example 'yellow'."""
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = [color]
        for mark in self.components:
            mark.colors = self._color

    @property
    def opacity(self) -> list[float]:
        """Rule opacity, for example 0.5."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        self._opacity = [opacity]
        for mark in self.components:
            mark.opacities = self._opacity
            if isinstance(mark, bq.Label):
                mark.opacity = self._opacity

    def close(self):
        """Remove rule from frontend."""
        self.figure.marks = [m for m in self.figure.marks if m not in self._components]
        for comp in self._components:
            comp.close()


class TrendRule:
    """Add a draggable fixed-length rule representing a trend boundary.

    Features:
        Rule can be dragged via a marker (optional).
        Labeled at extremes to describe x-axis range covered by rule.

    Parameters
    ----------
    x
        x value from which line should start.

    y
        y value from which line should start.

    length
        Initial length of the rule, as int representing the number of
        x-ticks that the rule is to cover.

    factors
        Array of length `length` that will be multiplied by `y` to give
        line's y values.

    scales: dictionary of Scale objects
        Must include keys 'y' and 'x'.

        `scales['x']` must be a `bq.OrdinalScale` in which x is
        represented.

        `scales['y']` must be a `bq.LogScale` or `bq.LinearScale`.

    ordinal_values: list[Any]
        x-xis values as list in same terms `x`. For example, if scale
        represents discontinuous dates then pass a list of the plottable
        dates.

    figure
        Figure to which line to be drawn.

    color: str
        Color of all rule elements. Color of any particular component
            can be overriden by including `colors` to the corresponding
            kwargs argument, for example `label_kwargs`.

    opacity: float
        Opacity of all elements of Crosshair. Will override any opacity
        defined via kwargs for any specific component.

    draggable: bool
        True to enable dragging of rule via the scatter mark.

    label_kwargs: dict | None
        Passed to Label constructor for each label. The following
        kwargs will be ignored if passed:
            'x', 'y', 'scales', 'text', 'y_offset', 'x_offset', 'align'

    scat_kwargs: dict | None
        Passed to Scatter constructor to create mark via which rule
        can be dragged. Ignored if not `draggable`.

    **kwargs:
        Passed on to bq.Lines to create line.
    """

    def __init__(
        self,
        x: np.datetime64,
        y: float | int,
        length: int,
        factors: np.ndarray,
        scales: dict,
        ordinal_values: list[Any],
        figure: bq.Figure,
        color: str = "yellow",
        opacity: float = 1.0,
        draggable: bool = True,
        label_kwargs: dict | None = None,
        scat_kwargs: dict | None = None,
        **kwargs,
    ):
        self.figure = figure
        if not isinstance(scales["x"], bq.OrdinalScale):
            raise TypeError(
                f"'x' scale type must be bq.OrdinalScale (not {type(scales['x'])})."
            )

        self._ordinal_values = ordinal_values
        self._fctrs = factors
        self._color = [color]
        self._opacity = [opacity]

        label_kwargs = label_kwargs if label_kwargs is not None else {}
        scat_kwargs = scat_kwargs if scat_kwargs is not None else {}
        for kws in [kwargs, scat_kwargs, label_kwargs]:
            kws["colors"] = self._color
            kws["opacities"] = self._opacity
            kws.setdefault("preserve_domain", {"y": True, "x": True})

        kwargs.setdefault("stroke_width", 5)

        xs = self._get_x_from_start(x, length)
        ys = self._get_y_from_start(y, length)
        # Create line
        self.line = bq.Lines(x=xs, y=ys, scales=scales, **kwargs)

        # Create labels
        self._label_format = None
        for axis in figure.axes:
            if axis.orientation == "horizontal":
                self._label_format = axis.tick_format

        neg = ys[-1] < ys[0]
        label_y_offset = kwargs["stroke_width"] + 10
        label_x_offset = 3

        label_kwargs.setdefault("opacity", self._opacity)
        label_kwargs.setdefault("font_weight", "normal")
        label_kwargs.setdefault("preserve_domain", {"y": True, "x": True})

        self.label_l = bq.Label(
            x=self.line.x[:1],
            y=self.line.y[:1],
            scales=scales,
            text=["placeholder"],
            y_offset=label_y_offset * (-1 if neg else 1),
            x_offset=label_x_offset,
            align="start",
            **label_kwargs,
        )
        dlink((self.line, "y"), (self.label_l, "y"), lambda v: v[:1])
        dlink((self.line, "x"), (self.label_l, "x"), lambda v: v[:1])
        dlink((self.line, "x"), (self.label_l, "text"), self._format_label_l)

        self.label_r = bq.Label(
            x=self.line.x[-1:],
            y=self.line.y[-1:],
            scales=scales,
            text=["placeholder"],
            y_offset=label_y_offset * (1 if neg else -1),
            x_offset=-label_x_offset,
            align="end",
            **label_kwargs,
        )
        dlink((self.line, "y"), (self.label_r, "y"), lambda v: v[-1:])
        dlink((self.line, "x"), (self.label_r, "x"), lambda v: v[-1:])
        dlink((self.line, "x"), (self.label_r, "text"), self._format_label_r)

        self._components = [
            self.line,
            self.label_l,
            self.label_r,
        ]

        if not draggable:
            self.grip_l, self.grip_r = None, None
            self._draw_to_figure(figure)
            return

        scat_kwargs.setdefault("marker", "ellipse")
        self.grip_r = bq.Scatter(
            x=self.line.x[-1:],
            y=self.line.y[-1:],
            scales=scales,
            enable_move=True,
            update_on_move=True,
            **scat_kwargs,
        )
        dlink((self.grip_r, "x"), (self.line, "x"), self._get_x_from_end)
        dlink(
            (self.grip_r, "x"),
            (self.grip_r, "y"),
            lambda _: self._get_y_from_start()[-1:],
        )
        dlink((self.grip_r, "y"), (self.line, "y"), lambda _: self._get_y_from_start())

        self.grip_l = bq.Scatter(
            x=self.line.x[:1],
            y=self.line.y[:1],
            scales=scales,
            enable_move=True,
            update_on_move=True,
            **scat_kwargs,
        )
        dlink((self.grip_l, "x"), (self.line, "x"), self._get_x_from_start)
        dlink((self.grip_l, "y"), (self.line, "y"), self._get_y_from_start)

        dlink(
            (self.grip_l, "x"),
            (self.grip_r, "x"),
            lambda x: self._get_x_from_start(x)[-1:],
        )
        dlink(
            (self.grip_l, "y"),
            (self.grip_r, "y"),
            lambda y: self._get_y_from_start(y)[-1:],
        )

        self._components.append(self.grip_r)
        self._components.append(self.grip_l)

        self._draw_to_figure(figure)

    def _format_label_l(self, value: list) -> list:
        """Format `value` for use as the left label.

        Converter for dlink between self.x and label's text.
        """
        if self._label_format is None:
            return value[:1]
        return [format_label(value[0], self._label_format)]

    def _format_label_r(self, value: list) -> list:
        """Format `value` for use as the right label.

        Converter for dlink between self.x and label's text.
        """
        if self._label_format is None:
            return value[-1:]
        return [format_label(value[-1], self._label_format)]

    def _draw_to_figure(self, figure: bq.Figure):
        figure.marks = list(figure.marks) + self.components

    def _get_x_from_start(
        self,
        x: np.datetime64,
        length: int | None = None,
    ) -> list[np.datetime64]:
        """Get x values from leftmost x value."""
        start = self._ordinal_values.index(x)
        if length is None:
            length = len(self.line.x)
        stop = min(start + length, len(self._ordinal_values))
        return self._ordinal_values[start:stop]

    def _get_x_from_end(self, x: np.datetime64) -> list[np.datetime64]:
        """Get x values from rightmost x value."""
        stop = self._ordinal_values.index(x[0]) + 1
        start = self._ordinal_values.index(self.line.x[0])
        return self._ordinal_values[start:stop]

    def _extended_fctrs(self, length: int) -> np.ndarray:
        fctrs = self._fctrs
        incr = fctrs[-1] - fctrs[-2]
        n = length - len(fctrs)
        ext = [fctrs[-1] + (incr * i) for i in range(1, n + 1)]
        return np.concatenate([fctrs, ext])

    def _get_fctrs(self, length: int):
        """Get fctrs, from left to right, for a given x length."""
        fctrs = self._fctrs
        if length == len(fctrs):
            return self._fctrs
        if length < len(fctrs):
            return fctrs[:length]
        return self._extended_fctrs(length)

    def _get_fctrs_rvr(self, length: int):
        """Get fctrs, from right to left, for a given x length."""
        fctrs = self._fctrs
        if length < len(fctrs):
            return (1 / fctrs[::-1])[-length:]
        if length > len(fctrs):
            fctrs = self._extended_fctrs(length)
        return 1 / fctrs[::-1]

    def _get_y_from_start(
        self, y: int | float | None = None, length: int | None = None
    ) -> list[int | float]:
        """Get y values from leftmost y value."""
        if y is None:
            y = self.line.y[0]
        if length is None:
            length = len(self.line.x)
        return y * self._get_fctrs(length)

    @property
    def components(self) -> list:
        """List of all Mark objects that comprise rule."""
        return self._components

    @property
    def color(self) -> str:
        """Rule color, for example 'yellow'."""
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = [color]
        for mark in self.components:
            mark.colors = self._color

    @property
    def opacity(self) -> list[float]:
        """Rule opacity, for example 0.5."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        self._opacity = [opacity]
        for mark in self.components:
            mark.opacities = self._opacity
            if isinstance(mark, bq.Label):
                mark.opacity = self._opacity

    def close(self):
        """Remove rule from frontend."""
        self.figure.marks = [m for m in self.figure.marks if m not in self._components]
        for comp in self._components:
            comp.close()


class FastIntervalSelectorExt(FastIntervalSelector):
    """`FastIntervalSelector` with additional functionality.

    Parameters
    ----------
    figure: Figure
        `bq.Figure` to which selector is to be associated.

    enable : bool, default: True
        Enable selector.

    kwargs:
        As `FastIntervalSelector`.
    """

    def __init__(self, figure: bq.Figure, enable=True, **kwargs):
        super().__init__(**kwargs)
        self.figure = figure
        if enable:
            self.enable()

    def enable(self):
        """Show interactive selector on figure."""
        self.figure.interaction = self

    def disable(self, reset: bool = True):
        """Disable selector.

        Remove selector from figure, prevent user from initiating a further
        selector, set selection to None.
        """
        if reset:
            # FastIntervalSelector.reset() has no effect when selector is not
            # currently assigned to figure.interaction. Both lines provides
            # for all circumstances.
            super().reset()
            self.selected = None
        self.figure.interaction = None

    def reset(self):
        """Reset selector.

        Remove selector from figure, set selection to None, provide for
        user to initiate a further selector via clicking figure.
        """
        self.disable(reset=True)
        self.enable()

    @property
    def has_selection(self) -> bool:
        """True if interval currently selected."""
        return self.selected is not None

    @property
    def start(self):
        """First selected value."""
        return self.selected[0]

    @property
    def end(self):
        """Last selected value."""
        return self.selected[-1]

    @property
    def interval(self) -> tuple | None:
        """2-tuple comprising first and last selected values."""
        if not self.has_selection:
            return None
        return (self.start, self.end)


class FastIntervalSelectorDD(FastIntervalSelectorExt):
    """`FastIntervalSelectorExt` for OrdinalScale of discontinuous dates.

    NB For continous dates use `FastIntervalSelectorExt` with DateScale.
    """

    @property
    def start(self) -> pd.Timestamp:
        """First selected value."""
        return discontinuous_date_to_timestamp(super().start)

    @property
    def end(self) -> pd.Timestamp:
        """Last selected value."""
        return discontinuous_date_to_timestamp(super().end)
