"""Utility constants, functions and classes for bqplot."""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from itertools import cycle
from typing import Literal

import pandas as pd
import numpy as np
from bqplot import Lines, Figure, Mark, OHLC, Scatter, Label
from bqplot.interacts import FastIntervalSelector
from traitlets import HasTraits, Any, dlink

# possible keys for dictionary taken by +scales+ parameter of Axes class.
ScaleKeys = Literal["x", "y", "color", "opacity", "size", "rotation", "skew"]

# Colors for use on charts with dark backgrounds.
COLORS_DARK_8 = [
    "yellow",
    "springgreen",
    "hotpink",
    "darkorange",
    "lightskyblue",
    "beige",
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
        Value of an OrdinalScale represtenting a date which require as
        pd.Timestamp.
    """
    # utc version of fromtimestamp ensures that operation is the inverse
    # of dates_to_posix
    return pd.Timestamp.fromtimestamp(value / 1000)


class _LevelLine(Lines):
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
        figure: Figure,
        direction: Literal["vertical", "horizontal"] = "vertical",
        draw_to_figure=True,
        **kwargs,
    ):
        kwargs.setdefault("colors", ["Yellow"])
        kwargs.setdefault("stroke_width", 1)
        kwargs.setdefault("preserve_domain", {"y": True, "x": True})

        if vertical := (direction == "vertical"):
            scales = {"x": scales["x"], "y": figure.scale_y}
        else:
            scales = {"y": scales["y"], "x": figure.scale_y}

        super().__init__(
            x=[level, level] if vertical else [0, 1],
            y=[0, 1] if vertical else [level, level],
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

    def _draw_to_figure(self, figure: Figure):
        figure.marks = list(figure.marks) + [self]


class HLevelLine(_LevelLine):
    """Horizontal Line at specific level along a scale."""

    def __init__(self, level, scales: dict, figure: Figure, **kwargs):
        """Constructor.

        Parameters
        ----------
        level:
            level of y-scale at which to draw line.

        scales: dictionary of Scale objects
            Must include key 'y' with value as Scale which +level+
                references.

        figure:
            Figure to which line to be drawn.

        **kwargs: passed on to bqplot.Line.
        """
        super().__init__(level, scales, figure, direction="horizontal", **kwargs)


class VLevelLine(_LevelLine):
    """Vertical Line at specific level along a scale."""

    def __init__(self, level, scales: dict, figure: Figure, **kwargs):
        """Constructor.

        Parameters
        ----------
        level:
            level of x-scale at which to draw line.

        scales: dictionary of Scale objects
            Must include key 'x' with value as Scale which +level+
                references.

        figure:
            Figure to which line to be drawn.

        **kwargs: passed on to bqplot.Line.
        """
        super().__init__(level, scales, figure, direction="vertical", **kwargs)


class _LabeledLevelLine(_LevelLine):
    """Labeled Line at specific level along a scale.

    Label indicating level along scale placed alongside axis.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: Figure,
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
        self.label = Label(
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
        """Format `value` for use a a label.

        Converter for dlink between self.x or self.y and label's text.

        Parameters
        ----------
        value
            Either self.x or self.y, i.e. line's level along the scale

        Returns
        -------
            `value` formatted in accordance with `self.label_format`. Can
            be displayed as label text.
        """
        if self._label_format is None:
            return value[:1]
        value_ = value[0]
        if isinstance(value_, (np.datetime64, pd.Timestamp)):
            return [pd.to_datetime(value).strftime(self._label_format)]
        elif isinstance(value_, (np.integer, np.floating)):
            try:
                f_str = "{:" + self._label_format + "}"
                return [f_str.format(value_)]
            except ValueError:
                return [f_str.format(int(value_))]
        else:
            msg = "_format_label does not support value type " + str(type(value_))
            raise TypeError(msg)

    def _draw_to_figure(self, figure: Figure):
        figure.marks = list(figure.marks) + [self, self.label]


class HLabeledLevelLine(_LabeledLevelLine):
    """Horizontal Line at specific level along a scale.

    Label indicating level along scale placed alongside axis.
    """

    def __init__(
        self,
        level,
        scales: dict,
        figure: Figure,
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
        figure: Figure,
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
        VLabelledLevelLine that forms vertical hair.

    hhair:
        HLabelledLevelLine that forms horizontal hair.

    cross:
        Scatter mark that represents cross.
    """

    _y_center = Any()
    _x_center = Any()

    def __init__(
        self,
        figure: Figure,
        existing_mark: Mark | None = None,
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
            Passed to `bqplot.Lines` to create hairs.

        label_kwargs: dict
            Passed to `bqplot.Label` to create hair labels alongside axis.

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
        self.cross = Scatter(
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
        label_format = self._y_tick_format_python_format
        self.hhair = HLabeledLevelLine(
            level=start_y,
            scales=self.scales,
            figure=figure,
            label_format=label_format,
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
        if isinstance(self._existing_mark, OHLC):
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

    def __init__(self, figure: Figure, colors: Iterable[str] | None = None):
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

    def __init__(self, figure: Figure, enable=True, **kwargs):
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
