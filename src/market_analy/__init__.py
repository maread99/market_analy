"""Market Analysis.

The market analysis package comprises the following modules:

mkt_analysis
    Price analysis of one or multiple instruments.

charts
   Classes to create bqplot figures displaying financial data.

gui_parts
    Classes to create parts of a GUI. NB Parts are independent (interaction
    between parts is the responsibility of `guis` module).

guis
    Interactive GUIs. GUIs comprise widgets and components from `gui_parts`
    module and a bqplots figure. `guis` is responsible for providing all
    interactive functionality between objects.

utils subpackage
    Various utility modules
"""

from .analysis import Analysis, Compare

__all__ = [Analysis, Compare]

__copyright__ = "Copyright (c) 2023 Marcus Read"


# Resolve version
__version__ = None

from importlib.metadata import version

try:
    # get version from installed package
    __version__ = version("market_analy")
except ImportError:
    pass

if __version__ is None:
    try:
        # if package not installed, get version as set when package built
        from ._version import version
    except Exception:
        # If package not installed and not built, leave __version__ as None
        pass
    else:
        __version__ = version

del version
