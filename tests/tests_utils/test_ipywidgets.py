"""Tests for the `utils.ipoywidgets_utils` module."""

import ipywidgets as w

from market_analy.utils.ipywidgets_utils import capture_widgets

# TODO: test modules is very notably incomplete!!


class TestCaptureWidgets:
    """Tests for the `capture_widgets` context manager."""

    def test_capture_and_restore(self):
        initial = w.Widget._widget_construction_callback
        outer: list[w.Widget] = []
        inner: list[w.Widget] = []
        with capture_widgets(outer):
            a = w.HTML()
            with capture_widgets(inner):
                b = w.HTML()
            c = w.HTML()
        assert a in outer
        # the outer capture resumes once the inner context exits
        assert c in outer
        assert b in inner
        assert b not in outer
        # the prior callback is restored (not simply cleared)
        assert w.Widget._widget_construction_callback is initial
