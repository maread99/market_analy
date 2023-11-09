"""Tests for the `market_analy.utils.list_utils` module."""

import pytest

from market_analy.utils import list_utils as m


def test_selectable_list():
    """Reasonably broad test for SelectableList."""
    elements = ["one", 2, 3.0, [4], {"five": 5}, {6}, (7, 7)]
    names = ["one", "two", "random", "four", 5, "six", "seven"]

    sl = m.SelectableList(elements, names, select_by="element")

    assert sl.selected is None

    sl = m.SelectableList(elements, names, select_by="element", initial_selection=3.0)
    assert sl.selected == 3.0

    sl.select([4])
    assert sl.selected == [4]

    with pytest.raises(AssertionError):
        sl.select(["not there"])

    sl.select_index(4)
    assert sl.selected == {"five": 5}
    sl.select_name("random")
    assert sl.selected == 3.0

    sl.deselect()
    assert sl.selected is None

    sl = m.SelectableList(elements, names, select_by="index", initial_selection=6)
    assert sl.selected == 6
    assert sl.selected_element == (7, 7)
    sl.select_name("random")
    assert sl.selected_element == 3.0
    assert sl.selected == 2

    sl = m.SelectableList(elements, names, select_by="name", initial_selection="six")
    assert sl.selected == "six"
    assert sl.selected_name == "six"
    assert sl.selected_element == {6}
    assert sl.selected_index == 5

    sl.select("four")
    assert sl.is_selected("four")
    assert not sl.is_selected("one")
    assert sl.is_selected_index(3)
    assert sl.is_selected_element([4])

    with pytest.raises(KeyError):
        sl.is_selected("not included")


def test_selectable_list_mult():
    """Reasonably broad test for SelectableListMult."""

    elements = ["one", 2, 3.0, [4], {"five": 5}, {6}, (7, 7)]
    names = ["one", "two", "random", "four", 5, "six", "seven"]

    slm = m.SelectableListMult(elements, names, select_by="elements")

    assert slm.selected is None

    slm = m.SelectableListMult(
        elements, names, select_by="elements", initial_selection=["one", 3.0]
    )
    assert slm.selected == ["one", 3.0]

    slm.select([[4]])
    assert slm.selected == ["one", 3.0, [4]]

    with pytest.raises(AssertionError):
        slm.select(["not there"])

    slm.select_indices([4, 6])
    assert slm.selected == ["one", 3.0, [4], {"five": 5}, (7, 7)]
    slm.select_names(["random"])
    assert slm.selected == ["one", 3.0, [4], {"five": 5}, (7, 7)]

    slm.select_names(["random", "six"], reset=True)
    assert slm.selected == [3.0, {6}]
    slm.select_elements([2, "one", (7, 7)])
    assert slm.selected == [3.0, {6}, 2, "one", (7, 7)]

    slm.deselect(["one", 2])
    assert slm.selected == [3.0, {6}, (7, 7)]

    with pytest.raises(ValueError):
        slm.deselect_indices([1, 2], pass_silently=False)

    assert slm.selected == [3.0, {6}, (7, 7)]
    slm.deselect_indices([1, 2])
    assert slm.selected == [{6}, (7, 7)]

    slm.deselect_names(["one", "seven"])
    assert slm.selected == [{6}]
    slm.select(["one", 3.0])
    slm.deselect_elements([3.0])
    assert slm.selected == [{6}, "one"]

    slm = m.SelectableListMult(
        elements, names, select_by="indices", initial_selection=[0, 2, 3, 6]
    )
    assert slm.selected == [0, 2, 3, 6]
    assert slm.selected_elements == ["one", 3.0, [4], (7, 7)]
    slm.deselect([1, 2, 3, 4])
    assert slm.selected_names == ["one", "seven"]
    slm.deselect_all()
    assert slm.selected is None

    slm = m.SelectableListMult(
        elements, names, select_by="names", initial_selection=["one", "six", 5, "seven"]
    )
    assert slm.selected == ["one", "six", 5, "seven"]
    slm.deselect(["six"])
    assert slm.selected_indices == [0, 4, 6]

    assert slm.is_selected("one")
    assert not slm.is_selected("random")

    with pytest.raises(KeyError):
        slm.is_selected("not included")

    slm.select(["two", "random"])
    assert slm.selected == ["one", 5, "seven", "two", "random"]

    assert slm.are_selected(["one", 5])
    assert not slm.are_selected(["one", "seven", "six"])
    assert slm.are_selected_elements([{"five": 5}, 3.0])
    assert not slm.are_selected_elements([{"five": 5}, 3.0, {6}])
    assert slm.are_selected_indices([0, 1, 2, 6])
    assert not slm.are_selected_indices([3, 6])
