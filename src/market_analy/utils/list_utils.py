"""Utility functions and classes for lists."""

from __future__ import annotations

from typing import Any, Literal
from copy import copy


# DEVELOPMENT NOTE: _SelectableList
# Instead of a dedicated SelectableListMult class, Why not just aad a
# `multiple`` argument to SelectableList? Arranged as two classes to
# provide a cleaner interface that doesn't require distinguishing if a
# list being added to the selection is a single element to be added
# or list of multiple elements.
class _SelectableList(list):
    """Base class to create a list object with selectable elements.

    Subclasses should define public methods to select elements. These i
    turn should call on of the following private methods:
        _add_element_to_selection()
        _add_elements_to_selection()

    Parameters
    ----------
    elements: list
        Elements that can be selected from.

    names: list | None
        Optional names by which elements can be referenced. If passed must
        be same length as elements. If not passed elements will be assigned
        names as integers using 0-based index.

    Properties
    ----------
    has_selection:
        True if there are selected items, otherwise False.

    selected:
        Current selected items

    named_elements:
        Dictionary with keys as name and values as associated element.

    Methods
    -------
    get_element():
        Get element by name.

    is_selected_element(element):
        Check if element currently selected

    is_selected_index(index):
        Check if element referenced by index is currently selected.

    is_selected_name(name):
        Check if element referenced by name is currently selected.

    deselect_all():
        Deselect all selected elements.
    """

    def __init__(self, elements: list, names: list[str | int] | None = None):
        super().__init__(elements)
        self._names = names if names is not None else list(range(len(self)))
        assert len(elements) == len(
            self._names
        ), "'elements' and 'names' must have same length"
        self._named_elements = dict(zip(self._names, self))

        self._selected: list[Any] = []

    @property
    def named_elements(self) -> dict:
        """Dictionary with keys as name and values as associated element."""
        return self._named_elements

    def get_element(self, name: str) -> Any:
        return self.named_elements[name]

    def _subset_from_indices(self, indices: list[int]):
        return [self[i] for i in indices]

    def _subset_from_names(self, names: list) -> list[Any]:
        return [self.named_elements[name] for name in names]

    @property
    def has_selection(self) -> bool:
        return bool(self._selected)

    @property
    def selected(self) -> list[Any]:
        """Currently selected elements."""
        return self._selected

    def is_selected_element(self, element: Any) -> bool:
        """Check if element is currently selected."""
        return element in self._selected

    def is_selected_index(self, index: int) -> bool:
        """Check if element at a given index is selected."""
        return self.is_selected_element(self[index])

    def is_selected_name(self, name: str) -> bool:
        """Check if element referenced by name is selected."""
        return self.is_selected_element(self.get_element(name))

    def _add_err_msg(self, elements: Any):
        msg = (
            "'" + str(elements) + "' cannot be added to selection as it "
            "is an object or includes an object that is not an element "
            "of the selectable list. Selectable elements are " + str(self) + "."
        )
        return msg

    def _add_element_to_selection(self, element: Any):
        """Add element to selection."""
        assert element in self, self._add_err_msg(element)
        self._selected.append(element)
        return element

    def _add_elements_to_selection(
        self, elements: list, reset=False, _check_in_list=True
    ):
        """Add elements to selection.

        reset:
            True/False to replace/extend existing selection.

        _check_in_list:
            False to skip checking that passed elements are all elements of
            list.
        """
        assert isinstance(elements, list)
        if _check_in_list:
            assert all(e in self for e in elements), self._add_err_msg(elements)

        if reset:
            self._selected = elements
        else:
            # remove any element already included in selection
            elements = [e for e in elements if e not in self._selected]
            self._selected.extend(elements)

        return elements

    def _remove_elements_from_selection(self, elements: list, pass_silently=True):
        """Remove elements from selection.

        pass_silently : bool, default: True
            False: raise ValueError on trying to remove an element that was
            not in selection.

            No element is removed if error raised.
        """
        assert isinstance(elements, list)

        saved_state = copy(self._selected)
        try:
            for elem in elements:
                try:
                    self._selected.remove(elem)
                except ValueError:
                    if pass_silently:
                        pass
                    else:
                        raise
        except Exception:
            self._selected = saved_state
            raise

    def deselect_all(self):
        """Remove all elements from selection."""
        self._selected = []


class SelectableList(_SelectableList):
    """List of elements, one or zero of which can be defined as selected.

    Functionality:
        Select an element
        Get selected element
        Check if a specific element is selected
        Deselect element

    For each functionality three methods are provided that each allow for
    the element to be referenced in a particular way:
        'element' direct reference to element object,
            e.g. `selected_element`
        'name' by name associated with element
            e.g. `selected_name`
        'index' by element's index position in list
            e.g. `selected_index`

    For each functionality there is a further method, with a less verbose
    name, e.g. `selected`, which references the elemment in the default
    manner defined by passing `select_by` to the constructor as one of
    'element', 'name' or 'index'.

    Properties
    ----------
    Get selected element:
        selected
        selected_element
        selected_name
        selected_index

    Methods
    -------
    Select an element:
        select()
        select_element()
        select_name()
        select_index()

    Check if an element is selected:
        is_selected()
        is_selected_element()
        is_selected_name()
        is_selected_index()

    deselect():
        Deselect any selected element.
    """

    def __init__(
        self,
        elements: list[Any],
        names: list[str | int] | None = None,
        select_by: Literal["element", "name", "index"] = "element",
        initial_selection: Any | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        elements: list
            Elements that can be selected from.

        names : list | None
            Optional names by which elements can be referenced. If passed
                must be same length as elememts. If not passed elements will
                be assigned names as integers using 0-based index.

        select_by : Literal['element', 'name', 'index'], default: 'element'
            Determines how default methods reference an element:
                'element' directly to element object
                'name' by name associated with element
                'index' by element's index position in list

        initial_selection:
            Reference to any element to be initially selected. Reference
            should be passed as either element, name or index, as
            determined by `select_by`.
        """
        super().__init__(elements, names)
        assert select_by in ["element", "name", "index"]
        self._select_by = select_by

        method_name = "select_" + select_by
        self._default_select_method = getattr(self, method_name)
        method_name = "is_selected_" + select_by
        self._default_is_selected_method = getattr(self, method_name)

        if initial_selection is not None:
            self.select(initial_selection)

    @property
    def selected_element(self) -> Any | None:
        """Selected element. None if no selection."""
        if not self.has_selection:
            return None
        return self._selected[0]

    @property
    def selected_index(self) -> int | None:
        """Index of selected element. None if no selection."""
        if not self.has_selection:
            return None
        return self.index(self.selected_element)

    @property
    def selected_name(self) -> str | int | None:
        """Name of selected element. None if no selection."""
        if not self.has_selection:
            return None
        return self._names[self.selected_index]

    @property
    def selected(self) -> Any:
        """Selected element, as default reference."""
        # NB implmentation differs from other fucntionatlity by
        # way of getting default method now rather than at time of
        # instantiation. This is because property object cannot be
        # referenced from constructor (rather returns value
        # property evaluates to)
        return getattr(self, "selected_" + self._select_by)

    def is_selected(self, reference: Any) -> bool:
        """Check if element is currently selected.

        reference:
            Element to check, referenced by default preference.
        """
        return self._default_is_selected_method(reference)

    def select_element(self, element: Any) -> Any:
        """Select element."""
        self.deselect()
        return self._add_element_to_selection(element)

    def select_index(self, index: int) -> Any:
        """Select element referenced by index."""
        return self.select_element(self[index])

    def select_name(self, name: str) -> Any:
        """Select element referenced by name."""
        return self.select_element(self.named_elements[name])

    def select(self, reference: Any) -> Any:
        """Select element referenced by default preference."""
        return self._default_select_method(reference)

    def deselect(self) -> Any:
        """Deselect any selected element."""
        super().deselect_all()


class SelectableListMult(_SelectableList):
    """List of elements, zero or any number of which can defined as selected.

    Functionality:
        Select elements
        Get selected elements
        Check if a specific element is selected
        Check if specific elements are all selected
        Deselect elements

    For most functionalities, three methods are provided that each allow
    for the element to be referenced in a particular way:
        'elements' direct reference to element object
            e.g. `selected_element`
        'names' by name associated with element
            e.g. `selected_name`
        'indices' by element's index position in list
            e.g. `selected_index`

    For each functionality there is a further method, with a less verbose
    name, e.g. `selected`, which references the elemment in the default
    manner defined by passing `select_by` to the constructor as one of
    'elements', 'names' or 'indices'.

    Properties
    ----------
    Get selected elements:
        selected
        selected_elements
        selected_names
        selected_indices

    Methods
    -------
    Select elements:
        select()
        select_elements()
        select_names()
        select_indices()

    Deselect elements:
        deselect()
        deselect_elements()
        deselect_names()
        deselect_indices()

    Check if an element is selected:
        is_selected()
        is_selected_element()
        is_selected_name()
        is_selected_index()

    Check if all of specified elements are selected:
        are_selected()
        are_selected_elements()
        are_selected_names()
        are_selected_indices()

    deselect_all():
        Deselect any selected element.
    """

    singular_map = {"elements": "element", "names": "name", "indices": "index"}

    def __init__(
        self,
        elements: list[Any],
        names: list[str | int] | None = None,
        select_by: Literal["elements", "names", "indices"] = "elements",
        initial_selection=None,
    ):
        """Constructor.

        Parameters
        ----------
        elements: list
            Elements that can be selected from.

        names : list | None
            Optional names by which elements can be referenced. If passed
                must be same length as elememts. If not passed elements will
                be assigned names as integers using 0-based index.

        select_by : Literal['elements', 'names', 'indices'], default: 'elements'
            Determines how default methods reference an element:
                'elements' directly to element objects
                'names' by names associated with elements
                'indices' by elements' index positions in list

        initial_selection:
            Reference to any elements to be initially selected. Reference
            should be passed as either elements, names or indices, as
            determined by `select_by`.
        """
        super().__init__(elements, names)
        assert select_by in ["elements", "names", "indices"]
        self._select_by = select_by

        method_name = "select_" + select_by
        self._default_select_method = getattr(self, method_name)
        method_name = "is_selected_" + self.singular_map[select_by]
        self._default_is_selected_method = getattr(self, method_name)
        method_name = "are_selected_" + select_by
        self._default_are_selected_method = getattr(self, method_name)
        method_name = "deselect_" + select_by
        self._default_deselect_method = getattr(self, method_name)

        if initial_selection is not None:
            self.select(initial_selection)

    @property
    def selected_elements(self) -> None | list[Any]:
        """Selected elements. None if no selection."""
        return None if not self.has_selection else self._selected

    @property
    def selected_indices(self) -> None | list[int]:
        """Indices of selected elements. None if no selection."""
        if not self.has_selection:
            return None
        else:
            return [self.index(e) for e in self.selected_elements]

    @property
    def selected_names(self) -> None | list[str | int]:
        """Names of selected elements. None if no selection."""
        if not self.has_selection:
            return None
        else:
            return [self._names[i] for i in self.selected_indices]

    @property
    def selected(self) -> None | list[Any]:
        """Selected elements, as default reference."""
        # NB implmentation differs from other fucntionatlity by
        # way of getting default method now rather than at time of
        # instantiation. This is because property object cannot be
        # referenced from constructor (rather returns value
        # property evaluates to)
        return getattr(self, "selected_" + self._select_by)

    def is_selected_element(self, element: Any) -> bool:
        """Check if element is currently selected."""
        return element in self._selected

    def is_selected_index(self, index: int) -> bool:
        """Check if element referenced by index is currently selected."""
        return self.is_selected_element(self[index])

    def is_selected_name(self, name: str) -> bool:
        """Check if element referenced by name is currently selected."""
        return self.is_selected_element(self.get_element(name))

    def is_selected(self, reference: Any) -> bool:
        """Check if element is currently selected.

        reference:
            Element to check, referenced by default preference.
        """
        return self._default_is_selected_method(reference)

    def are_selected_elements(self, elements: list[Any]) -> bool:
        """Check if all passed elements are selected."""
        return all(e in self.selected_elements for e in elements)

    def are_selected_indices(self, indices: list[int]) -> bool:
        """Check if all elements at given indices are selected."""
        elements = self._subset_from_indices(indices)
        return self.are_selected_elements(elements)

    def are_selected_names(self, names: list[str]) -> bool:
        """Check if all elements of given names are selected."""
        elements = self._subset_from_names(names)
        return self.are_selected_elements(elements)

    def are_selected(self, references: list[Any]) -> bool:
        """Check if all passed elements are selected.

        references
            Elements to check for selection, referenced as default
            preference.
        """
        return self._default_are_selected_method(references)

    def select_elements(
        self, elements: list, reset: bool = False, _check_in_list: bool = True
    ):
        """Select elements.

        reset
            True/False to replace/extend existing selection.

        _check_in_list : bool, default: True
            False: skip checking passed elements are all elements of list.
        """
        self._add_elements_to_selection(elements, reset, _check_in_list)

    def select_indices(self, indices: list[int], reset: bool = False):
        """Select elements referenced by indices.

        reset : bool, default: False
            True/False to replace/extend existing selection.
        """
        elements = self._subset_from_indices(indices)
        self.select_elements(elements, reset, _check_in_list=False)

    def select_names(self, names: list[str], reset: bool = False):
        """Select elements referenced by names.

        reset : bool, default: False
            True/False to replace/extend existing selection.
        """
        elements = self._subset_from_names(names)
        self.select_elements(elements, reset, _check_in_list=False)

    def select(self, references: list[Any], reset: bool = False):
        """Select elements referenced by default preference.

        reset : bool, default: False
            True/False to replace/extend existing selection.
        """
        return self._default_select_method(references, reset)

    def deselect_elements(self, elements: list[Any], pass_silently: bool = True):
        """Deselect elements.

        Parameters
        ----------
        pass_silently : bool, default: True
            False to raise ValueError on trying to remove an element that
            is not selected.
        """
        return self._remove_elements_from_selection(elements, pass_silently)

    def deselect_indices(self, indices: list[int], pass_silently: bool = True):
        """Deselect elements referenced by indices.

        Parameters
        ----------
        pass_silently : bool, default: True
            False to raise ValueError on trying to remove an element that
            is not selected.
        """
        elements = self._subset_from_indices(indices)
        self.deselect_elements(elements, pass_silently)

    def deselect_names(self, names: list[str], pass_silently: bool = True):
        """Deselect elements reference by names.

        Parameters
        ----------
        pass_silently : bool, default: True
            False to raise ValueError on trying to remove an element that
            is not selected.
        """
        elements = self._subset_from_names(names)
        self.deselect_elements(elements, pass_silently)

    def deselect(self, references: list[Any], pass_silently: bool = True):
        """Deselect elements referenced by default preference."""
        return self._default_deselect_method(references, pass_silently)
