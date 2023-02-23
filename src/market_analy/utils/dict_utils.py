"""Dictionay-related utility functions and classes"""

# pylint: disable=line-too-long


def set_kwargs_from_dflt(passed: dict, dflt: dict, deep: bool = False) -> dict:
    """Add default kwargs.

    Parameters
    ----------
        passed : dict
            Dictionary of kwargs to which default kwargs, not otherwise
            present, to be added.

        dflt : dict
            Dictionary of default kwargs.

        deep : bool, default: False
            If True will execute recurssively to add default kwargs to
            any value of passed that is itself a dictionary for
            which a corresponding dictionary in dflt exists.

    Examples
    --------
    >>> default = {
    ...     'one': 1,
    ...     'two': {
    ...         'two.one': 2.1,
    ...         'two.two': 2.2,
    ...     },
    ...     'three': 3
    ... }
    >>> passed = {
    ...     'one': 1.7,
    ...     'two': {
    ...         'two.one': '2.1.1',
    ...         'two.three': '2.3.1',
    ...     },
    ...     'four': 4
    ... }
    >>> set_kwargs_from_dflt(passed, default, deep=True)
    {'one': 1.7, 'two': {'two.one': '2.1.1', 'two.three': '2.3.1', 'two.two': 2.2}, 'four': 4, 'three': 3}
    """
    for key, value in dflt.items():
        if isinstance(value, dict) and deep and key in passed:
            set_kwargs_from_dflt(passed[key], value)
        passed.setdefault(key, value)
    return passed


def exec_kwargs(kwargs: dict) -> dict:
    """Update callable dictionary values to callable's return."""
    for k, v in kwargs.items():
        if callable(v):
            kwargs[k] = v()
    return kwargs


def set_kwargs_from_dflt_exec(passed: dict, dflt: dict) -> dict:
    """Update dictionary, including with callable values.

    Updates `passed` dictionary for missing items defined in `dflt`. Also
    sets value of any kwarg that represents a callable to the callable's
    return value (regardless of whether orinally defined in `passed` or
    `dflt`).
    """
    kwargs = set_kwargs_from_dflt(passed, dflt)
    return exec_kwargs(kwargs)


def update_deep(d: dict, u: dict, /) -> dict:
    """Updates dictionary to include any nested dictionaries.

    Code copied from Alex Telon / Alex Martelli response at:
        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth.

    Parameters
    ----------
    d: dictionary to be updated.
    u: dictionary holding items with which to update d.

    Examples
    --------
    >>> default = {
    ...     'one': 1,
    ...     'two': {
    ...         'two.one': 2.1,
    ...         'two.two': 2.2,
    ...     },
    ...     'three': 3
    ... }
    >>> new = {
    ...     'one': 1.7,
    ...     'two': {
    ...         'two.one': '2.1.1',
    ...         'two.three': '2.3.1',
    ...     },
    ...     'four': 4
    ... }
    >>> update_deep(default, new)
    {'one': 1.7, 'two': {'two.one': '2.1.1', 'two.two': 2.2, 'two.three': '2.3.1'}, 'three': 3, 'four': 4}
    >>> default
    {'one': 1.7, 'two': {'two.one': '2.1.1', 'two.two': 2.2, 'two.three': '2.3.1'}, 'three': 3, 'four': 4}
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_deep(d.get(k, {}), v)
        else:
            d[k] = v
    return d
