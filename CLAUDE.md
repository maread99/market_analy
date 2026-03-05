# LLM Assistant Guide for `market-analy` package

This file provides context for LLM assistants (Claude Code and similar tools) working in this repository.

## Project Overview

**market_analy** is a Python package for interactive charting and analysis of financial instruments. It provides GUIs built on bqplot, ipywidgets, and ipyvuetify for use in JupyterLab.
- Note that only a JupyterLab dark theme is currently supported (there is no support for a light theme).

See @pyproject.toml for project metadata and dependencies.

### Repository Layout

```
src/market_analy/
├── analysis.py        # Core Analysis and Compare classes
├── charts.py          # bqplot figure creation
├── guis.py            # Interactive GUI components
├── gui_parts.py       # GUI building blocks
├── trends/            # Trend analysis subpackage
│   ├── analy.py
│   ├── charts.py
│   ├── guis.py
│   └── movements.py
└── utils/             # Utility subpackage
    ├── bq_utils.py
    ├── pandas_utils.py
    ├── ipywidgets_utils.py
    ├── ipyvuetify_utils.py
    └── ...

tests/
├── test_analysis.py   # Main test suite
├── test_trends.py
├── conftest.py        # Fixtures and configuration
└── resources/         # Test data
```

## Technology Stack

| Category | Tools |
|---|---|
| Python | 3.10–3.13 (`.python-version` pins 3.13) |
| Package manager | `uv` |
| Build backend | `setuptools` + `setuptools_scm` |
| Testing | `pytest` |
| Linting/formatting | `ruff` |
| Type checking | `mypy` |
| Git hooks | `pre-commit` |
| Data Manipulation | `pandas`, `numpy` |
| Charting | `bqplot` |
| GUI Widgets | `ipywidgets`, `ipyvuetify` |
| Price Data | `market-prices` |
| Calendars of Market Hours | `exchange-calendars` |

The current project version is managed by `setuptools_scm` and written to `src/market_analy/_version.py`.
IMPORTANT: `src/maket_analy/_version.py` is auto-generated and you should not edit it.


## Development Workflows

### Setup

```bash
# Install dependencies using uv
uv sync

# Install pre-commit hooks
pre-commit install
```

### Testing

- test with `pytest`
- see @pytest.ini for configuration; options are applied automatically via `addopts`.
- shared fixtures are in `tests/conftest.py`
- tests are in `tests/`
- doctests are included to some methods/functions

Commands to run tests:
```bash
# All tests (including doctests in src/market_analy/)
pytest

# Specific test file
pytest tests/test_analysis.py

# With verbose output
pytest -v
```

### Pre-commit Hooks

See @.pre-commit-config.yaml for pre-commit implementation.

Pre-commit runs automatically on `git commit`.

To run manually:
```bash
pre-commit run --all-files
```

### Dependency Management

```bash
# Add a dependency
uv add <package>

# update all dependencies
uv lock --upgrade
uv sync --inexact
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt
```

---

### CI

GitHub Actions is used for Continuous Integration. Defined workflows include:
- @.github/workflows/build-test.yml - runs full test suite on matrix of platforms and python versions
- @.github/workflows/release.yml - releases a new version to PyPI

## Code Conventions

### Architecture

The project employs a hierarchal class structure although compositional elements can be used if considered beneficial.

### Formatting

- format to `ruff` (Black-compatible).  
- see @ruff.toml for configuration.

```bash
# Format code
ruff format .
```

### Linting

- lint with `ruff`.
- See lint sections of @ruff.toml for configuration (includes excluded files).
- type check with `mypy`.

```bash
# Check lint issues
ruff check .

# Type checking
uv run mypy src/market_analy/
```

### Imports

- No wildcard imports (`from x import *`).

### Type Annotations

- Type annotations are required on all public functions and methods.
- See @mypy.ini for configuration
    - `ignore_missing_imports = True` is set globally (many dependencies lack stubs).
- `valimp` library is used for runtime parameter validation:
    - use `@parse` decorator with typed signatures.
    - use`@parse_cls` for dataclasses.

### Docstrings

Public modules, classes, and functions MUST all have docstrings.

Docstrings should follow **NumPy convention**. Familiarise yourself with this as described at https://numpydoc.readthedocs.io/en/latest/format.html. That said, the following should always be adhered to and allowed to override any NumPy convention:
- 75 character line limit for public documentation
- 88 character line limit for private documentation
- formatted to ruff
- parameter types should not be included to the docstring unless this provides useful information that users could not otherwise ascertain from the typed function signature.
- default values should only be noted in function/module docstrings if not defined in the signature - for example if the parameter's default value is None and when received as None the default takes a concrete dynamically evaluated default value. When a default value is included to the parameter documentation it should be defined after a comma at the end of the parameter description, for example:
    - description of parameter 'whatever', defaults to 0.
- **subclasses** documentation should:
    - list only methods and attributes added by the subclass. A note should be included referring users to documentation of base classes for the methods and attributes defined there.
    - include a NOTES section documenting how to implement the subclass (only if not trivial).
- documentation of **subclass methods that extend methods of a base class** should only include any parameters added by the extension. With respect to undocumented parameters a note should be included to refer the user to the corresponding 'super' method(s)' documentation on the corresponding base class or classes.
- **documentation of exceptions and warnings** should be limited to only **unusual** exceptions and warnings that are raised directly by the function/method itself or by any private function/method that is called directly or indirectly by the function/method.
- summary line should be in the imperative mood only when sensical to do so.
- magic methods do not require documentation if their functionality is fully implied by the method name.
- unit tests do not require docstrings.

Example documentation:
```python
def my_func(param1: int, param2: str = "default", param3: None | str = None) -> bool:
    """Short summary line.

    Extended description if needed.

    Parameters
    ----------
    param1
        Description of param1.
    param2
        Description of param2.
    param3
        Description of param3, defaults to value of `param2`.

    Returns
    -------
    bool
        Description of return value.
    """
```

### Comments

- pay particular attention to comments starting with...: 
    - 'NOTE'
    - 'TODO'
    - 'AIDEV-NOTE' - these comments are specifically addressed to you.
    - 'AIDEV-TODO' - these comments are specifically requesting you do something.
    - 'AIDEV-QUESTION' - these comments are asking a question for specifically you to answer.

---

## Important Notes for AI Agents

1. **NEVER DO RULES**:
	- Never edit the file `src/market_analy/_version.py` - this is auto-generated by the build process.

2. **Use `valimp` for parameter validation** — see 'Type Annotations' section of this CLAUDE.md file.

3. **NumPy docstring style** — all new public functions/classes must use NumPy-convention docstrings and rules as defined under Docstrings section of this CLAUDE.md file.

4. **Branch naming** — feature branches should follow the `llm/<description>` pattern.
