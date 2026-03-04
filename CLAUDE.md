# CLAUDE.md

## Project Overview

**market_analy** is a Python package for interactive charting and analysis of financial instruments. It provides GUIs built on bqplot, ipywidgets, and ipyvuetify for use in JupyterLab.

Key classes:
- `Analysis` — analyze a single financial instrument
- `Compare` — compare multiple instruments

## Development Setup

This project uses `uv` for dependency management.

```bash
# Install all dependencies (including dev)
uv sync --locked --group dev
```

## Common Commands

### Run Tests
```bash
uv run pytest
```

Tests include both unit tests (`tests/`) and doctests (`src/market_analy/`).

### Linting & Formatting
```bash
# Check and auto-fix lint issues
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking
```bash
uv run mypy src/market_analy/
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Project Structure

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

## Key Dependencies

- **bqplot** — interactive plotting for Jupyter
- **ipyvuetify** — Material Design widgets
- **ipywidgets** — Jupyter widgets
- **market-prices** — financial price data (Yahoo Finance via yahooquery)
- **pandas** / **numpy** — data manipulation
- **exchange-calendars** — trading calendars

## CI/CD

GitHub Actions runs on Python 3.10 and 3.13:
1. Pre-commit checks (ruff lint/format, standard hooks)
2. Full pytest suite

See `.github/workflows/build-test.yml`.

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata and dependencies |
| `pytest.ini` | Pytest settings (includes `--doctest-modules`) |
| `ruff.toml` | Linting/formatting (line length 88, Python 3.10+) |
| `mypy.ini` | Type checking (strict settings, `ignore_missing_imports`) |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff v0.13.0) |

## Notes

- Designed for **JupyterLab dark theme**
- Python 3.10+ required
- License: MIT
