<!-- NB any links not defined as aboslute will not resolve on PyPI page -->
# market_analy

[![PyPI](https://img.shields.io/pypi/v/market-analy)](https://pypi.org/project/market-analy/) ![Python Support](https://img.shields.io/pypi/pyversions/market-analy) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python package to analyse financial instruments.

There's a load of great financial libraries out there for Technical Analysis, Charting, Backtesting, Portfolio Analysis etc. This library isn't reinventing the wheel (not intentionally anyway), rather it provides functions and interactive charting that I think are useful and couldn't find elsewhere. It's not comprehensive, but rather fills some of the gaps.

Some functionality is general, some is focused on defining and identifying trends. Interactive charting is offered via guis created from widgets of the bqplot, ipywidgets and ipyvuetify libraries. Users can use the underlying parts to develop their own interactive charts and analyses. Contributions very much welcome! This is a WIP and it's anticipated that further analyses will be added.

The [demo video](https://vimeo.com/801302973) gives an overview of what´s on offer. All analyses are accessed via the classes `analysis.Analysis` (single instrument analyses) and `analysis.Compare` (to compare multiple instruments). For example:

```python
from market_prices import PricesYahoo
from market_analy import Analysis,  Compare

prices = PricesYahoo("MSFT")
analy = Analysis(prices)
gui = analy.plot(days=30)
```
https://user-images.githubusercontent.com/56914820/220773777-df0d0bec-bbe1-45bb-b067-d679666450cd.mp4

```python
comp = Compare(PricesYahoo("MSFT, AMZN, TSLA"))
gui = comp.plot(hours=30)
```
https://user-images.githubusercontent.com/56914820/220773790-1fdabf13-25bb-4205-acc2-6bac9b832dae.mp4

```python
gui = comp.chg_every_interval("20T", days=5, chart=True)
```
https://user-images.githubusercontent.com/56914820/220773802-ae329259-4a4e-4e5e-8d02-d4ee88b8b452.mp4


For further documentation, see the [analysis](https://github.com/maread99/market_analy/blob/master/src/market_analy/analysis.py) module.

## Installation and environment

It's recommended that `market-analy` is installed to a new virtual environment created using `venv`.

The package can be installed to the activated environment via pip:

`$ pip install market-analy`

Plots are intended to be created in a Jupyter Notebook or JupyterLab. The 'jupyter' optional dependencies can be specified to additionally install `jupyter` and `jupyterlab` to the target environment.

`$ pip install market-analy[jupyter]`

Then call:

`jupyter nbextension enable --py --sys-prefix ipyvuetify`

Alternatively, it's possible to use an existing Jupyter installation in a separate environment to that in which `market_analy` is installed. In this case:
* The following dependencies should additionally be installed **in the environment to which Jupyter is installed**:
  - `jupyterlab>=3.0`
  - `ipyvuetify`
  - `bqplot`
* Jupyter should be called with the following arguments:
  - `jupyter nbextension enable --py --sys-prefix ipyvuetify`

> :information_source: Unfortunately plots do not render in a VSCode notebook.

### Color scheme
The color scheme assumes the package is being used with the JupyterLab dark theme. There are no plans to provide a 'light theme' option (although a contribution would certainly be welcome from anyone seeking one).

### `market-prices` dependency
`market-analy` depends on the [market-prices][market-prices] library for price data. This provides for functionality including:
* defining analysis periods in terms of number of sessions and trading minutes rather than calendar days and times.
* complete data sets regardless of liquidity (regular data points during market hours, no data points outside of market hours).

Most of the arguments available to the market-prices `get` function can be passed directly to the `market_analy` functions. See the [market-prices][market-prices] documentation for further info.

## Disclaimers
`market-analy` should not be assumed sufficiently reliable to undertake analysis intended to inform investment decisions. Users should inspect the source code and the test suite of the library and its dependencies in order to make their own assessment of the packages' suitability for their purposes. **The `market-analy` package is used entirely at the user's own risk.**

The test suite is limited. It's pretty much guaranteed that there are bugs. Please raise an [issue](https://github.com/maread99/market_analy/issues) if you find one or come across unexpected behaviour.

The default `market_prices.PricesYahoo` class gets data from publically available Yahoo APIs. **See the [Disclaimers section of the market-prices README](https://github.com/maread99/market_prices#disclaimers) for conditions of use**, including restrictions.

## Alternative packages

* [awesome-quant](https://github.com/wilsonfreitas/awesome-quant) offers an extensive listing of libraries for all things finance.

## License

[MIT License][license]


[license]: https://github.com/maread99/beanahead/blob/master/LICENSE.txt
[market-prices]: https://github.com/maread99/market_prices
