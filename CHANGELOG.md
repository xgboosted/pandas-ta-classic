# Changelog

All notable changes to this project will be documented in this file.

## **[Unreleased]**

### Added
* **UV Package Manager Support**: All documentation now includes installation instructions for both `uv` (recommended for faster installs) and traditional `pip`. This includes README.md, CONTRIBUTING.md, docs/installation.rst, docs/index.rst, index.md, and docs/indicators.rst.
* **Parameterized Package Configuration**: Version and Python classifiers in `pyproject.toml` now use placeholder templates (`UPDATEME!` and `{{LATEST-N}}`) that are dynamically replaced during CI/CD release workflows, eliminating hardcoded values and enabling automated version management.

### Changed
* **Dynamic Category Discovery**: The `Category` dictionary in `_meta.py` is now built dynamically by scanning the filesystem structure. This eliminates manual maintenance, ensures it stays in sync with available indicators, and automatically discovered several previously undocumented indicators (`cdl_doji`, `cdl_inside`, `hwma`, `ma`, `drawdown`, `dm`, `vp`).
* **Updated Indicator Counts**: Corrected total indicator count from 143 to 141 indicators, and total count from 205 to 203 (141 indicators + 62 TA-Lib patterns) to reflect actual available indicators.
* **Documentation Updates**: Updated README.md, docs/indicators.rst, and CONTRIBUTING.md to reflect dynamic discovery and correct indicator counts.
* **Python Version Support**: Updated to Python 3.9-3.13 following a rolling 5-version support policy. Version requirements are now dynamically managed via CI/CD workflows (`LATEST_PYTHON_VERSION` in `.github/workflows/ci.yml`).
* **Development Status**: Changed from Beta to Production/Stable in `pyproject.toml` to reflect library maturity.

<br />

## **General**
* A __Strategy__ Class to help name and group your favorite indicators.
* If a **TA Lib** is already installed, Pandas TA will run TA Lib's version. (**BETA**)
* Some indicators have had their ```mamode``` _kwarg_ updated with more _moving average_ choices with the **Moving Average Utility** function ```ta.ma()```. For simplicity, all _choices_ are single source _moving averages_. This is primarily an internal utility used by indicators that have a ```mamode``` _kwarg_. This includes indicators: _accbands_, _amat_, _aobv_, _atr_, _bbands_, _bias_, _efi_, _hilo_, _kc_, _natr_, _qqe_, _rvi_, and _thermo_; the default ```mamode``` parameters have not changed. However, ```ta.ma()``` can be used by the user as well if needed. For more information: ```help(ta.ma)```
    * **Moving Average Choices**: dema, ema, fwma, hma, linreg, midpoint, pwma, rma, sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma.
* An _experimental_ and independent __Watchlist__ Class located in the [Examples](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples/watchlist.py) Directory that can be used in conjunction with the new __Strategy__ Class.
* _Linear Regression_ (**linear_regression**) is a new utility method for Simple Linear Regression using _Numpy_ or _Scikit Learn_'s implementation.
* Added utility/convience function, ```to_utc```, to convert the DataFrame index to UTC. See: ```help(ta.to_utc)``` **Now** as a Pandas TA DataFrame Property to easily convert the DataFrame index to UTC.

<br />

## **Breaking / Depreciated Indicators**
* _Trend Return_ (**trend_return**) has been removed and replaced with **tsignals**. When given a trend Series like ```close > sma(close, 50)``` it returns the Trend, Trade Entries and Trade Exits of that trend to make it compatible with [**vectorbt**](https://github.com/polakowo/vectorbt) by setting ```asbool=True``` to get boolean Trade Entries and Exits. See ```help(ta.tsignals)```

<br/>

## **New Indicators**
* _Arnaud Legoux Moving Average_ (**alma**) uses the curve of the Normal (Gauss) distribution to allow regulating the smoothness and high sensitivity of the indicator. See: ```help(ta.alma)```
* _Draw Down_ (**drawdown**) calculates the percentage decline from the peak equity of a trading account, or fund. See ```help(ta.drawdown)```
* _Candle Patterns_ (**cdl_pattern**) If TA Lib is installed, then all those Candle Patterns are available. See the list and examples above on how to call the patterns. See ```help(ta.cdl_pattern)```
* _Candle Z Score_ (**cdl_z**) normalizes OHLC Candles with a rolling Z Score. See ```help(ta.cdl_z)```
* _Correlation Trend Indicator_ (**cti**) is an oscillator created by John Ehler in 2020. See ```help(ta.cti)```
* _Cross Signals_ (**xsignals**) was created by Kevin Johnson. It is a wrapper of Trade Signals that returns Trends, Trades, Entries and Exits. Cross Signals are commonly used for **bbands**, **rsi**, **zscore** crossing some value either above or below two values at different times. See ```help(ta.xsignals)```
* _Directional Movement_ (**dm**) developed by J. Welles Wilder in 1978 attempts to determine which direction the price of an asset is moving. See ```help(ta.dm)```
* _Even Better Sinewave_ (**ebsw**) measures market cycles and uses a low pass filter to remove noise. See: ```help(ta.ebsw)```
* _Jurik Moving Average_ (**jma**) attempts to eliminate noise to see the "true" underlying activity.. See: ```help(ta.jma)```
* _Klinger Volume Oscillator_ (**kvo**) was developed by Stephen J. Klinger. It is designed to predict price reversals in a market by comparing volume to price.. See ```help(ta.kvo)```
* _Schaff Trend Cycle_ (**stc**) is an evolution of the popular MACD incorportating two cascaded stochastic calculations with additional smoothing. See ```help(ta.stc)```
* _Squeeze Pro_ (**squeeze_pro**) is an extended version of "TTM Squeeze" from John Carter. See ```help(ta.squeeze_pro)```
* _Tom DeMark's Sequential_ (**td_seq**) attempts to identify a price point where an uptrend or a downtrend exhausts itself and reverses. Currently exlcuded from ```df.ta.strategy()``` for performance reasons. See ```help(ta.td_seq)```
* _Think or Swim Standard Deviation All_ (**tos_stdevall**) indicator which
returns the standard deviation of data for the entire plot or for the interval
of the last bars defined by the length parameter. See ```help(ta.tos_stdevall)```
* _Vertical Horizontal Filter_ (**vhf**) was created by Adam White to identify trending and ranging markets.. See ```help(ta.vhf)```

<br/>

## **Updated Indicators**

* _Acceleration Bands_ (**accbands**) Argument ```mamode``` renamed to ```mode```. See ```help(ta.accbands)```.
* _ADX_ (**adx**): Added ```mamode``` with default "**RMA**" and with the same ```mamode``` options as TradingView. New argument ```lensig``` so it behaves like TradingView's builtin ADX indicator. See ```help(ta.adx)```.
* _Archer Moving Averages Trends_ (**amat**): Added ```drift``` argument and more descriptive column names.
* _Average True Range_ (**atr**): The default ```mamode``` is now "**RMA**" and with the same ```mamode``` options as TradingView. See ```help(ta.atr)```.
* _Bollinger Bands_ (**bbands**): New argument ```ddoff``` to control the Degrees of Freedom. Also included BB Percent (BBP) as the final column. Default is 0. See ```help(ta.bbands)```.
* _Choppiness Index_ (**chop**): New argument ```ln``` to use Natural Logarithm (True) instead of the Standard Logarithm (False). Default is False.  See ```help(ta.chop)```.
* _Chande Kroll Stop_ (**cksp**): Added ```tvmode``` with default ```True```. When ```tvmode=False```, **cksp** implements "The New Technical Trader" with default values. See ```help(ta.cksp)```.
* _Chande Momentum Oscillator_ (**cmo**): New argument ```talib``` will use TA Lib's version and if TA Lib is installed. Default is True. See ```help(ta.cmo)```.
* _Decreasing_ (**decreasing**): New argument ```strict``` checks if the series is continuously decreasing over period ```length``` with a faster calculation. Default: ```False```. The ```percent``` argument has also been added with default None. See ```help(ta.decreasing)```.
* _Increasing_ (**increasing**): New argument ```strict``` checks if the series is continuously increasing over period ```length``` with a faster calculation. Default: ```False```. The ```percent``` argument has also been added with default None. See ```help(ta.increasing)```.
* _Klinger Volume Oscillator_ (**kvo**): Implements TradingView's Klinger Volume Oscillator version. See ```help(ta.kvo)```.
* _Linear Regression_ (**linreg**): Checks **numpy**'s version to determine whether to utilize the ```as_strided``` method or the newer ```sliding_window_view``` method. This should resolve Issues with Google Colab and it's delayed dependency updates as well as TensorFlow's dependencies as discussed in Issues [#285](https://github.com/twopirllc/pandas-ta/issues/285) and [#329](https://github.com/twopirllc/pandas-ta/issues/329).
* _Moving Average Convergence Divergence_ (**macd**): New argument ```asmode``` enables AS version of MACD. Default is False.  See ```help(ta.macd)```.
* _Parabolic Stop and Reverse_ (**psar**): Bug fix and adjustment to match TradingView's ```sar```. New argument ```af0``` to initialize the Acceleration Factor. See ```help(ta.psar)```.
* _Percentage Price Oscillator_ (**ppo**): Included new argument ```mamode``` as an option. Default is **sma** to match TA Lib. See ```help(ta.ppo)```.
* _True Strength Index_ (**tsi**): Added ```signal``` with default ```13``` and Signal MA Mode ```mamode``` with default **ema** as arguments. See ```help(ta.tsi)```.
* _Volume Profile_ (**vp**): Calculation improvements. See [Pull Request #320](https://github.com/twopirllc/pandas-ta/pull/320) See ```help(ta.vp)```.
* _Volume Weighted Moving Average_ (**vwma**): Fixed bug in DataFrame Extension call. See ```help(ta.vwma)```.
* _Volume Weighted Average Price_ (**vwap**): Added a new parameter called ```anchor```. Default: "D" for "Daily". See [Timeseries Offset Aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for additional options. **Requires** the DataFrame index to be a DatetimeIndex. See ```help(ta.vwap)```.
* _Volume Weighted Moving Average_ (**vwma**): Fixed bug in DataFrame Extension call. See ```help(ta.vwma)```.
* _Z Score_ (**zscore**): Changed return column name from ```Z_length``` to ```ZS_length```. See ```help(ta.zscore)```.