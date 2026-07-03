# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
* **backtesting.py integration** (PR #124 by @Adhyansinghgupta): Bridge function (`ta_bridge`), `SMACrossover` example strategy (`examples/backtesting_py_strategy.py`), and integration tutorial (`docs/tutorials/backtesting_py.md`). Added `backtesting` to `integration` optional dependencies.
* **backtrader integration** (`docs/tutorials/backtrader.md`, `examples/backtrader_strategy.py`): Precompute-then-feed pattern using dynamic `PandasData` subclass (`make_feed()`); covers single-output, multi-output (MACD), and OHLCV-dependent (ATR) indicators. Added `backtrader` to `integration` optional dependencies.
* **vectorbt integration tutorial** (`docs/tutorials/vectorbt.md`): Documented the `df.ta.tsignals()` → `vbt.Portfolio.from_signals()` integration pattern, including multi-output indicator handling and benchmark comparison.
* **Type stubs for `AnalysisIndicators`** (`pandas_ta_classic/core.pyi`): IDE autocomplete and mypy now see full method signatures for every indicator.

### Changed
* **Faster cold import (~3–4×)**: `import pandas_ta_classic` takes ~280 ms (was ~930 ms) because indicator functions load on demand. Public API unchanged.
* **Math operators individually importable**: Each math/trig operator (`add`, `sub`, `mult`, `div`, `rolling_max/min/sum`, `acos`, `cos`, `exp`, `ln`, …) now lives in its own submodule, e.g. `from pandas_ta_classic.math.add import add`.
* **`combination()` delegates to `math.comb`**: Replaces the hand-rolled nCr loop. Signature and results unchanged; the unused `multichoose` kwarg alias was dropped (use `repetition`).
* **`npNaN` alias removed internally**: All modules now use `np.nan` directly. No public API change.
* **All `npX` numpy import aliases removed** (26 aliases, e.g. `npSqrt`, `npArange`, `npNdArray`): modules now use `import numpy as np` + `np.x` directly. Internal only; no public API change.

### Deprecated
* **`CDL_PATTERN_NAMES`**: Use `ALL_PATTERNS` instead. Accessing the old name emits a `DeprecationWarning`.

### Removed
* **Dead code (audit, zero callers)**: `BasePandasObject` and `_check_na_columns()` in `core.py`; CPR helpers `round_to_strike`, `calculate_option_strikes`, `detect_cpr_breakout`, `detect_cpr_rejection`; math utils `geometric_mean`, `log_geometric_mean`, and the hand-rolled `erf`; time utils `df_dates`, `df_month_to_date`/`mtd`, `df_quarter_to_date`/`qtd`; `category_files()`; six unused `CandleArrays` helper methods; the `pkg_resources` Python < 3.8 fallback in `_meta.py`; `tests/context.py` and other unused test scaffolding.
* **`crossany()`**: zero callers and zero tests; use `cross(a, b, above=True) | cross(a, b, above=False)` if needed. Docs entry removed.
* **`high_low_range()` / `real_body()` wrappers** (`utils/_candles.py`): call sites now use `non_zero_range()` directly (`real_body(open_, close)` ≡ `non_zero_range(close, open_)`).
* **`ytd` alias**: use `df_year_to_date()`. Internal `_camelCase2Title` inlined at its single call site.
* **Unused dev/optional dependencies**: `stochastic` (never imported), `cython` (no `.pyx` sources), `pytest-cov` and `pytest-benchmark` (no coverage/benchmark runs), `isort` (black owns formatting). Corresponding `Imports` probes and the Makefile isort step removed.

### Fixed
* **Cross-package indicator imports**: A submodule import could overwrite a re-exported function of the same name on the parent package (e.g. `from pandas_ta_classic.volatility import atr` occasionally returned the `atr` *module* instead of the function). Resolved.
* **Test-fixture regen no longer drops `cmo_14` entries** when running tests without tulipy installed. Generators now merge with existing JSON instead of overwriting.

## [0.6.52] - 2026-06-25

### Added
* **SMC Liquidity Sweep** (`smc_sweep`) (PR #123 by @Adhyansinghgupta): Bullish and bearish smart-money liquidity sweep indicator for momentum analysis. Available as `df.ta.smc_sweep()`.
* **ichimoku `append_span`** (PR #120): New `append_span` parameter on the ichimoku accessor for span-appending control.

### Changed
* **CPR numeric encoding** (PR #117): `calculate_price_position()` and `calculate_cpr_width()` now return `np.int8` instead of string labels. `CPR_POSITION`: `1` (above TC), `0` (inside CPR), `-1` (below BC). `CPR_WIDTH_CLASS`: `1` (wide), `0` (medium), `-1` (narrow). All indicator columns are now numeric.
* **`AnalysisIndicators.__call__` fail-fast**: Removed `except BaseException: pass` swallowing all indicator errors. Indicator exceptions now propagate to callers instead of silently returning `None`. Code that relied on `df.ta.rsi()` never raising must add its own error handling.

### Fixed
* **ichimoku accessor returns DataFrame** (PR #116): `df.ta.ichimoku()` now returns a DataFrame instead of a tuple.
* **macdext silent double-fallback** (PR #121): Eliminated silent double-fallback for KAMA/MAMA matypes in `macdext`.

### Documentation
* **long_run / short_run / xsignals** (PR #119): Clarified `fast`/`slow` as pre-computed Series; added `xa`/`xb` range guidance.
* **beta / correl benchmark** (PR #118): Documented that `benchmark` is required for non-`None` output.

### Dependencies
* **actions/checkout v7** (PR #122): Bumped from v6 to v7.

---

## [0.6.20] - 2026-05-21

### Added
* **`apply_offset` / `apply_fill` helpers** (PR #105): Extracted into `utils/` and all indicators migrated to use them. Removes ~200 lines of duplicated offset/fill logic.
* **Comprehensive candle pattern tests** (PR #106): 59 TA-Lib candle patterns covered by CI tests.
* **Fluent API chaining** (PR #113): DataFrame accessor supports method chaining.
* **Property-based testing** (PR #114): Hypothesis-based tests added to test suite.
* **Full Indicator Name Comments**: All indicator files include full indicator names as comments on line 1 (format: `# Full Indicator Name (ABBREVIATION)`).
* **UV Package Manager Support**: All documentation includes `uv` install instructions alongside `pip`.
* **Native Candlestick Patterns**: Added native `cdl_doji` and `cdl_inside` implementations (no TA-Lib required). Accessible via `df.ta.cdl_doji()`, `df.ta.cdl_inside()`, or `df.ta.cdl_pattern()`.

### Changed
* **Wilder smoothing shared utility** (PR #112 remediation): Extracted `wilder_smooth()` into `utils/_wilder.py` for TA-Lib-exact cumulative smoothing. Used by `dm.py` for PLUS_DM/MINUS_DM parity.
* **Chained EMA helper** (PR #112 remediation): Added `_ema_chain()` to `overlap/ema.py` for consistent NaN-stripping in DEMA, TEMA, T3. Reduces repetitive boilerplate by ~60 lines.
* **Fixture auto-regeneration**: `tests/__init__.py` now regenerates `expected_values.json` and `regression_snapshots.json` on import when TA-Lib is available. `make fixtures` and `make test-all` targets added to `Makefile`.
* **`tal` → `talib` rename**: All test files now import `talib` directly instead of aliasing as `tal`, matching source code convention.
* **Dead code removal**: Removed ~370 instances of unused imports (F401), 8 unused local variables (F841), 1 duplicate method (`test_custom_a`), 65 useless f-string prefixes (F541), and 279 unnecessary UTF-8 encoding declarations (UP009).
* **Code modernization**: Applied pyupgrade (UP) and flake8-return (RET) fixes — `Optional[X]`→`X|None`, `List`→`list`, removed redundant `else` after `return`.
* **Linreg TA-Lib dispatch**: Replaced 6-branch `if`/`elif` chain with `_TALIB_DISPATCH` lookup dictionary.
* **PSAR cleanup**: Removed `import numpy as _np` from function body; uses module-level `np` instead.
* **test_strategy**: Fixed duplicate `test_custom_a` method (was shadowed, never ran). Fixed stale column count assertion.

### Fixed
* **TA-Lib-exact native implementations** (PR #112): Corrected native paths for many indicators to match TA-Lib output exactly.
* **stdev variance calculation** (PR #109): `ddof` default updated to `False` for population variance.
* **Modularity refactor** (PR #108): Issue #46 modularity improvements.
* **Accessor usability** (PR #107): Issue #48 usability and docs fixes.
* **Alpha Vantage integration** (PRs #110, #111): Support for both `alpha_vantage` and `alphaVantage` libraries; improved import checks and error handling.
* **Feature fix PR #112** (PR #115): Follow-up corrections to the TA-Lib-exact implementation from PR #112.

---

## [0.5.44] - 2026-04-30

### Added
* **TA-Lib / tulipy parity indicator set** (PR #104): Added wrappers and native implementations for `msw`, `fosc`, `macdext`, `macdfix`, `rocp`, `rocr`, `rocr100`, `stochf`, `avgprice`, `medprice`, `typprice`, `linregangle`, `linregintercept`, `linregslope`, `mavp`, `md`, `stderr`, `dx`, `edecay`, `plus_dm`, `minus_dm`, `sarext`, `avolume`, `cvi`, `hvol`, `emv`, `marketfi`, `vosc`, and `wad`.
* **Math operator namespace**: Added `pandas_ta_classic/math/` exposing arithmetic operators (`add`, `sub`, `mult`, `div`), rolling operators (`rolling_max`, `rolling_min`, `rolling_sum`), and math transforms for TA-Lib/tulipy compatibility.
* **Oracle parity suites**: Added `tests/test_oracle_talib.py` and `tests/test_oracle_tulipy.py` for cross-library validation on shared SPY fixtures.
* **60 native CDL pattern files** (PR #87): Added `candles/cdl_*.py` implementations for 60 patterns. Combined with `cdl_doji` and `cdl_inside`, total accessible via `cdl_pattern()` is **62**. TA-Lib is **never** used for CDL patterns — native implementations take priority regardless of TA-Lib installation. Added shared `_cdl_math.py` helper.
* **5 Hilbert Transform cycle indicators** (PR #83): `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode`. Shared `_hilbert.py` helper. Cycles category grows from 2 to 7.
* **MAMA / FAMA** (PR #84): MESA Adaptive Moving Average with FAMA output. Uses Ehlers' adaptive phase computation.
* **HT_TRENDLINE** (PR #84): Hilbert Transform Instantaneous Trendline. Added to overlap category.
* **TSF** (PR #85): Time Series Forecast — linear regression projected one period ahead. Matches TA-Lib TSF.
* **Beta** (PR #86): Asset volatility relative to a benchmark series. Matches TA-Lib BETA.
* **CORREL** (PR #86): Pearson Correlation Coefficient between two series. Matches TA-Lib CORREL.
* **ADXR** (PR #89): Average Directional Movement Index Rating — smoothed average of ADX. Matches TA-Lib ADXR.
* **CPR** (PR #77): Central Pivot Range with 4 calculation methods (classic, camarilla, fibonacci, woodie).
* **Chandelier Exit** (`ce`): Trailing stop. `CE_L = rolling_max(high, length) - multiplier × ATR`, `CE_S = rolling_min(low, length) + multiplier × ATR`. Default `length=22`, `multiplier=3.0`. Returns DataFrame with `CE_L_{length}_{multiplier}` and `CE_S_{length}_{multiplier}`.
* **9 New Technical Indicators** (Issue #29): LRSI (Laguerre RSI), PMAX (Price Max), VFI (Volume Flow Indicator), MMAR (Madrid Moving Average Ribbon), Rainbow (Rainbow Charts), PO (Projection Oscillator), DSP (Detrended Synthetic Price), TRIXH (TRIX Histogram), VWMACD (Volume Weighted MACD).

### Changed
* **QQE output columns** (PR #97): `qqe()` now returns 6 columns instead of 3. New columns: `QQEb_l` (long band), `QQEb_s` (short band), `QQEd` (±1 trend direction). **Breaking change**: code relying on a fixed column count or positional indexing must be updated.
* **Updated indicator counts**: 164 indicators in Category (was 151); total 224 with native CDL patterns (was 213).
* **`linreg` breaking default change**: `degrees` kwarg now defaults to `True` (was `False`) to match TA-Lib. Callers using `linreg(close, angle=True)` without `degrees=False` will now receive degrees instead of radians.
* **`stdev`/`variance` breaking default change**: `ddof` now defaults to `0` (population, was `1` sample) to match TA-Lib. Callers relying on sample variance must pass `ddof=1` explicitly.
* **`natr` breaking default change**: `mamode` now defaults to `'rma'` (Wilder smoothing, was `'ema'`) to match TA-Lib. Callers relying on EMA-based NATR must pass `mamode='ema'` explicitly.
* **Native preferred by default** for all indicators with `talib` parameter: Native implementation used by default across all 59 indicators. Callers wanting TA-Lib output must pass `talib=True` explicitly.
* **Enhanced RVGI**: Relative Vigor Index now includes histogram column (RVGI − Signal).
* **Dynamic Category Discovery**: `Category` dict in `_meta.py` now built dynamically from filesystem structure. Auto-discovers previously undocumented indicators (`cdl_doji`, `cdl_inside`, `hwma`, `ma`, `drawdown`, `dm`, `vp`).
* **Python Version Support**: Rolling 5-version support policy (LATEST-4 through LATEST). Managed dynamically via `LATEST_PYTHON_VERSION` in `.github/workflows/ci.yml`.
* **Development Status**: Changed from Beta to Production/Stable in `pyproject.toml`.

### Performance
* **numpy vectorization** (PR #88): 15 indicators (QQE, PSAR, HWC, HT_TRENDLINE, SSF, squeeze, squeeze_pro, RVGI, TD_SEQ, TOS_STDEVALL, ALMA, SINWMA, SWMA, TRIMA, VIDYA) now use `sliding_window_view` instead of pandas `.iloc` loops. Adds shared `_sliding_weighted_ma()` utility.
* **Numba JIT acceleration** (PR #99): 10 indicators (SSF, MCGD, HWMA, RSX, PSAR, Supertrend, QQE, and 3 more) gain optional `@njit(cache=True)` via numba. Graceful no-op fallback in `utils/_njit.py`. Enable with `pip install pandas-ta-classic[performance]`. Speedups: RSX 230×, HWMA 70×, MCGD 43×, SSF 42×, Supertrend 13×, QQE 10×, PSAR 6×.

### Fixed
* **Oracle test policy**: Replaced skip-based oracle tests with explicit assertions for value equivalence or documented divergence. Zero skipped tests.
* **TA-Lib compatibility paths**: Added/updated `talib=True` behavior for `macdfix`, `psar`, `stochrsi`, `plus_dm`, `minus_dm`.
* **Indicator formula parity**: Corrected `edecay` (multiplicative exponential decay), `emv` scaling (`divisor=10000`).

---

## [0.4.47] - 2026-03-17

### Fixed
* **Dependency cleanup, pandas 3.0 compat, Windows pool fix** (PR #79).
* **Initialization and edge-case bugs** across 11 indicators (PR #80).
* **Numerical bugs** in `linreg`, `tsi`, `brar`, `bbands`, `cti` (PR #94).
* **TA-Lib reference alignment** for 8 indicators (PR #81).
* **Rolling stats cross-version determinism**: Replaced pandas rolling stats with numpy (PR #82).
* **None-guards** to prevent crashes on short/invalid input across 26 files (PR #95).
* **print() → logging** across library code (PR #93). **Breaking**: code catching print output must switch to log capture.
* **Strategy dataclass bugs** (PR #96). **Breaking**: some Strategy API surface changed.
* **Code of Conduct Contact**: Updated enforcement contact from original maintainer email to GitHub Issues.
* **PyPI Release Version**: Set `SETUPTOOLS_SCM_PRETEND_VERSION` from the release tag to prevent `.dev0` versions being published.
* **Version Scheme**: Changed from post-release to default scheme. Tagged releases get clean versions (e.g., `0.4.47`); commits after a tag get `.devN` suffix.
* **PyPI Image Display**: Updated README.md to use absolute GitHub URLs so images render correctly on the PyPI package page.
* **CI/CD Shallow Clone**: Added `fetch-depth: 0` to all checkout steps to ensure full git history for setuptools-scm.
* **Version Fallback**: Changed fallback from `0.0.0.dev0` to `0.0.0` (PEP 440 compliant).

### Added
* **Automatic Version Management**: Version now determined from git tags via `setuptools-scm`. Development builds get `.dev` suffix; tagged releases use the tag exactly. See CONTRIBUTING.md for documentation.

---

## [0.3.78] - 2026-02-27

### Changed
* **Type hints** (PR #67 by @rmarcink): Type annotations added across all public function signatures.

---

## Pre-0.3.78 — Historical

> Items below describe the state of the library as inherited from the original `pandas-ta` project and pre-2026 development. Preserved for historical reference.

### General
* A __Strategy__ Class to help name and group your favorite indicators.
* If a **TA Lib** is already installed, Pandas TA will run TA Lib's version. (**BETA**)
* Some indicators have had their ```mamode``` _kwarg_ updated with more _moving average_ choices with the **Moving Average Utility** function ```ta.ma()```. For simplicity, all _choices_ are single source _moving averages_. This is primarily an internal utility used by indicators that have a ```mamode``` _kwarg_. This includes indicators: _accbands_, _amat_, _aobv_, _atr_, _bbands_, _bias_, _efi_, _hilo_, _kc_, _natr_, _qqe_, _rvi_, and _thermo_; the default ```mamode``` parameters have not changed. However, ```ta.ma()``` can be used by the user as well if needed. For more information: ```help(ta.ma)```
    * **Moving Average Choices**: dema, ema, fwma, hma, linreg, midpoint, pwma, rma, sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma.
* An _experimental_ and independent __Watchlist__ Class located in the [Examples](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples/watchlist.py) Directory that can be used in conjunction with the new __Strategy__ Class.
* _Linear Regression_ (**linear_regression**) is a new utility method for Simple Linear Regression using _Numpy_ or _Scikit Learn_'s implementation.
* Added utility/convience function, ```to_utc```, to convert the DataFrame index to UTC. See: ```help(ta.to_utc)``` **Now** as a Pandas TA DataFrame Property to easily convert the DataFrame index to UTC.

<br />

### Breaking / Depreciated Indicators
* _Trend Return_ (**trend_return**) has been removed and replaced with **tsignals**. When given a trend Series like ```close > sma(close, 50)``` it returns the Trend, Trade Entries and Trade Exits of that trend to make it compatible with [**vectorbt**](https://github.com/polakowo/vectorbt) by setting ```asbool=True``` to get boolean Trade Entries and Exits. See ```help(ta.tsignals)```

<br/>

### New Indicators
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

### Updated Indicators

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
