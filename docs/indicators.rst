Indicators Reference
===================

**Pandas TA Classic** includes 141 indicators and utility functions plus 62 TA-Lib candlestick patterns (203 total) organized into the following categories:

* **Candles** (67) - Candlestick patterns for market sentiment analysis (5 native + 62 TA-Lib patterns)  
* **Cycles** (1) - Cycle-based technical indicators  
* **Momentum** (41) - Momentum and oscillator indicators
* **Overlap** (34) - Moving averages and trend-following indicators
* **Performance** (3) - Performance and return metrics
* **Statistics** (10) - Statistical analysis functions
* **Trend** (18) - Trend identification and direction indicators
* **Utility** (10) - Helper and utility functions
* **Volatility** (14) - Volatility and range-based indicators
* **Volume** (15) - Volume analysis indicators

.. note::
   The category system now uses **dynamic discovery** - indicators are automatically detected from the package structure, ensuring the list is always up-to-date with available indicators.

Candles (67)
------------

Candlestick patterns for identifying market sentiment and potential reversals. This includes 5 native patterns (**cdl_doji**, **cdl_inside**, **cdl_pattern**, **cdl_z**, **ha**) plus 62 TA-Lib patterns. Patterns that are **not bold** require TA-Lib to be installed:

- Using ``uv``: ``uv pip install TA-Lib``
- Using ``pip``: ``pip install TA-Lib``

.. code-block:: python

    # Get all candle patterns (This is the default behaviour)
    df = df.ta.cdl_pattern(name="all")

    # Get only one pattern
    df = df.ta.cdl_pattern(name="doji")

    # Get some patterns
    df = df.ta.cdl_pattern(name=["doji", "inside"])

Available patterns:

* 2crows, 3blackcrows, 3inside, 3linestrike, 3outside, 3starsinsouth, 3whitesoldiers
* abandonedbaby, advanceblock, belthold, breakaway, closingmarubozu, concealbabyswall, counterattack
* darkcloudcover, **doji**, dojistar, dragonflydoji, engulfing, eveningdojistar, eveningstar
* gapsidesidewhite, gravestonedoji, hammer, hangingman, harami, haramicross, highwave
* hikkake, hikkakemod, homingpigeon, identical3crows, **inside**, inneck, invertedhammer
* kicking, kickingbylength, ladderbottom, longleggeddoji, longline, marubozu, matchinglow
* mathold, morningdojistar, morningstar, onneck, piercing, rickshawman, risefall3methods
* separatinglines, shootingstar, shortline, spinningtop, stalledpattern, sticksandwich
* takuri, tasukigap, thrusting, tristar, unique3river, upsidegap2crows, xsidegap3methods
* *Heikin-Ashi*: **ha**
* *Z Score*: **cdl_z**

.. note::
   **Bold patterns** are native implementations. Use ``df.ta.cdl_doji()`` or ``df.ta.cdl_inside()`` to access the native doji and inside bar patterns directly.

Cycles (1)
----------

* *Even Better Sinewave*: **ebsw**

Momentum (41)
-------------

Momentum and oscillator indicators for measuring the speed of price changes:

* *Awesome Oscillator*: **ao**
* *Absolute Price Oscillator*: **apo** 
* *Bias*: **bias**
* *Balance of Power*: **bop**
* *BRAR*: **brar**
* *Commodity Channel Index*: **cci**
* *Chande Forecast Oscillator*: **cfo**
* *Center of Gravity*: **cg**
* *Chande Momentum Oscillator*: **cmo**
* *Coppock Curve*: **coppock**
* *Correlation Trend Indicator*: **cti** (wrapper for ``ta.linreg(series, r=True)``)
* *Directional Movement*: **dm**
* *Efficiency Ratio*: **er**
* *Elder Ray Index*: **eri**
* *Fisher Transform*: **fisher**
* *Inertia*: **inertia**
* *KDJ*: **kdj**
* *KST Oscillator*: **kst**
* *Moving Average Convergence Divergence*: **macd**
* *Momentum*: **mom**
* *Pretty Good Oscillator*: **pgo**
* *Percentage Price Oscillator*: **ppo**
* *Psychological Line*: **psl**
* *Percentage Volume Oscillator*: **pvo**
* *Quantitative Qualitative Estimation*: **qqe**
* *Rate of Change*: **roc**
* *Relative Strength Index*: **rsi**
* *Relative Strength Xtra*: **rsx**
* *Relative Vigor Index*: **rvgi**
* *Schaff Trend Cycle*: **stc**
* *Slope*: **slope**
* *SMI Ergodic*: **smi**
* *Squeeze*: **squeeze** (Default is John Carter's. Enable Lazybear's with ``lazybear=True``)
* *Squeeze Pro*: **squeeze_pro**
* *Stochastic Oscillator*: **stoch**
* *Stochastic RSI*: **stochrsi**
* *TD Sequential*: **td_seq** (Excluded from ``df.ta.strategy()``)
* *Trix*: **trix**
* *True strength index*: **tsi**
* *Ultimate Oscillator*: **uo**
* *Williams %R*: **willr**

Overlap (34)
------------

Moving averages and trend-following indicators:

* *Arnaud Legoux Moving Average*: **alma**
* *Double Exponential Moving Average*: **dema**
* *Exponential Moving Average*: **ema**
* *Fibonacci's Weighted Moving Average*: **fwma**
* *Gann High-Low Activator*: **hilo**
* *High-Low Average*: **hl2**
* *High-Low-Close Average*: **hlc3** (Commonly known as 'Typical Price')
* *Hull Exponential Moving Average*: **hma**
* *Holt-Winter Moving Average*: **hwma**
* *Ichimoku Kinkō Hyō*: **ichimoku** (Returns two DataFrames. ``lookahead=False`` drops the Chikou Span Column)
* *Jurik Moving Average*: **jma**
* *Kaufman's Adaptive Moving Average*: **kama**
* *Linear Regression*: **linreg**
* *Moving Average*: **ma** (Generic moving average selector)
* *McGinley Dynamic*: **mcgd**
* *Midpoint*: **midpoint**
* *Midprice*: **midprice**
* *Open-High-Low-Close Average*: **ohlc4**
* *Pascal's Weighted Moving Average*: **pwma**
* *WildeR's Moving Average*: **rma**
* *Sine Weighted Moving Average*: **sinwma**
* *Simple Moving Average*: **sma**
* *Ehler's Super Smoother Filter*: **ssf**
* *Supertrend*: **supertrend**
* *Symmetric Weighted Moving Average*: **swma**
* *T3 Moving Average*: **t3**
* *Triple Exponential Moving Average*: **tema**
* *Triangular Moving Average*: **trima**
* *Variable Index Dynamic Average*: **vidya**
* *Volume Weighted Average Price*: **vwap** (**Requires** the DataFrame index to be a DatetimeIndex)
* *Volume Weighted Moving Average*: **vwma**
* *Weighted Closing Price*: **wcp**
* *Weighted Moving Average*: **wma**
* *Zero Lag Moving Average*: **zlma**

Performance (3)
---------------

Performance and return metrics. Use parameter ``cumulative=True`` for cumulative results:

* *Draw Down*: **drawdown**
* *Log Return*: **log_return**
* *Percent Return*: **percent_return**

Statistics (10)
---------------

Statistical analysis functions:

* *Entropy*: **entropy**
* *Kurtosis*: **kurtosis**  
* *Mean Absolute Deviation*: **mad**
* *Median*: **median**
* *Quantile*: **quantile**
* *Skew*: **skew**
* *Standard Deviation*: **stdev**
* *Think or Swim Standard Deviation All*: **tos_stdevall**
* *Variance*: **variance**
* *Z Score*: **zscore**

Trend (18)
----------

Trend identification and direction indicators:

* *Average Directional Movement Index*: **adx** (Also includes **dmp** and **dmn**)
* *Archer Moving Averages Trends*: **amat**
* *Aroon & Aroon Oscillator*: **aroon**
* *Choppiness Index*: **chop**
* *Chande Kroll Stop*: **cksp**
* *Decay*: **decay** (Formally: **linear_decay**)
* *Decreasing*: **decreasing**
* *Detrended Price Oscillator*: **dpo** (Set ``lookahead=False`` to disable centering)
* *Increasing*: **increasing**
* *Long Run*: **long_run**
* *Parabolic Stop and Reverse*: **psar**
* *Q Stick*: **qstick**
* *Short Run*: **short_run**
* *Trend Signals*: **tsignals**
* *TTM Trend*: **ttm_trend**
* *Vertical Horizontal Filter*: **vhf**
* *Vortex*: **vortex**
* *Cross Signals*: **xsignals**

Utility (10)
------------

Helper and utility functions:

* *Above*: **above**
* *Above Value*: **above_value**
* *Below*: **below**
* *Below Value*: **below_value**
* *Cross*: **cross**
* *Cross Value*: **cross_value**
* *Decreasing*: **decreasing**
* *Increasing*: **increasing**
* *Long Run*: **long_run**
* *Short Run*: **short_run**

Volatility (14)
---------------

Volatility and range-based indicators:

* *Aberration*: **aberration**
* *Acceleration Bands*: **accbands**
* *Average True Range*: **atr**
* *Bollinger Bands*: **bbands**
* *Donchian Channel*: **donchian**
* *Holt-Winter Channel*: **hwc**
* *Keltner Channel*: **kc**
* *Mass Index*: **massi**
* *Normalized Average True Range*: **natr**
* *Price Distance*: **pdist**
* *Relative Volatility Index*: **rvi**
* *Elder's Thermometer*: **thermo**
* *True Range*: **true_range**
* *Ulcer Index*: **ui**

Volume (15)
-----------

Volume analysis indicators:

* *Accumulation/Distribution Index*: **ad**
* *Accumulation/Distribution Oscillator*: **adosc**
* *Archer On-Balance Volume*: **aobv**
* *Chaikin Money Flow*: **cmf**
* *Elder's Force Index*: **efi**
* *Ease of Movement*: **eom**
* *Klinger Volume Oscillator*: **kvo**
* *Money Flow Index*: **mfi**
* *Negative Volume Index*: **nvi**
* *On-Balance Volume*: **obv**
* *Positive Volume Index*: **pvi**
* *Price-Volume*: **pvol**
* *Price Volume Rank*: **pvr**
* *Price Volume Trend*: **pvt**
* *Volume Profile*: **vp**