Indicators Reference
===================

**Pandas TA Classic** includes 180 indicators in the Category system plus 62 CDL patterns accessible via ``cdl_pattern()`` (240 unique total — ``cdl_doji`` and ``cdl_inside`` are counted in both) organized into the following categories:

* **Candles** (67) - 62 CDL patterns via ``cdl_pattern()`` + ``cdl_doji``, ``cdl_inside``, ``cdl_z``, ``ha``, ``cdl_pattern`` as Category entries (``cdl_doji`` and ``cdl_inside`` appear in both counts)
* **Cycles** (8) - Cycle-based and Hilbert Transform indicators  
* **Momentum** (53) - Momentum and oscillator indicators
* **Overlap** (46) - Moving averages and trend-following indicators
* **Performance** (3) - Performance and return metrics
* **Statistics** (14) - Statistical analysis functions
* **Trend** (27) - Trend identification and direction indicators
* **Volatility** (18) - Volatility and range-based indicators
* **Volume** (20) - Volume analysis indicators
* **Math** (28) - Element-wise math operators and transforms

.. note::
   The category system now uses **dynamic discovery** - indicators are automatically detected from the package structure, ensuring the list is always up-to-date with available indicators.

Candles (67)
------------

Candlestick patterns for identifying market sentiment and potential reversals.

All 62 CDL patterns have native Python implementations. The dispatch order inside ``cdl_pattern()`` is: **native first → TA-Lib fallback → warning**. Because every pattern in ``ALL_PATTERNS`` has a native implementation, the TA-Lib branch is never reached in practice. Patterns are accessible via ``df.ta.cdl_pattern(name=...)``, or for ``doji`` and ``inside`` specifically via their dedicated accessor methods.

.. code-block:: python

    # All 62 patterns at once (native only, no TA-Lib needed)
    df = df.ta.cdl_pattern(name="all")

    # Single pattern
    df = df.ta.cdl_pattern(name="engulfing")

    # Multiple patterns
    df = df.ta.cdl_pattern(name=["hammer", "morningstar", "engulfing"])

    # Dedicated accessor methods (only these two have them)
    result = df.ta.cdl_doji()
    result = df.ta.cdl_inside()

.. note::
   Native implementations take priority in ``cdl_pattern()``'s dispatch chain. TA-Lib is only used as a fallback for any pattern that lacks a native implementation — which is none of the 62 patterns in ``ALL_PATTERNS``.

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

.. note::
   ``cdl_doji()`` and ``cdl_inside()`` have dedicated implementations accessible as ``df.ta.cdl_doji()`` and ``df.ta.cdl_inside()``.

Other candle indicators:

* *Heikin-Ashi*: **ha** — ``df.ta.ha()`` — not a CDL pattern, not valid as ``cdl_pattern(name=...)``
* *Z Score*: **cdl_z** — ``df.ta.cdl_z()`` — Z-score normalisation of candle bodies, not a CDL pattern

.. note::
   **TA-Lib and core indicators**: For 34 non-candle indicators (``ema``, ``sma``, ``rsi``, ``macd``, ``obv``, ``atr``, and others), TA-Lib’s implementation is used **by default** when TA-Lib is installed, for numerical consistency with TA-Lib-based workflows. Every such indicator accepts a ``talib=False`` kwarg to force the native implementation:

   .. code-block:: python

       # Uses TA-Lib EMA if installed (default behaviour)
       ema = df.ta.ema(length=20)

       # Force native implementation regardless of TA-Lib
       ema = df.ta.ema(length=20, talib=False)

       # Indicators with TA-Lib passthrough:
       # ad, adosc, apo, aroon, atr, bbands, bop, cci, cmo, dema, dm,
       # ema, hlc3, macd, mfi, midpoint, midprice, mom, natr, obv, ppo,
       # roc, rsi, sma, stdev, t3, tema, trima, true_range, uo,
       # variance, wcp, willr, wma

Cycles (8)
----------

* *Detrended Synthetic Price*: **dsp**
* *Even Better Sinewave*: **ebsw**
* *Hilbert Transform — Dominant Cycle Period*: **ht_dcperiod**
* *Hilbert Transform — Dominant Cycle Phase*: **ht_dcphase**
* *Hilbert Transform — Phasor Components*: **ht_phasor** (returns InPhase + Quadrature)
* *Hilbert Transform — SineWave*: **ht_sine** (returns Sine + LeadSine)
* *Hilbert Transform — Trend vs Cycle Mode*: **ht_trendmode**
* *Mesa Sine Wave*: **msw** (returns MSW_SINE + MSW_LEAD; period-based DFT cycle detector)

Momentum (53)
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
* *Forecast Oscillator*: **fosc**
* *Inertia*: **inertia**
* *KDJ*: **kdj**
* *KST Oscillator*: **kst**
* *Linear Regression RSI*: **lrsi**
* *Moving Average Convergence Divergence*: **macd**
* *MACD Extended*: **macdext** (MACD with controllable MA type per line; MA types: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3)
* *MACD Fixed*: **macdfix** (MACD with fixed 12/26 periods; only signal period is configurable; uses TA-Lib ``MACDFIX`` when available)
* *Momentum*: **mom**
* *Pretty Good Oscillator*: **pgo**
* *Projection Oscillator*: **po**
* *Percentage Price Oscillator*: **ppo**
* *Psychological Line*: **psl**
* *Percentage Volume Oscillator*: **pvo**
* *Quantitative Qualitative Estimation*: **qqe** (returns QQE, QQEs, QQEl, QQEb_l, QQEb_s, QQEd)
* *Rate of Change*: **roc**
* *Rate of Change Percentage*: **rocp**
* *Rate of Change Ratio*: **rocr**
* *Rate of Change Ratio * 100*: **rocr100**
* *Relative Strength Index*: **rsi**
* *Relative Strength Xtra*: **rsx**
* *Relative Vigor Index*: **rvgi**
* *Schaff Trend Cycle*: **stc**
* *Slope*: **slope**
* *SMI Ergodic*: **smi**
* *Squeeze*: **squeeze** (Default is John Carter's. Enable Lazybear's with ``lazybear=True``)
* *Squeeze Pro*: **squeeze_pro**
* *Stochastic Oscillator*: **stoch**
* *Stochastic Fast*: **stochf**
* *Stochastic RSI*: **stochrsi**
* *TD Sequential*: **td_seq** (Excluded from ``df.ta.strategy()``)
* *Trix*: **trix**
* *TRIX Histogram*: **trixh**
* *True strength index*: **tsi**
* *Ultimate Oscillator*: **uo**
* *Volume Weighted MACD*: **vwmacd**
* *Williams %R*: **willr**

Overlap (46)
------------

Moving averages and trend-following indicators:

* *Arnaud Legoux Moving Average*: **alma**
* *Average Price (OHLC/4)*: **avgprice** (arithmetic mean of open, high, low, close; equivalent to TA-Lib ``AVGPRICE`` and tulipy ``avgprice``)
* *Double Exponential Moving Average*: **dema**
* *Exponential Moving Average*: **ema**
* *Fibonacci's Weighted Moving Average*: **fwma**
* *Gann High-Low Activator*: **hilo**
* *High-Low Average*: **hl2**
* *High-Low-Close Average*: **hlc3** (Commonly known as 'Typical Price')
* *Hull Exponential Moving Average*: **hma**
* *Hilbert Transform Instantaneous Trendline*: **ht_trendline**
* *Holt-Winter Moving Average*: **hwma**
* *Ichimoku Kinkō Hyō*: **ichimoku** (Returns two DataFrames. ``lookahead=False`` drops the Chikou Span Column)
* *Jurik Moving Average*: **jma**
* *Kaufman's Adaptive Moving Average*: **kama**
* *Linear Regression*: **linreg**
* *Linear Regression Angle*: **linregangle** (angle in degrees of the linear regression slope)
* *Linear Regression Intercept*: **linregintercept** (y-intercept of the linear regression line)
* *Linear Regression Slope*: **linregslope** (slope of the linear regression line)
* *Moving Average*: **ma** (Generic moving average selector)
* *MESA Adaptive Moving Average*: **mama** (returns MAMA + FAMA)
* *Moving Average with Variable Period*: **mavp**
* *Madrid Moving Average Ribbon*: **mmar**
* *Median Price (H+L)/2*: **medprice** (arithmetic mean of high and low; equivalent to TA-Lib ``MEDPRICE`` and tulipy ``medprice``)
* *McGinley Dynamic*: **mcgd**
* *Midpoint*: **midpoint**
* *Midprice*: **midprice**
* *Open-High-Low-Close Average*: **ohlc4**
* *Pascal's Weighted Moving Average*: **pwma**
* *Rainbow Moving Average*: **rainbow**
* *WildeR's Moving Average*: **rma**
* *Sine Weighted Moving Average*: **sinwma**
* *Simple Moving Average*: **sma**
* *Ehler's Super Smoother Filter*: **ssf**
* *Supertrend*: **supertrend**
* *Symmetric Weighted Moving Average*: **swma**
* *T3 Moving Average*: **t3**
* *Triple Exponential Moving Average*: **tema**
* *Time Series Forecast*: **tsf**
* *Triangular Moving Average*: **trima**
* *Typical Price (H+L+C)/3*: **typprice** (arithmetic mean of high, low, close; equivalent to TA-Lib ``TYPPRICE`` and tulipy ``typprice``)
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

Statistics (14)
---------------

Statistical analysis functions:

* *Beta*: **beta** (asset volatility relative to a benchmark series)
* *Pearson Correlation Coefficient*: **correl**
* *Entropy*: **entropy**
* *Kurtosis*: **kurtosis**  
* *Mean Absolute Deviation*: **mad**
* *Mean Deviation*: **md** (equivalent to tulipy ``md``; rolling mean absolute deviation from mean)
* *Median*: **median**
* *Quantile*: **quantile**
* *Skew*: **skew**
* *Standard Deviation*: **stdev**
* *Standard Error*: **stderr**
* *Think or Swim Standard Deviation All*: **tos_stdevall**
* *Variance*: **variance**
* *Z Score*: **zscore**

Trend (27)
----------

Trend identification and direction indicators:

* *Average Directional Movement Index*: **adx** (Also includes **dmp** and **dmn**)
* *Average Directional Movement Index Rating*: **adxr**
* *Archer Moving Averages Trends*: **amat**
* *Aroon & Aroon Oscillator*: **aroon**
* *Choppiness Index*: **chop**
* *Chande Kroll Stop*: **cksp**
* *Central Pivot Range*: **cpr** / **cpr_option** (4 pivot methods: standard, camarilla, fibonacci, woodie)
* *Decay*: **decay** (Formally: **linear_decay**)
* *Decreasing*: **decreasing**
* *Detrended Price Oscillator*: **dpo** (Set ``lookahead=False`` to disable centering)
* *Directional Index*: **dx**
* *Exponential Decay*: **edecay** (multiplicative exponential decay; equivalent to tulipy ``edecay``)
* *Increasing*: **increasing**
* *Long Run*: **long_run**
* *Minus Directional Movement*: **minus_dm** (raw Wilder-smoothed −DM before ATR normalisation; uses TA-Lib ``MINUS_DM`` by default)
* *Parabolic Stop and Reverse*: **psar** (pass ``talib=True`` for exact TA-Lib ``SAR`` output)
* *Plus Directional Movement*: **plus_dm** (raw Wilder-smoothed +DM before ATR normalisation; uses TA-Lib ``PLUS_DM`` by default)
* *Price Max*: **pmax**
* *Q Stick*: **qstick**
* *Parabolic SAR Extended*: **sarext**
* *Short Run*: **short_run**
* *Trend Signals*: **tsignals**
* *TTM Trend*: **ttm_trend**
* *Vertical Horizontal Filter*: **vhf**
* *Vortex*: **vortex**
* *Cross Signals*: **xsignals**

Volatility (18)
---------------

Volatility and range-based indicators:

* *Aberration*: **aberration**
* *Acceleration Bands*: **accbands**
* *Annualised Volatility*: **avolume** (rolling annualised log-return standard deviation; ``length * sqrt(252)``-scaled)
* *Average True Range*: **atr**
* *Bollinger Bands*: **bbands**
* *Chandelier Exit*: **ce**
* *Chaikins Volatility*: **cvi**
* *Donchian Channel*: **donchian**
* *Historical Volatility*: **hvol** (Annualized; ``annualization=252`` by default)
* *Holt-Winter Channel*: **hwc**
* *Keltner Channel*: **kc**
* *Mass Index*: **massi**
* *Normalized Average True Range*: **natr**
* *Price Distance*: **pdist**
* *Relative Volatility Index*: **rvi**
* *Elder's Thermometer*: **thermo**
* *True Range*: **true_range**
* *Ulcer Index*: **ui**

Volume (20)
-----------

Volume analysis indicators:

* *Accumulation/Distribution Index*: **ad**
* *Accumulation/Distribution Oscillator*: **adosc**
* *Archer On-Balance Volume*: **aobv**
* *Chaikin Money Flow*: **cmf**
* *Elder's Force Index*: **efi**
* *Ease of Movement*: **eom**
* *Ease of Movement (EMV)*: **emv** (equivalent to tulipy ``emv``; uses ``divisor=10000`` for scale; rolling-averaged variant with ``length`` parameter)
* *Klinger Volume Oscillator*: **kvo**
* *Market Facilitation Index*: **marketfi**
* *Money Flow Index*: **mfi**
* *Negative Volume Index*: **nvi**
* *On-Balance Volume*: **obv**
* *Positive Volume Index*: **pvi**
* *Price-Volume*: **pvol**
* *Price Volume Rank*: **pvr**
* *Price Volume Trend*: **pvt**
* *Volume Flow Indicator*: **vfi**
* *Volume Oscillator*: **vosc**
* *Volume Profile*: **vp**
* *Williams Accumulation/Distribution*: **wad**

Math (28)
---------

Element-wise arithmetic operators, rolling aggregation, and mathematical transforms.
All functions are available via ``df.ta.<name>()``.

Element-wise binary operators:

* *Add*: **add** — element-wise addition of two series
* *Subtract*: **sub** — element-wise subtraction of two series
* *Multiply*: **mult** — element-wise multiplication of two series
* *Divide*: **div** — element-wise division of two series

Rolling aggregation operators:

* *Rolling Maximum*: **rolling_max** — rolling maximum over a window
* *Rolling Minimum*: **rolling_min** — rolling minimum over a window
* *Rolling Sum*: **rolling_sum** — rolling sum over a window

Mathematical transforms (wrapping NumPy / SciPy math, TA-Lib ``MATH TRANSFORM`` and ``MATH OPERATORS`` group, and tulipy equivalents):

* *Inverse Sine*: **asin**
* *Inverse Cosine*: **acos**
* *Inverse Tangent*: **atan**
* *Ceiling*: **ceil**
* *Cosine*: **cos**
* *Hyperbolic Cosine*: **cosh**
* *Exponential*: **exp**
* *Floor*: **floor**
* *Natural Logarithm*: **ln**
* *Logarithm Base 10*: **log10**
* *Sine*: **sin**
* *Hyperbolic Sine*: **sinh**
* *Square Root*: **sqrt**
* *Tangent*: **tan**
* *Hyperbolic Tangent*: **tanh**

Utility / signal functions (accessible directly or via ``df.ta``):

* *Crossover*: **crossover** (returns Boolean Series that is True on the bar where ``a`` crosses above ``b``)
* *Crossany*: **crossany** (returns Boolean Series that is True on any bar where ``a`` and ``b`` cross in either direction)
* *Lag*: **lag** (returns a Series offset by ``n`` periods; equivalent to tulipy ``lag``)