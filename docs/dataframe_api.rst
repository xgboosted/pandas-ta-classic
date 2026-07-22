DataFrame Properties and Methods
================================

Programming Conventions
-----------------------

**Pandas TA Classic** has three primary "styles" of processing Technical Indicators for your use case and/or requirements. They are: *Standard*, *DataFrame Extension*, and the *Pandas TA Classic Strategy*. Each with increasing levels of abstraction for ease of use.

Standard Usage
~~~~~~~~~~~~~~

You explicitly define the input columns and take care of the output.

.. note::
   Column names are **case-sensitive** in Standard usage. Pass the exact column name from your DataFrame (e.g. ``df["Close"]`` or ``df["close"]``).

.. code-block:: python

    import pandas_ta_classic as ta
    
    # Simple Moving Average
    sma10 = ta.sma(df["Close"], length=10)
    # Returns a Series with name: SMA_10
    
    # Donchian Channel
    donchiandf = ta.donchian(df["HIGH"], df["low"], lower_length=10, upper_length=15)
    # Returns a DataFrame named DC_10_15 and column names: DCL_10_15, DCM_10_15, DCU_10_15
    
    # Chaining indicators
    ema10_ohlc4 = ta.ema(ta.ohlc4(df["Open"], df["High"], df["Low"], df["Close"]), length=10)

DataFrame Extension
~~~~~~~~~~~~~~~~~~~

Calling ``df.ta`` will automatically lowercase *OHLCVA* to *ohlcva*: *open, high, low, close, volume*, *adj_close*. By default, ``df.ta`` will use the *ohlcva* for the indicator arguments.

.. code-block:: python

    # Simple usage
    sma10 = df.ta.sma(length=10)
    # Returns a Series with name: SMA_10
    
    # With chaining and suffix
    ema10_ohlc4 = df.ta.ema(close=df.ta.ohlc4(), length=10, suffix="OHLC4")
    # Returns a Series with name: EMA_10_OHLC4
    
    # Appending results directly to DataFrame
    df.ta.sma(length=10, append=True)
    df.ta.donchian(lower_length=10, upper_length=15, append=True)

Properties
----------

adjusted
~~~~~~~~

.. code-block:: python

    # Set ta to default to an adjusted column, 'adj_close', overriding default 'close'.
    df.ta.adjusted = "adj_close"
    df.ta.sma(length=10, append=True)

    # To reset back to 'close', set adjusted back to None.
    df.ta.adjusted = None

categories
~~~~~~~~~~

.. code-block:: python

    # List of Pandas TA categories.
    df.ta.categories

cores
~~~~~

.. code-block:: python

    # Set the number of cores to use for strategy multiprocessing
    # Defaults to the number of cpus you have.
    df.ta.cores = 4

    # Set the number of cores to 0 for no multiprocessing.
    df.ta.cores = 0

    # Returns the number of cores you set or your default number of cpus.
    df.ta.cores

datetime_ordered
~~~~~~~~~~~~~~~~

.. code-block:: python

    # The 'datetime_ordered' property returns True if the DataFrame
    # index is of Pandas datetime64 and df.index[0] < df.index[-1].
    # Otherwise it returns False.
    df.ta.datetime_ordered

exchange
~~~~~~~~

.. code-block:: python

    # Sets the Exchange
    df.ta.exchange = "NYSE"

    # Returns the Exchange
    df.ta.exchange

last_run
~~~~~~~~

.. code-block:: python

    # Returns the time it took to run the last indicator or strategy
    df.ta.last_run

reverse
~~~~~~~

.. code-block:: python

    # The 'reverse' is a helper property that returns the DataFrame
    # in reverse order; useful for some indicators
    df.ta.reverse

prefix & suffix
~~~~~~~~~~~~~~~

``prefix`` and ``suffix`` are **per-call kwargs**, not settable properties on the
accessor.  Pass them directly to any indicator call to customise the output
column name(s).

.. code-block:: python

    # Prefix the output column name
    df.ta.sma(length=10, prefix="MY")       # column: MY_SMA_10

    # Suffix the output column name
    df.ta.sma(length=10, suffix="SLOW")     # column: SMA_10_SLOW

    # Both together
    df.ta.sma(length=10, prefix="MY", suffix="SLOW")  # column: MY_SMA_10_SLOW

    # Works with multi-column results too (e.g. MACD)
    df.ta.macd(prefix="MY", suffix="v1")   # columns: MY_MACD_12_26_9_v1, …

time_range
~~~~~~~~~~

``time_range`` controls the **time unit** used when ``df.ta.time_range`` is
read back as a numeric value (e.g. for annualisation).  Valid values are
``"years"`` (default), ``"months"``, ``"weeks"``, ``"days"``, ``"hours"``,
``"minutes"``, and ``"seconds"``.

.. note::
   Any unrecognised string silently falls back to ``"years"`` (matches the
   behaviour of the underlying ``total_time()`` helper).

.. code-block:: python

    # Return the span of the DataFrame in years (default)
    df.ta.time_range = "years"
    print(df.ta.time_range)   # e.g. 2.5

    # Switch to months
    df.ta.time_range = "months"
    print(df.ta.time_range)   # e.g. 30.1

    # Reset to the default unit
    df.ta.time_range = "years"

to_utc
~~~~~~

``to_utc`` is a **read-only property** (not a method).  Accessing it
converts the DataFrame index to UTC in place.

.. code-block:: python

    # Convert DataFrame index to UTC (accesses the property — no parentheses)
    df.ta.to_utc

Methods
-------

constants
~~~~~~~~~

``constants(append: bool, values: list | np.ndarray)`` — adds or removes
constant horizontal-line columns from the DataFrame.

.. deprecated:: 0.6.53
   ``df.ta.constants()`` is deprecated and will be removed in a future release;
   adding horizontal charting lines is out of scope for a technical-analysis
   library. Assign the columns directly instead, e.g. ``df["0"] = 0``.

.. code-block:: python

    import numpy as np

    # Add constants 0 and 100 as new columns
    df.ta.constants(True, [0, 100])

    # Add a range of chart lines at once
    chart_lines = np.append(np.arange(-4, 5, 1), np.arange(-100, 110, 10))
    df.ta.constants(True, chart_lines)

    # Remove specific constants
    df.ta.constants(False, [0, 100])

indicators
~~~~~~~~~~

``indicators(**kwargs)`` — prints or returns the list of available indicators.

.. code-block:: python

    # Print all available indicators to the log
    df.ta.indicators()

    # Return indicators as a Python list
    ind_list = df.ta.indicators(as_list=True)

    # Return the list excluding specific indicators
    ind_list = df.ta.indicators(as_list=True, exclude=["vp", "td_seq"])

ticker
~~~~~~

.. deprecated:: 0.6.53
   ``df.ta.ticker()`` (and ``ta.yf()`` / ``ta.av()``) is deprecated and will be
   removed in a future release; data fetching is out of scope for a
   technical-analysis library. It emits a ``FutureWarning``. Fetch OHLCV with
   yfinance / alpha-vantage directly and pass the DataFrame to pandas-ta-classic
   — see ``examples/fetch_market_data.py`` for the replacement patterns.

.. code-block:: python

    # Deprecated — emits FutureWarning (requires the [data] extra)
    df = df.ta.ticker("AAPL")

    # Preferred: fetch with yfinance directly, then use pandas-ta-classic
    import yfinance as yf
    df = yf.download("AAPL", period="1y", interval="1d")
    df.ta.rsi(append=True)

chain
~~~~~

``chain(append: bool = True)`` — activates fluent chaining mode, where every
indicator call auto-appends to the DataFrame and returns the DataFrame (so
``.ta`` is available for the next call).  Added in v0.6.

.. code-block:: python

    # Chain four indicators in a single expression
    df.ta.chain().sma(20).ta.rsi(14).ta.macd().ta.bbands(20)

    # With per-indicator kwargs
    df.ta.chain().sma(20, prefix="FAST").ta.sma(50, prefix="SLOW")

    # Rename output columns inline
    df.ta.chain().bbands(
        20, col_names=("LOWER", "MID", "UPPER", "BW", "PCT")
    )

.. note::
   Chain state is stored on ``df.attrs`` and persists across ``.ta`` accessor
   re-entries.  Use :meth:`unchain` to exit chain mode.

unchain
~~~~~~~

``unchain()`` — deactivates fluent chaining mode and returns the DataFrame
for normal, non-chained usage.

.. code-block:: python

    # Start chaining, then exit mid-expression
    df.ta.chain().sma(20).ta.unchain().ta.rsi(14)
    #                               ^
    #                               rsi() returns a Series (not appended)

    # Manual exit
    df.ta.chain()
    # ... do some chained calls ...
    df.ta.unchain()  # chain mode deactivated
