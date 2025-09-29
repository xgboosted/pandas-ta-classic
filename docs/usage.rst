Usage Guide
===========

Programming Conventions
------------------------

**Pandas TA Classic** has three primary "styles" of processing Technical Indicators for your use case and/or requirements. They are: *Standard*, *DataFrame Extension*, and the *Pandas TA Classic Strategy*. Each with increasing levels of abstraction for ease of use.

Standard Usage
~~~~~~~~~~~~~~

You explicitly define the input columns and take care of the output.

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

Strategy System
~~~~~~~~~~~~~~~

A **Pandas TA Classic** Strategy is a named group of indicators to be run by the *strategy* method. All Strategies use **multiprocessing** except when using the ``col_names`` parameter.

.. code-block:: python

    # Create a custom strategy
    MyStrategy = ta.Strategy(
        name="DCSMA10",
        ta=[
            {"kind": "ohlc4"},
            {"kind": "sma", "length": 10},
            {"kind": "donchian", "lower_length": 10, "upper_length": 15},
            {"kind": "ema", "close": "OHLC4", "length": 10, "suffix": "OHLC4"},
        ]
    )
    
    # Run the Strategy
    df.ta.strategy(MyStrategy)

Quick Start
-----------

.. code-block:: python

    import pandas as pd
    import pandas_ta_classic as ta

    # Load your data
    df = pd.read_csv("path/to/symbol.csv", sep=",")
    # OR if you have yfinance installed
    df = df.ta.ticker("aapl")

    # VWAP requires the DataFrame index to be a DatetimeIndex
    df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

    # Calculate Returns and append to the df DataFrame
    df.ta.log_return(cumulative=True, append=True)
    df.ta.percent_return(cumulative=True, append=True)

    # Check new columns
    df.columns

    # Take a look
    df.tail()