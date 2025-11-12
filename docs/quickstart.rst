Quickstart Guide
================

Welcome to **Pandas TA Classic**! This guide will help you get started quickly with calculating technical indicators for your financial data analysis.

Installation
------------

Quick Install
~~~~~~~~~~~~~

The fastest way to get started:

.. code-block:: bash

   pip install pandas-ta-classic

Or using ``uv`` (faster):

.. code-block:: bash

   uv pip install pandas-ta-classic

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full functionality including TA-Lib candlestick patterns:

.. code-block:: bash

   pip install pandas-ta-classic TA-Lib

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas_ta_classic as ta
   print(ta.version)

Your First Indicators
---------------------

Method 1: Standard Approach (Explicit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perfect for when you want full control:

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   # Create sample data
   df = pd.DataFrame({
       'open': [100, 101, 102, 103, 104],
       'high': [105, 106, 107, 108, 109],
       'low': [99, 100, 101, 102, 103],
       'close': [104, 105, 106, 107, 108],
       'volume': [1000, 1100, 1200, 1300, 1400]
   })

   # Calculate a Simple Moving Average
   sma_20 = ta.sma(df['close'], length=20)
   
   # Calculate RSI (Relative Strength Index)
   rsi_14 = ta.rsi(df['close'], length=14)
   
   # Calculate MACD
   macd = ta.macd(df['close'])

Method 2: DataFrame Extension (Convenient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way for most users:

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   # Load your data
   df = pd.read_csv('your_data.csv')

   # Calculate and append indicators directly
   df.ta.sma(length=20, append=True)         # Adds SMA_20 column
   df.ta.rsi(length=14, append=True)         # Adds RSI_14 column
   df.ta.macd(append=True)                   # Adds MACD columns
   df.ta.bbands(length=20, append=True)      # Adds Bollinger Bands

   # View your DataFrame with new indicators
   print(df.tail())

Method 3: Strategy System (Powerful)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run multiple indicators at once with multiprocessing:

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   df = pd.read_csv('your_data.csv')

   # Use a built-in strategy
   df.ta.strategy("CommonStrategy")

   # Or create your own custom strategy
   my_strategy = ta.Strategy(
       name="MyStrategy",
       ta=[
           {"kind": "sma", "length": 20},
           {"kind": "sma", "length": 50},
           {"kind": "rsi", "length": 14},
           {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
           {"kind": "bbands", "length": 20},
       ]
   )

   # Run your strategy
   df.ta.strategy(my_strategy)

Working with Real Data
-----------------------

Using yfinance
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta
   import yfinance as yf

   # Download stock data
   df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

   # Calculate indicators
   df.ta.sma(length=50, append=True)
   df.ta.rsi(length=14, append=True)
   df.ta.macd(append=True)

From CSV Files
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   # Load CSV with OHLCV data
   df = pd.read_csv('stock_data.csv', parse_dates=['date'], index_col='date')

   # Ensure column names are lowercase
   df.columns = [col.lower() for col in df.columns]

   # Calculate indicators
   df.ta.sma(length=20, append=True)
   df.ta.atr(length=14, append=True)

Common Use Cases
----------------

Trend Following
~~~~~~~~~~~~~~~

Identify the trend direction:

.. code-block:: python

   # Moving averages for trend
   df.ta.sma(length=20, append=True)   # Short-term
   df.ta.sma(length=50, append=True)   # Medium-term  
   df.ta.sma(length=200, append=True)  # Long-term

   # Trend indicator
   df.ta.adx(length=14, append=True)   # Average Directional Index

Momentum Trading
~~~~~~~~~~~~~~~~

Identify overbought/oversold conditions:

.. code-block:: python

   # RSI for momentum
   df.ta.rsi(length=14, append=True)

   # Stochastic Oscillator
   df.ta.stoch(append=True)

   # Williams %R
   df.ta.willr(length=14, append=True)

Volatility Analysis
~~~~~~~~~~~~~~~~~~~

Measure market volatility:

.. code-block:: python

   # Bollinger Bands
   df.ta.bbands(length=20, std=2, append=True)

   # Average True Range
   df.ta.atr(length=14, append=True)

   # Keltner Channels
   df.ta.kc(length=20, append=True)

Volume Analysis
~~~~~~~~~~~~~~~

Analyze trading volume:

.. code-block:: python

   # On-Balance Volume
   df.ta.obv(append=True)

   # Volume Weighted Average Price
   df.ta.vwap(append=True)

   # Accumulation/Distribution
   df.ta.ad(append=True)

Exploring Available Indicators
-------------------------------

List All Indicators
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas_ta_classic as ta

   # See all indicator categories
   print(ta.Category)

   # List indicators by category
   print(ta.momentum)    # All momentum indicators
   print(ta.overlap)     # All overlap indicators
   print(ta.trend)       # All trend indicators

Get Help on Indicators
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get indicator documentation
   help(ta.sma)
   help(ta.macd)
   help(ta.rsi)

Troubleshooting
---------------

KeyError: 'close'
~~~~~~~~~~~~~~~~~

Ensure your DataFrame has the required columns:

.. code-block:: python

   # Check column names
   print(df.columns)

   # Rename columns if needed
   df.columns = ['open', 'high', 'low', 'close', 'volume']

   # Or specify columns explicitly
   df.ta.sma(close=df['price'], length=20)

All NaN values
~~~~~~~~~~~~~~

Check if you have enough data points:

.. code-block:: python

   # Most indicators need minimum data
   print(len(df))  # Should be > indicator length

   # Example: SMA(20) needs at least 20 data points
   if len(df) >= 20:
       df.ta.sma(length=20, append=True)

Import Error
~~~~~~~~~~~~

Verify installation:

.. code-block:: bash

   pip list | grep pandas-ta-classic
   # or
   pip install --upgrade pandas-ta-classic

Next Steps
----------

Now that you've got the basics, explore more:

1. **Tutorials** - Step-by-step guides in ``TUTORIALS.md``
2. **Full Documentation** - Complete API reference at https://xgboosted.github.io/pandas-ta-classic/
3. **Indicator Reference** - All 203 indicators
4. **Strategy Guide** - Advanced strategy system
5. **Examples** - Jupyter notebooks with real examples

Quick Reference
---------------

=============================================== =============================================
Task                                            Code
=============================================== =============================================
Install                                         ``pip install pandas-ta-classic``
Import                                          ``import pandas_ta_classic as ta``
Simple indicator                                ``df.ta.sma(length=20, append=True)``
Multiple indicators                             ``df.ta.strategy("CommonStrategy")``
Custom strategy                                 ``ta.Strategy(name="My", ta=[...])``
List categories                                 ``print(ta.Category)``
Get help                                        ``help(ta.sma)``
Fetch data                                      ``df.ta.ticker("AAPL")``
=============================================== =============================================

Need Help?
----------

- **Issues**: https://github.com/xgboosted/pandas-ta-classic/issues
- **Discussions**: https://github.com/xgboosted/pandas-ta-classic/discussions
- **Examples**: Check the ``examples/`` directory

Happy Trading! 📈
