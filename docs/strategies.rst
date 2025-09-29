Strategy System
===============

Overview
--------

A **Pandas TA Classic** Strategy is a named group of indicators to be run by the *strategy* method. The Strategy Class is a simple way to name and group your favorite TA Indicators using a Data Class.

**Pandas TA** comes with two prebuilt basic Strategies:

* **AllStrategy** - Runs all available indicators
* **CommonStrategy** - Runs commonly used indicators

Strategy Requirements
---------------------

* **name**: Some short memorable string. *Note*: Case-insensitive "All" is reserved.
* **ta**: A list of dicts containing keyword arguments to identify the indicator and the indicator's arguments
* **Note**: A Strategy will fail when consumed by Pandas TA if there is no ``{"kind": "indicator name"}`` attribute.

Optional Parameters
-------------------

* **description**: A more detailed description of what the Strategy tries to capture. Default: None
* **created**: A datetime string of when it was created. Default: Automatically generated.

Types of Strategies
-------------------

Builtin Strategies
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Running the Builtin CommonStrategy
    df.ta.strategy(ta.CommonStrategy)

    # The Default Strategy is the ta.AllStrategy. The following are equivalent:
    df.ta.strategy()
    df.ta.strategy("All")
    df.ta.strategy(ta.AllStrategy)

Categorical Strategies
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # List of indicator categories
    df.ta.categories

    # Running a Categorical Strategy only requires the Category name
    df.ta.strategy("Momentum")  # Default values for all Momentum indicators
    df.ta.strategy("overlap", length=42)  # Override all Overlap 'length' attributes

Custom Strategies
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create your own Custom Strategy
    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            {"kind": "sma", "length": 50},
            {"kind": "sma", "length": 200},
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "macd", "fast": 8, "slow": 21},
            {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        ]
    )
    # To run your "Custom Strategy"
    df.ta.strategy(CustomStrategy)

Multiprocessing
---------------

The **Pandas TA Classic** *strategy* method utilizes **multiprocessing** for bulk indicator processing of all Strategy types with **ONE EXCEPTION!** When using the ``col_names`` parameter to rename resultant column(s), the indicators in ``ta`` array will be ran in order.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    # VWAP requires the DataFrame index to be a DatetimeIndex.
    df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

    # Runs and appends all indicators to the current DataFrame by default
    df.ta.strategy()

    # Use verbose if you want to make sure it is running
    df.ta.strategy(verbose=True)

    # Use timed if you want to see how long it takes to run
    df.ta.strategy(timed=True)

    # Choose the number of cores to use. Default is all available cores.
    df.ta.cores = 4

    # For no multiprocessing, set this value to 0.
    df.ta.cores = 0

Excluding Indicators
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Maybe you do not want certain indicators
    df.ta.strategy(exclude=["bop", "mom", "percent_return", "wcp", "pvi"], verbose=True)

    # Perhaps you want to use different values for indicators
    df.ta.strategy(fast=10, slow=50, verbose=True)

Custom Strategy without Multiprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Remember**: These will not be utilizing **multiprocessing**

.. code-block:: python

    NonMPStrategy = ta.Strategy(
        name="EMAs, BBs, and MACD",
        description="Non Multiprocessing Strategy by rename Columns",
        ta=[
            {"kind": "ema", "length": 8},
            {"kind": "ema", "length": 21},
            {"kind": "bbands", "length": 20, "col_names": ("BBL", "BBM", "BBU")},
            {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("MACD", "MACD_H", "MACD_S")}
        ]
    )
    # Run it
    df.ta.strategy(NonMPStrategy)