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

.. note::
   Keyword arguments passed to ``strategy()`` reach every indicator in the run.
   Since 0.9.0 an indicator raises ``ValueError`` for a value it cannot use
   (for example ``ebsw`` needs ``length > 38``) instead of silently using its
   default, so exclude such indicators: ``df.ta.strategy("All", length=10, exclude=["ebsw"])``.

.. note::
   ``strategy()`` accepts ``"all"``, a category name or a ``Strategy``; any
   other name raises ``ValueError``. For ``"all"`` and a category, indicators
   that need a column the DataFrame does not have (``obv`` and the other
   volume indicators on OHLC-only data) are skipped and listed with
   ``verbose=True``; call one directly and it raises ``KeyError``.

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

Running a Strategy
------------------

``strategy()`` resolves the indicators you asked for into a plan, then runs it.
Entries that read a column an earlier entry produces are held back until it
exists, so a chained Strategy gives the same columns however it is executed.

By default everything runs in the calling process. Parallel execution is opt-in;
see `Parallel execution`_ for when it is worth it.

A column that neither exists nor is produced by an earlier entry raises ``KeyError`` naming the column.

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

    # strategy() runs serially by default. Ask for worker processes either
    # per frame or per call; see "Parallel execution" below for when that pays.
    df.ta.cores = 4
    df.ta.strategy(cores=4)

    # Back to serial execution.
    df.ta.cores = 0

Excluding Indicators
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Maybe you do not want certain indicators (a list, tuple or set of
    # indicator names; an unknown name raises ValueError)
    df.ta.strategy(exclude=["bop", "mom", "percent_return", "wcp", "pvi"], verbose=True)

    # Perhaps you want to use different values for indicators
    df.ta.strategy(fast=10, slow=50, verbose=True)

.. _Parallel execution:

Parallel execution
------------------

``strategy()`` runs serially unless you ask for workers. Starting a process pool
costs 0.5 to 1.7 seconds, so on a single DataFrame it only pays from roughly
100,000 rows upwards. Below that, serial is faster.

.. code-block:: python

    # Per call, or per frame. strategy() opens the pool and closes it again.
    df.ta.strategy(cores=8)
    df.ta.cores = 8

Reusing your own pool
~~~~~~~~~~~~~~~~~~~~~

Most of the cost of ``cores=`` is paid on every call: each fresh worker imports
pandas and loads the numba cache, about 0.66 s per process. Pass an
:class:`~concurrent.futures.Executor` you keep open instead and that is paid
once.

Because worker processes are started with *spawn*, they re-import the calling
script. Without the ``if __name__ == "__main__":`` guard, a script that calls
``strategy()`` at module level starts fresh workers from every worker.

.. code-block:: python

    import os

    # Before numpy is imported, so the workers inherit it. Each interpreter
    # otherwise reserves roughly 750 MB for OpenBLAS thread buffers that the
    # indicators never use: measured per worker, 786 MB against 47 MB.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from concurrent.futures import ProcessPoolExecutor

    import pandas_ta_classic as ta

    if __name__ == "__main__":
        with ProcessPoolExecutor(8) as pool:
            for df in frames:
                df.ta.strategy("all", executor=pool)

.. note::

    An ``initializer=`` that sets ``OPENBLAS_NUM_THREADS`` does **not** work.
    *spawn* re-imports the calling script -- and with it numpy -- before the
    executor runs the initializer, so the buffers are already reserved.

Parallelising over symbols usually wins by more than parallelising one frame:
16 frames of 10,000 rows take 22.9 s one after another, and 6.45 s when the
caller spreads the frames over its own pool and each ``strategy()`` runs serially.

``strategy()`` inside a *daemonic* worker -- a ``multiprocessing.Pool`` one --
ignores ``cores`` and ``executor`` and runs serially, because such a worker
cannot start children of its own. A ``ProcessPoolExecutor`` worker is not
daemonic, so there ``strategy(cores=N)`` does open a nested pool and the process
count multiplies; pass ``cores=0`` in code that may run inside one.

Renaming columns with col_names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    RenamedStrategy = ta.Strategy(
        name="EMAs, BBs, and MACD",
        description="Strategy that renames its result columns",
        ta=[
            {"kind": "ema", "length": 8},
            {"kind": "ema", "length": 21},
            {"kind": "bbands", "length": 20, "col_names": ("BBL", "BBM", "BBU", "BBB", "BBP")},  # one name per column
            {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("MACD", "MACD_H", "MACD_S")}
        ]
    )
    # Run it. col_names works with workers too: results are named and appended
    # in the calling process, in the order the entries are listed.
    df.ta.strategy(RenamedStrategy)

Chained Custom Strategy
~~~~~~~~~~~~~~~~~~~~~~~

An entry can use an earlier entry's output as its input. It waits for its own stage, after the entry that produces the column, so the result is the same serially and on workers (see above).

.. code-block:: python

    ChainedStrategy = ta.Strategy(
        name="Cumulative Log Return EMA",
        ta=[
            {"kind": "log_return", "cumulative": True},                             # appends CUMLOGRET_1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"},  # reads it: EMA_5_CLR
        ]
    )
    df.ta.strategy(ChainedStrategy)
