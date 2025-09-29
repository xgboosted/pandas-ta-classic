DataFrame Properties and Methods
================================

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

.. code-block:: python

    # Prefix all Technical Analysis column names
    df.ta.prefix = "TA"

    # Suffix all Technical Analysis column names  
    df.ta.suffix = "XYZ"

    # Use both prefix and suffix
    df.ta.prefix, df.ta.suffix = "TA", "XYZ"

    # Reset
    df.ta.prefix = df.ta.suffix = None

time_range
~~~~~~~~~~

.. code-block:: python

    # Set the time range for indicators (if datetime indexed)
    df.ta.time_range = "1y"  # Last year
    df.ta.time_range = "6m"  # Last 6 months
    df.ta.time_range = None  # Reset to full range

to_utc
~~~~~~

.. code-block:: python

    # Convert DataFrame index to UTC
    df.ta.to_utc(inplace=True)

Methods
-------

constants
~~~~~~~~~

.. code-block:: python

    # Add constant values as new columns
    df.ta.constants(pi=3.14159, e=2.71828)

indicators
~~~~~~~~~~

.. code-block:: python

    # List all available indicators
    df.ta.indicators()

    # List indicators by category
    df.ta.indicators("momentum")

ticker
~~~~~~

.. code-block:: python

    # Download stock data (requires yfinance)
    df = df.ta.ticker("AAPL")
    
    # With period and interval
    df = df.ta.ticker("AAPL", period="1y", interval="1d")