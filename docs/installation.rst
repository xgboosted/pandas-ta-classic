Installation
============

Requirements
------------

**Pandas TA Classic** requires:

- Python 3.9 or higher
- pandas
- numpy

Optional Dependencies
---------------------

For enhanced functionality, consider installing:

- **TA-Lib**: Enables all 60+ candlestick patterns
- **yfinance**: For downloading stock data with ``df.ta.ticker()``
- **vectorbt**: For backtesting integration

Installation Methods
--------------------

Stable Release (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release from PyPI:

.. code-block:: bash

    pip install pandas-ta-classic

Latest Development Version
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the most recent version with all latest features and bug fixes:

.. code-block:: bash

    pip install -U git+https://github.com/xgboosted/pandas-ta-classic

Cutting Edge (Development Branch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Warning**: This version may contain bugs and breaking changes. Use at your own risk!

.. code-block:: bash

    pip install -U git+https://github.com/xgboosted/pandas-ta-classic.git@development

Installing TA-Lib
------------------

To use all candlestick patterns, install TA-Lib:

**Windows/macOS/Linux**:

.. code-block:: bash

    pip install TA-Lib

**Note**: If you encounter installation issues with TA-Lib, refer to the `TA-Lib installation guide <https://github.com/mrjbq7/ta-lib#installation>`_.

Installing Optional Dependencies
--------------------------------

For complete functionality:

.. code-block:: bash

    # For stock data download
    pip install yfinance
    
    # For backtesting
    pip install vectorbt
    
    # For enhanced performance (if available for your system)
    pip install numba

Verification
------------

Verify your installation:

.. code-block:: python

    import pandas_ta_classic as ta
    import pandas as pd
    
    # Create a simple DataFrame
    df = pd.DataFrame({'close': [100, 101, 102, 101, 100]})
    
    # Test an indicator
    sma = df.ta.sma(length=3)
    print(sma)
    
    # List all available indicators
    print(f"Available indicators: {len(df.ta.indicators())}")

If this runs without errors, you're ready to use Pandas TA Classic!