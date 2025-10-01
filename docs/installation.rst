Installation
============

Requirements
------------

**Pandas TA Classic** requires:

- Python (the latest stable version plus the prior 4 minor versions)
- pandas
- numpy

.. note::
   Python version support follows a **rolling policy**: the latest stable Python version plus 4 preceding minor versions. When new Python versions are released, support is automatically updated via CI/CD workflows. Check the `CI workflow <https://github.com/xgboosted/pandas-ta-classic/blob/main/.github/workflows/ci.yml>`_ for the current supported versions.

Optional Dependencies
---------------------

For enhanced functionality, consider installing:

- **TA-Lib**: Enables all 60+ candlestick patterns
- **yfinance**: For downloading stock data with ``df.ta.ticker()``
- **vectorbt**: For backtesting integration

Installation Methods
--------------------

**Pandas TA Classic** supports both modern **uv** and traditional **pip** package managers.

Stable Release (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest stable release from PyPI:

Using ``uv`` (recommended - faster):

.. code-block:: bash

    uv pip install pandas-ta-classic

Using ``pip``:

.. code-block:: bash

    pip install pandas-ta-classic

Latest Development Version
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the most recent version with all latest features and bug fixes:

Using ``uv``:

.. code-block:: bash

    uv pip install git+https://github.com/xgboosted/pandas-ta-classic

Using ``pip``:

.. code-block:: bash

    pip install -U git+https://github.com/xgboosted/pandas-ta-classic

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributing to the project or testing unreleased features:

Using ``uv``:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/xgboosted/pandas-ta-classic.git
    cd pandas-ta-classic
    
    # Install in editable mode with all dependencies
    uv pip install -e ".[all]"
    
    # Or install specific dependency groups:
    uv pip install -e ".[dev]"      # Development tools
    uv pip install -e ".[optional]" # Optional features like TA-Lib

Using ``pip``:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/xgboosted/pandas-ta-classic.git
    cd pandas-ta-classic
    
    # Install in editable mode with all dependencies
    pip install -e ".[all]"
    
    # Or install specific dependency groups:
    pip install -e ".[dev]"      # Development tools
    pip install -e ".[optional]" # Optional features like TA-Lib

.. note::
   **Development Installation Requirements**:
   
   - Full git repository with history and tags (not a shallow clone)
   - setuptools-scm is automatically installed as a build dependency
   - Git tags determine the package version (e.g., ``0.3.36.dev1`` for development, ``0.3.35`` for releases)
   - See the `Version Management section in CONTRIBUTING.md <https://github.com/xgboosted/pandas-ta-classic/blob/main/CONTRIBUTING.md#version-management>`_ for details

Installing TA-Lib
------------------

To use all candlestick patterns, install TA-Lib:

**Windows/macOS/Linux**:

Using ``uv``:

.. code-block:: bash

    uv pip install TA-Lib

Using ``pip``:

.. code-block:: bash

    pip install TA-Lib

**Note**: If you encounter installation issues with TA-Lib, refer to the `TA-Lib installation guide <https://github.com/mrjbq7/ta-lib#installation>`_.

Installing Optional Dependencies
--------------------------------

For complete functionality:

Using ``uv``:

.. code-block:: bash

    # For stock data download
    uv pip install yfinance
    
    # For backtesting
    uv pip install vectorbt
    
    # For enhanced performance (if available for your system)
    uv pip install numba
    
    # Install all optional dependencies at once
    uv pip install pandas-ta-classic[optional]

Using ``pip``:

.. code-block:: bash

    # For stock data download
    pip install yfinance
    
    # For backtesting
    pip install vectorbt
    
    # For enhanced performance (if available for your system)
    pip install numba
    
    # Install all optional dependencies at once
    pip install pandas-ta-classic[optional]

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