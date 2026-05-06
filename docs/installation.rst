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

- **TA-Lib**: Optional — all 62 CDL patterns work natively without it (see :ref:`Installing TA-Lib` below)
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

Oracle Libraries (TA-Lib and tulipy)
-------------------------------------

Both TA-Lib and tulipy are **fully optional**. They serve different roles:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Library
     - Role
     - Effect when installed
   * - TA-Lib
     - **Acceleration backend + oracle**
     - 34 core indicators use TA-Lib's C implementation by default; also used by ``test_oracle_talib.py`` for parity checks
   * - tulipy
     - **Oracle only**
     - Never used as computation backend; only ``test_oracle_tulipy.py`` uses it to verify native output

Installing TA-Lib
^^^^^^^^^^^^^^^^^^

TA-Lib has a **dual role**: acceleration backend for 34 core indicators, and parity oracle. The two behavioural areas affected are:

**Candlestick patterns (CDL family)**
   All 62 CDL patterns have native Python implementations that are always used. TA-Lib is **never** invoked for CDL patterns — the TA-Lib fallback code path in ``cdl_pattern()`` is only retained for hypothetical future patterns without a native implementation.

**34 core indicators** (``ema``, ``sma``, ``rsi``, ``macd``, ``obv``, ``atr``, etc.)
   When TA-Lib is installed, these indicators use TA-Lib's implementation by default for numerical consistency with TA-Lib-based workflows. Pass ``talib=False`` to any call to force the native implementation instead.

   .. code-block:: python

       import pandas_ta_classic as ta

       # Uses TA-Lib EMA (if TA-Lib installed) — default behaviour
       ema = df.ta.ema(length=20)

       # Force native implementation regardless of TA-Lib
       ema = df.ta.ema(length=20, talib=False)

       # CDL patterns — always native, talib= kwarg has no effect here
       df = df.ta.cdl_pattern(name="engulfing")       # native
       result = df.ta.cdl_pattern(name="hammer")      # native

Installing TA-Lib:

Using ``uv``:

.. code-block:: bash

    uv pip install TA-Lib

Using ``pip``:

.. code-block:: bash

    pip install TA-Lib

**Note**: If you encounter installation issues with TA-Lib, refer to the `TA-Lib installation guide <https://github.com/mrjbq7/ta-lib#installation>`_.

Installing tulipy
^^^^^^^^^^^^^^^^^^

tulipy is an **oracle-only** library. It is never used as a computation backend — its sole purpose is ``test_oracle_tulipy.py``, which verifies that native indicator output matches tulipy's reference values.

.. note::

   Installing tulipy has **no effect** on indicator behaviour or performance at runtime. It only enables the tulipy oracle test suite.

Using ``uv``:

.. code-block:: bash

    uv pip install tulipy

Using ``pip``:

.. code-block:: bash

    pip install tulipy

Or install both oracle libraries at once:

.. code-block:: bash

    # uv
    uv pip install pandas-ta-classic[oracle]

    # pip
    pip install pandas-ta-classic[oracle]

Both oracle test suites (``test_oracle_talib.py``, ``test_oracle_tulipy.py``) are guarded with ``@unittest.skipUnless`` and skip automatically when the respective library is not installed.

Installing Optional Dependencies
--------------------------------

For complete functionality:

Using ``uv``:

.. code-block:: bash

    # For stock data download
    uv pip install yfinance
    
    # For backtesting
    uv pip install vectorbt
    
    # For enhanced performance (optional — provides 6–230× speedups on hot-loop indicators)
    uv pip install pandas-ta-classic[performance]
    
    # Or install numba directly
    uv pip install numba
    
    # Install all optional dependencies at once
    uv pip install pandas-ta-classic[optional]

Using ``pip``:

.. code-block:: bash

    # For stock data download
    pip install yfinance
    
    # For backtesting
    pip install vectorbt
    
    # For enhanced performance (optional — provides 6–230× speedups on hot-loop indicators)
    pip install pandas-ta-classic[performance]
    
    # Or install numba directly
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