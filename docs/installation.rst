Installation
============

Requirements
------------

**Pandas TA Classic** requires:

- Python (the latest stable version plus the prior 4 minor versions)
- pandas >= 3.1.0
- numpy >= 2.4.6 (the newest numpy that still supports Python 3.11)

.. note::
   Python version support follows a **rolling policy**: the latest stable Python version plus 4 preceding minor versions, currently **Python 3.11–3.15**. The window is set by ``LATEST_PYTHON_VERSION`` in the `CI workflow <https://github.com/xgboosted/pandas-ta-classic/blob/main/.github/workflows/ci.yml>`_, which a maintainer raises after a new Python release.

Optional Dependencies
---------------------

For enhanced functionality, consider installing:

- **TA-Lib**: Optional — all 62 CDL patterns work natively without it (see :ref:`Installing TA-Lib` below)
- **tulipy**: Optional — only needed to *regenerate* the frozen oracle golden file (``tests/fixtures/generate_tulipy_oracle.py`` on CPython <3.12); ``test_oracle_tulipy.py`` compares against the committed snapshot and no longer imports tulipy at test time
- **yfinance / alpha-vantage**: pandas-ta-classic does not download data. Install a data provider yourself (``pip install yfinance``) and pass its OHLCV DataFrame in (see ``examples/fetch_market_data.py``)
- **vectorbt / backtrader / backtesting**: For backtesting integration (``pip install pandas-ta-classic[backtest]``)
- **pytest + Hypothesis**: For running the test suite and property-based tests (``pip install pandas-ta-classic[test]``)

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

.. note::
   **Upgrading from 0.8.x:** 0.9.0 is a breaking release. Invalid arguments and
   non-Series input now raise instead of falling back to a default, short input
   returns an all-NaN result instead of ``None``, and some results and column
   names change. Read `Upgrading from 0.8.x <https://github.com/xgboosted/pandas-ta-classic/blob/main/CHANGELOG.md#upgrading-from-08x>`_ before you upgrade.

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
    # (excludes backtest — install it explicitly, see below)
    uv pip install -e ".[all]"

    # Or install specific dependency groups:
    uv pip install -e ".[dev]"      # Development tools
    uv pip install -e ".[test]"     # Testing: pytest, Hypothesis
    uv pip install -e ".[optional]" # Optional runtime features (tqdm progress bars)
    uv pip install -e ".[oracle]"   # Oracle parity lib: TA-Lib (tulipy values are frozen)
    uv pip install -e ".[backtest]" # Backtesting: backtesting, vectorbt, backtrader

Using ``pip``:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/xgboosted/pandas-ta-classic.git
    cd pandas-ta-classic
    
    # Install in editable mode with all dependencies
    # (excludes backtest — install it explicitly, see below)
    pip install -e ".[all]"

    # Or install specific dependency groups:
    pip install -e ".[dev]"      # Development tools
    pip install -e ".[test]"     # Testing: pytest, Hypothesis
    pip install -e ".[optional]" # Optional runtime features (tqdm progress bars)
    pip install -e ".[oracle]"   # Oracle parity lib: TA-Lib (tulipy values are frozen)
    pip install -e ".[backtest]" # Backtesting: backtesting, vectorbt, backtrader

.. note::
   **Development Installation Requirements**:
   
   - Full git repository with history and tags (not a shallow clone)
   - setuptools-scm is automatically installed as a build dependency
   - Git tags determine the package version (e.g., ``0.9.1.dev1`` for development, ``0.9.0`` for releases)
   - A GitHub source archive (``pip install https://github.com/xgboosted/pandas-ta-classic/archive/main.zip`` or *Download ZIP*) carries its version in ``.git_archival.txt``, which ``git archive`` fills in
   - A source copy with neither ``.git`` nor that file (for example a Docker ``COPY`` without ``.git``) fails to build with setuptools-scm's ``LookupError`` instead of installing as ``0.0.0``; include ``.git`` or set ``SETUPTOOLS_SCM_PRETEND_VERSION``
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
     - Core indicators — native by default, opt-in via ``talib=True``; also used by ``test_oracle_talib.py`` for parity checks
   * - tulipy
     - **Oracle only**
     - Never used as computation backend; only ``test_oracle_tulipy.py`` uses it to verify native output

.. _installing ta-lib:

Installing TA-Lib
~~~~~~~~~~~~~~~~~

TA-Lib has a **dual role**: acceleration backend for core indicators (opt-in via ``talib=True``), and parity oracle. The two behavioural areas affected are:

**Candlestick patterns (CDL family)**
   All 62 CDL patterns have native Python implementations that are always used. TA-Lib is **never** invoked for CDL patterns — the TA-Lib fallback code path in ``cdl_pattern()`` is only retained for hypothetical future patterns without a native implementation.

**Core indicators** (``ema``, ``sma``, ``rsi``, ``macd``, ``obv``, ``atr``, etc.)
   The native implementation is used by default. TA-Lib is opt-in — pass ``talib=True`` to any call to use TA-Lib's implementation instead.

   .. code-block:: python

        import pandas_ta_classic as ta

        # Uses native EMA — default behaviour
        ema = df.ta.ema(length=20)

        # Use TA-Lib implementation if installed
        ema = df.ta.ema(length=20, talib=True)

        # CDL patterns — always native, talib= kwarg has no effect here
        engulfing = df.ta.cdl_pattern(name="engulfing")  # native
        result = df.ta.cdl_pattern(name="hammer")      # native

Installation
^^^^^^^^^^^^

Binary wheels (v0.6.5+) bundle the TA-Lib C library — no separate C library
setup needed. Versions before 0.6.5 lack binary wheels and require manual C library
installation. We require ``>=0.8.3``: its binary wheels work out of the box on
Linux, macOS, and Windows (x86\_64, arm64), and from 0.8.3 on TA-Lib's ``RSI``
reads the neutral 50 on a series that has not moved since its first bar, as
native ``rsi`` does (older releases read ``0.0`` there):

Using ``uv``:

.. code-block:: bash

    uv pip install "ta-lib>=0.8.3"

Using ``pip``:

.. code-block:: bash

    pip install "ta-lib>=0.8.3"

If your platform is not covered by binary wheels, or you prefer a source build, install the `TA-Lib C library <https://ta-lib.org/install/>`_ first. For Conda users:

.. code-block:: bash

    conda install -c conda-forge libta-lib
    conda install -c conda-forge ta-lib

See the `TA-Lib Python README <https://github.com/TA-Lib/ta-lib-python>`_ for platform-specific instructions and troubleshooting.

Installing tulipy
~~~~~~~~~~~~~~~~~

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

``test_oracle_talib.py`` is guarded with ``@unittest.skipUnless`` and skips automatically when TA-Lib is not installed. ``test_oracle_tulipy.py`` compares against tulipy values frozen in ``tests/fixtures/tulipy_oracle.json``, so it runs without tulipy installed.

Installing Optional Dependencies
--------------------------------

For complete functionality:

Using ``uv``:

.. code-block:: bash

    # For stock data download
    uv pip install yfinance
    
    # For backtesting
    uv pip install backtesting
    uv pip install backtrader
    uv pip install vectorbt

    # For enhanced performance (optional — 6–200× faster recursive indicators on 100,000-bar series)
    uv pip install pandas-ta-classic[performance]

    # Or install numba directly
    uv pip install numba

    # Progress bar for strategy() runs (tqdm)
    uv pip install pandas-ta-classic[optional]

    # The three backtesting libraries above in one go
    uv pip install pandas-ta-classic[backtest]

    # Every extra (optional, oracle, dev, test, docs, performance)
    uv pip install pandas-ta-classic[all]

Using ``pip``:

.. code-block:: bash

    # For stock data download
    pip install yfinance

    # For backtesting
    pip install backtesting
    pip install backtrader
    pip install vectorbt
    
    # For enhanced performance (optional — 6–200× faster recursive indicators on 100,000-bar series)
    pip install pandas-ta-classic[performance]
    
    # Or install numba directly
    pip install numba
    
    # Progress bar for strategy() runs (tqdm)
    pip install pandas-ta-classic[optional]

    # The three backtesting libraries above in one go
    pip install pandas-ta-classic[backtest]

    # Every extra (optional, oracle, dev, test, docs, performance)
    pip install pandas-ta-classic[all]

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
    print(f"Available indicators: {len(df.ta.indicators(as_list=True))}")

If this runs without errors, you're ready to use Pandas TA Classic!