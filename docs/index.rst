Pandas TA Classic Documentation
===============================

**Pandas TA Classic** is an easy to use library that leverages the Pandas package with **141 indicators and utility functions** plus **62 TA-Lib candlestick patterns** (203 total). This is the **community maintained version** of the popular pandas-ta library.

.. note::
   The library features **dynamic configuration management**:
   
   - **Category Discovery**: Indicators are automatically detected from the package structure
   - **Version Management**: Python version support (the latest stable plus the prior 4 versions) is dynamically managed via CI/CD workflows
   - Both ensure the library metadata stays in sync with actual capabilities

.. image:: https://img.shields.io/github/license/xgboosted/pandas-ta-classic
   :target: #license
   :alt: License

.. image:: https://github.com/xgboosted/pandas-ta-classic/workflows/CI/badge.svg
   :target: https://github.com/xgboosted/pandas-ta-classic/actions
   :alt: Build Status

.. image:: https://img.shields.io/pypi/v/pandas-ta-classic?style=flat
   :target: https://pypi.org/project/pandas-ta-classic/
   :alt: PyPI Version

Quick Start
-----------

**Pandas TA Classic** supports both modern ``uv`` and traditional ``pip`` package managers.

Using ``uv`` (recommended - faster):

.. code-block:: bash

   uv pip install pandas-ta-classic

Using ``pip``:

.. code-block:: bash

   pip install pandas-ta-classic

Basic usage:

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   # Load your data
   df = pd.read_csv("path/to/symbol.csv")
   
   # Calculate indicators
   df.ta.sma(length=20, append=True)
   df.ta.rsi(append=True)
   df.ta.macd(append=True)

.. note::
   **New to Pandas TA Classic?** Check out our :doc:`quickstart` guide for a comprehensive introduction, or explore the :doc:`usage` guide for detailed programming conventions.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart
   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   strategies
   dataframe_api
   indicators
   performance

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources:

   GitHub Repository <https://github.com/xgboosted/pandas-ta-classic>
   Examples <https://github.com/xgboosted/pandas-ta-classic/tree/main/examples>
   Changelog <https://github.com/xgboosted/pandas-ta-classic/blob/main/CHANGELOG.md>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
