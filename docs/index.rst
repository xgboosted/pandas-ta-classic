Pandas TA Classic Documentation
===============================

**Pandas TA Classic** is an easy to use library that leverages the Pandas package with more than 130 Indicators and Utility functions and more than 60 TA Lib Candlestick Patterns. This is the **classic/community maintained version** of the popular pandas-ta library.

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

.. code-block:: bash

   pip install pandas-ta-classic

.. code-block:: python

   import pandas as pd
   import pandas_ta_classic as ta

   # Load your data
   df = pd.read_csv("path/to/symbol.csv")
   
   # Calculate indicators
   df.ta.sma(length=20, append=True)
   df.ta.rsi(append=True)
   df.ta.macd(append=True)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

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
