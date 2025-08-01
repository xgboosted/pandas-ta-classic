# -*- coding: utf-8 -*-
from distutils.core import setup

long_description = "An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib. This is the classic/community maintained version."

setup(
    name="pandas-ta-classic",
    packages=[
        "pandas_ta_classic",
        "pandas_ta_classic.candles",
        "pandas_ta_classic.cycles",
        "pandas_ta_classic.momentum",
        "pandas_ta_classic.overlap",
        "pandas_ta_classic.performance",
        "pandas_ta_classic.statistics",
        "pandas_ta_classic.trend",
        "pandas_ta_classic.utils",
        "pandas_ta_classic.utils.data",
        "pandas_ta_classic.volatility",
        "pandas_ta_classic.volume"
    ],
    version=".".join(("0", "3", "14b")),
    description=long_description,
    long_description=long_description,
    author="xgboosted",
    author_email="",
    url="https://github.com/xgboosted/pandas-ta-classic",
    maintainer="xgboosted",
    maintainer_email="",
    download_url="https://github.com/xgboosted/pandas-ta-classic.git",
    keywords=["technical analysis", "trading", "python3", "pandas"],
    license="The MIT License (MIT)",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_data={
        "data": ["data/*.csv"],
    },
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ],
    # List additional groups of dependencies here (e.g. development dependencies).
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": [
            "alphaVantage-api", "matplotlib", "mplfinance", "scipy",
            "sklearn", "statsmodels", "stochastic",
            "talib", "tqdm", "vectorbt", "yfinance",
        ],
        "test": ["ta-lib"],
    },
)
