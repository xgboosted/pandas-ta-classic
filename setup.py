# -*- coding: utf-8 -*-
from distutils.core import setup

# Latest stable Python version - update this when new Python versions are released
LATEST_PYTHON_VERSION = "3.13"

def generate_python_classifiers():
    """Generate Python version classifiers based on latest version constant."""
    # Extract minor version from LATEST_PYTHON_VERSION (e.g., "3.13" -> 13)
    minor_version = int(LATEST_PYTHON_VERSION.split('.')[1])
    
    # Generate classifiers for latest and 4 previous minor versions
    classifiers = []
    for i in range(4, -1, -1):  # 4, 3, 2, 1, 0
        version_num = minor_version - i
        classifiers.append(f"Programming Language :: Python :: 3.{version_num}")
    
    return classifiers

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
    version=".".join(("0", "3", "14b1")),
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
    ] + generate_python_classifiers() + [
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
        "numpy>=2.0.0",
        "pandas>=2.0.0"
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
