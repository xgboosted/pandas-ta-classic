# -*- coding: utf-8 -*-
"""
Meta information for pandas-ta-classic
Contains Category definitions, version information, and import checks.
"""
from importlib.util import find_spec
from pathlib import Path

# Version information
try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("pandas-ta-classic")
    except PackageNotFoundError:
        __version__ = "UPDATEME!"  # Placeholder - replaced by CI/CD during release
except ImportError:
    # Fallback for when importlib.metadata is not available (Python < 3.8)
    try:
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            _dist = get_distribution("pandas-ta-classic")
            __version__ = _dist.version
        except DistributionNotFound:
            __version__ = "UPDATEME!"  # Placeholder - replaced by CI/CD during release
    except ImportError:
        __version__ = "UPDATEME!"  # Placeholder - replaced by CI/CD during release

version = __version__

# Import availability checks
# These correspond to the optional dependencies defined in pyproject.toml
Imports = {
    "alphaVantage-api": find_spec("alphaVantageAPI") is not None,
    "backtrader": find_spec("backtrader") is not None,
    "cython": find_spec("cython") is not None,
    "matplotlib": find_spec("matplotlib") is not None,
    "mplfinance": find_spec("mplfinance") is not None,
    "numba": find_spec("numba") is not None,
    "scipy": find_spec("scipy") is not None,
    "sklearn": find_spec("sklearn") is not None,
    "statsmodels": find_spec("statsmodels") is not None,
    "stochastic": find_spec("stochastic") is not None,
    "talib": find_spec("talib") is not None,
    "tqdm": find_spec("tqdm") is not None,
    "vectorbt": find_spec("vectorbt") is not None,
    "yaml": find_spec("yaml") is not None,
    "yfinance": find_spec("yfinance") is not None,
}


def _build_category_dict():
    """
    Dynamically build the Category dictionary by scanning the package directory structure.
    
    This function automatically discovers all indicator modules by:
    1. Finding all subdirectories in pandas_ta_classic (except special ones like __pycache__)
    2. For each subdirectory, listing all .py files (except __init__.py)
    3. Building a dictionary mapping category names to lists of indicator names
    
    Returns:
        dict: Category dictionary mapping category names to lists of indicator function names
    """
    categories = {}
    
    # Get the directory containing this file (pandas_ta_classic/)
    package_dir = Path(__file__).parent
    
    # Define categories that should be included (subdirectories with indicators)
    # This excludes utility directories that don't contain indicators
    valid_categories = {
        "candles", "cycles", "momentum", "overlap", "performance",
        "statistics", "trend", "volatility", "volume"
    }
    
    # Scan each subdirectory
    for category_path in package_dir.iterdir():
        # Skip if not a directory or not a valid category
        if not category_path.is_dir():
            continue
        
        category_name = category_path.name
        
        # Skip special directories and non-indicator directories
        if category_name.startswith("_") or category_name.startswith("."):
            continue
        if category_name == "__pycache__":
            continue
        if category_name not in valid_categories:
            continue
        
        # Find all .py files in this category (excluding __init__.py)
        indicators = []
        for file_path in category_path.glob("*.py"):
            if file_path.name != "__init__.py":
                # Remove .py extension to get the indicator name
                indicators.append(file_path.stem)
        
        # Sort indicators alphabetically for consistency
        if indicators:
            categories[category_name] = sorted(indicators)
    
    return categories


# Dynamically build the Category dictionary
# This replaces the previous hardcoded dictionary and automatically
# stays in sync with the filesystem structure
Category = _build_category_dict()

CANGLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# https://www.worldtimezone.com/markets24.php
EXCHANGE_TZ = {
    "NZSX": 12,
    "ASX": 11,
    "TSE": 9,
    "HKE": 8,
    "SSE": 8,
    "SGX": 8,
    "NSE": 5.5,
    "DIFX": 4,
    "RTS": 3,
    "JSE": 2,
    "FWB": 1,
    "LSE": 1,
    "BMF": -2,
    "NYSE": -4,
    "TSX": -4,
}

RATE = {
    "DAYS_PER_MONTH": 21,
    "MINUTES_PER_HOUR": 60,
    "MONTHS_PER_YEAR": 12,
    "QUARTERS_PER_YEAR": 4,
    "TRADING_DAYS_PER_YEAR": 252,  # Keep even
    "TRADING_HOURS_PER_DAY": 6.5,
    "WEEKS_PER_YEAR": 52,
    "YEARLY": 1,
}
