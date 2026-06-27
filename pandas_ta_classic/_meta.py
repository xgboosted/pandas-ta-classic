"""
Meta information for pandas-ta-classic
Contains Category definitions, version information, and import checks.
"""

from importlib.util import find_spec
from pathlib import Path

# Version information - dynamically determined from git tags via setuptools_scm
try:
    # Try to import version from setuptools_scm generated file
    from pandas_ta_classic._version import version as __version__
except ImportError:
    # Fallback: try to get version from installed package metadata
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            __version__ = version("pandas-ta-classic")
        except PackageNotFoundError:
            __version__ = "0.0.0"  # Fallback if package not installed
    except ImportError:
        # Fallback for Python < 3.8
        try:
            from pkg_resources import get_distribution, DistributionNotFound

            try:
                _dist = get_distribution("pandas-ta-classic")
                __version__ = _dist.version
            except DistributionNotFound:
                __version__ = "0.0.0"  # Fallback if package not installed
        except ImportError:
            __version__ = "0.0.0"  # Final fallback

version = __version__

# Import availability checks
# Keys correspond to optional dependency names defined in pyproject.toml.
Imports = {
    "alpha-vantage": find_spec("alpha_vantage") is not None,
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
    "tulipy": find_spec("tulipy") is not None,
    "vectorbt": find_spec("vectorbt") is not None,
    "yaml": find_spec("yaml") is not None,
    "yfinance": find_spec("yfinance") is not None,
}


# Top-level candle indicator names exposed as public API.  All other cdl_*
# pattern files are sub-patterns accessed via cdl_pattern() only.
_CANDLE_TOP_LEVEL = {"cdl_doji", "cdl_inside", "cdl_pattern", "cdl_z", "ha"}

# Subdirectories that contain indicator modules (excludes utils, math, etc.)
_VALID_CATEGORIES = {
    "candles",
    "cycles",
    "math",
    "momentum",
    "overlap",
    "performance",
    "statistics",
    "trend",
    "volatility",
    "volume",
}


def _collect_category_indicators(category_path, category_name):
    """Return a sorted list of public indicator names found in *category_path*.

    Files whose names start with ``_`` are treated as internal helpers and are
    excluded.  For the ``candles`` category only the handful of top-level
    indicators defined in :data:`_CANDLE_TOP_LEVEL` are included; the
    individual ``cdl_*`` pattern modules are accessed through
    ``cdl_pattern()`` and must not appear as standalone indicators.

    Args:
        category_path (Path): Directory to scan.
        category_name (str): Name of the category (e.g. ``"candles"``).

    Returns:
        list[str]: Sorted indicator stem names.
    """
    indicators = []
    for file_path in category_path.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        stem = file_path.stem
        if category_name == "candles" and stem not in _CANDLE_TOP_LEVEL:
            continue
        indicators.append(stem)
    return sorted(indicators)


def _build_category_dict():
    """Dynamically build the Category dictionary by scanning the package
    directory structure.

    Discovers all indicator modules by iterating over valid category
    sub-directories and delegating per-directory collection to
    :func:`_collect_category_indicators`.

    Returns:
        dict: Mapping of category names to sorted lists of indicator names.
    """
    categories = {}
    package_dir = Path(__file__).parent

    for category_path in package_dir.iterdir():
        if not category_path.is_dir():
            continue
        category_name = category_path.name
        if category_name.startswith(("_", ".")) or category_name == "__pycache__" or category_name not in _VALID_CATEGORIES:
            continue
        indicators = _collect_category_indicators(category_path, category_name)
        if indicators:
            categories[category_name] = indicators

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
