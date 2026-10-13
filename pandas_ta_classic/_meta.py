"""
Meta information for pandas-ta-classic
Contains Category definitions, version information, and import checks.
"""

import pkgutil
from importlib.util import find_spec
from pathlib import Path

# Version information - dynamically determined from git tags via setuptools_scm.
# An installed package always has one of the two; a made-up "0.0.0" used to
# hide a broken install.
try:
    from pandas_ta_classic._version import version as __version__
except ImportError:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("pandas-ta-classic")  # raises PackageNotFoundError when not installed

version = __version__

# Import availability checks
# Keys correspond to optional dependency names defined in pyproject.toml.
Imports = {
    "talib": find_spec("talib") is not None,
    "tqdm": find_spec("tqdm") is not None,
    "tulipy": find_spec("tulipy") is not None,
}


# Top-level candle indicator names exposed as public API.  All other cdl_*
# pattern files are sub-patterns accessed via cdl_pattern() only.
_CANDLE_TOP_LEVEL = {"cdl_doji", "cdl_inside", "cdl_pattern", "cdl_z", "ha"}

_MATH_ALIASES: dict[str, str] = {"max": "rolling_max", "min": "rolling_min", "sum": "rolling_sum"}

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

    Modules are listed through the import system (:func:`pkgutil.iter_modules`),
    not the file system, so a package imported from a zip archive is scanned
    too. Modules whose names start with ``_`` are treated as internal helpers and are
    excluded.  For the ``candles`` category only the handful of top-level
    indicators defined in :data:`_CANDLE_TOP_LEVEL` are included; the
    individual ``cdl_*`` pattern modules are accessed through
    ``cdl_pattern()`` and must not appear as standalone indicators.

    Args:
        category_path (Path): Directory (or path inside a zip archive) to scan.
        category_name (str): Name of the category (e.g. ``"candles"``).

    Returns:
        list[str]: Sorted indicator stem names.
    """
    indicators = []
    for module in pkgutil.iter_modules([str(category_path)]):
        if module.ispkg or module.name.startswith("_"):
            continue
        if category_name == "candles" and module.name not in _CANDLE_TOP_LEVEL:
            continue
        indicators.append(module.name)
    return sorted(indicators)


def _build_category_dict():
    """Dynamically build the Category dictionary by scanning the package
    structure.

    Discovers all indicator modules by iterating over the valid category
    subpackages, in name order, and delegating per-subpackage collection to
    :func:`_collect_category_indicators`. ``Path.iterdir`` used to fail with
    FileNotFoundError when the package was imported from a zip archive, and its
    order was the file system's.

    Returns:
        dict: Mapping of category names to sorted lists of indicator names.
    """
    categories = {}
    package_dir = Path(__file__).parent

    for package in sorted(pkgutil.iter_modules([str(package_dir)]), key=lambda m: m.name):
        category_name = package.name
        if not package.ispkg or category_name not in _VALID_CATEGORIES:
            continue
        indicators = _collect_category_indicators(package_dir / category_name, category_name)
        if indicators:
            categories[category_name] = indicators

    # A layout pkgutil cannot list (frozen, sourceless) would leave Category
    # empty without a word, and every lookup fail later; say so here instead.
    missing = sorted(_VALID_CATEGORIES - categories.keys())
    if missing:
        raise ImportError(f"pandas_ta_classic could not list the indicator modules of {missing} under {package_dir}")

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
