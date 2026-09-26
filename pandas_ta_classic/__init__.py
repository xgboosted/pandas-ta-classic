import logging
from typing import Any

# Metadata comes from _meta to avoid circular imports; it must be imported
# before core/utils because submodules read Imports off this package.
from pandas_ta_classic._meta import (
    _VALID_CATEGORIES,
    CANGLE_AGG,
    RATE,
    Category,
    Imports,
    version,
)
from pandas_ta_classic.core import (
    AllStrategy,
    AnalysisIndicators,
    CommonStrategy,
    Strategy,
)
from pandas_ta_classic.utils import (
    above,
    above_value,
    apply_fill,
    apply_offset,
    below,
    below_value,
    cagr,
    calmar_ratio,
    candle_color,
    combination,
    cross,
    cross_value,
    crossover,
    df_error_analysis,
    df_year_to_date,
    downside_deviation,
    fibonacci,
    final_time,
    get_drift,
    get_offset,
    is_datetime_ordered,
    is_percent,
    jensens_alpha,
    lag,
    linear_regression,
    log_max_drawdown,
    max_drawdown,
    non_zero_range,
    np_rolling_moments,
    optimal_leverage,
    pascals_triangle,
    pure_profit_score,
    recent_maximum_index,
    recent_minimum_index,
    sharpe_ratio,
    signals,
    signed_series,
    sortino_ratio,
    symmetric_triangle,
    tal_ma,
    to_utc,
    total_time,
    unsigned_differences,
    verify_series,
    weights,
    zero,
)

from . import utils

name = "pandas-ta-classic"
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = version
__description__ = (
    "An easy to use Python 3 Pandas Extension providing a comprehensive set of Technical Analysis indicators."
    "Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."
    "This is the classic/community maintained version."
)

__all__ = [
    "CANGLE_AGG",
    "RATE",
    "AllStrategy",
    "AnalysisIndicators",
    "Category",
    "CommonStrategy",
    "Imports",
    "Strategy",
    "above",
    "above_value",
    "apply_fill",
    "apply_offset",
    "below",
    "below_value",
    "cagr",
    "calmar_ratio",
    "candle_color",
    "combination",
    "cross",
    "cross_value",
    "crossover",
    "df_error_analysis",
    "df_year_to_date",
    "downside_deviation",
    "fibonacci",
    "final_time",
    "get_drift",
    "get_offset",
    "is_datetime_ordered",
    "is_percent",
    "jensens_alpha",
    "lag",
    "linear_regression",
    "log_max_drawdown",
    "max_drawdown",
    "name",
    "non_zero_range",
    "np_rolling_moments",
    "optimal_leverage",
    "pascals_triangle",
    "pure_profit_score",
    "recent_maximum_index",
    "recent_minimum_index",
    "sharpe_ratio",
    "signals",
    "signed_series",
    "sortino_ratio",
    "symmetric_triangle",
    "tal_ma",
    "to_utc",
    "total_time",
    "unsigned_differences",
    "utils",
    "verify_series",
    "version",
    "volatility",
    "weights",
    "zero",
]


def __dir__() -> list[str]:
    from pandas_ta_classic._indicator_loader import _INDICATOR_TO_CATEGORY

    names = set(globals().keys())
    names.update(_INDICATOR_TO_CATEGORY.keys())
    names.update(_VALID_CATEGORIES)
    names.update(("ALL_PATTERNS", "cdl"))
    return sorted(names)


def __getattr__(name: str) -> Any:
    """Lazy-load indicator functions for direct module-level access (e.g. ta.rsi(...)).

    For regular indicators (in Category), returns the callable function.
    For individual candle-pattern submodules (cdl_2crows, cdl_3blackcrows, etc.
    that live in pandas_ta_classic.candles but are NOT in Category), returns the
    submodule directly — preserving the ta.cdl_2crows.cdl_2crows(...) access
    pattern used by some tests and the old wildcard-import behaviour.
    """
    import importlib
    import sys

    from pandas_ta_classic._indicator_loader import (
        _INDICATOR_TO_CATEGORY,
        _MATH_ALIASES,
        _find_indicator_func,
    )

    # Regular indicators in Category → return the function
    cat = _INDICATOR_TO_CATEGORY.get(name)
    if cat is not None:
        try:
            func = _find_indicator_func(name)
        except ModuleNotFoundError as exc:
            # only a missing indicator module means "no such attribute"; a missing
            # dependency inside it must surface as itself
            if exc.name != f"pandas_ta_classic.{cat}.{_MATH_ALIASES.get(name, name)}":
                raise
            raise AttributeError(f"module 'pandas_ta_classic' has no attribute '{name}'") from exc
        setattr(sys.modules[__name__], name, func)  # cache in module dict
        return func

    # Category subpackages (momentum, overlap, trend, volume, statistics,
    # candles, cycles, math; also volatility/performance, which are already
    # bound above/transitively but are handled here too so the guarantee
    # holds even if that changes) — resolved lazily via PEP 562 so
    # `ta.<category>` is deterministic regardless of what has already been
    # accessed, without importing all 10 categories at package-import time.
    if name in _VALID_CATEGORIES:
        mod = importlib.import_module(f"{__name__}.{name}")
        setattr(sys.modules[__name__], name, mod)  # cache in module dict
        return mod

    # cdl: the shorthand candle-pattern wrapper. It lives inside
    # candles/cdl_pattern.py rather than a cdl.py of its own, so it is not a
    # Category entry and neither branch above finds it.
    if name == "cdl":
        from pandas_ta_classic.candles.cdl_pattern import cdl

        setattr(sys.modules[__name__], name, cdl)  # cache in module dict
        return cdl

    # ALL_PATTERNS: canonical public name for the candle pattern name list
    if name == "ALL_PATTERNS":
        from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS

        setattr(sys.modules[__name__], name, ALL_PATTERNS)
        return ALL_PATTERNS

    # Individual candle-pattern submodules (cdl_*) not tracked in Category
    # → return the submodule (mimics old `from candles import *` behaviour)
    if name.startswith("cdl_"):
        try:
            mod = importlib.import_module(f"pandas_ta_classic.candles.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"pandas_ta_classic.candles.{name}":
                raise  # a dependency of the module is missing, not the module
        else:
            setattr(sys.modules[__name__], name, mod)  # cache in module dict
            return mod

    raise AttributeError(f"module 'pandas_ta_classic' has no attribute '{name}'")
