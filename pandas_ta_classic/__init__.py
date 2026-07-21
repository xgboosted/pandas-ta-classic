import logging
from typing import Any

# Metadata comes from _meta to avoid circular imports; it must be imported
# before core/utils because submodules read Imports off this package.
# EXCHANGE_TZ is intentionally NOT imported here: it is deprecated as a public
# top-level name and served lazily (with a FutureWarning) via __getattr__.
# Internal callers import it directly from pandas_ta_classic._meta.
from pandas_ta_classic._meta import (
    CANGLE_AGG,
    Category,
    Imports,
    RATE,
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
    av,
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
    yf,
    zero,
)

# The utils.volatility() metric stays off the top-level namespace so it never
# shadows this subpackage.
from . import utils
from . import volatility

name = "pandas-ta-classic"
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = version
__description__ = (
    "An easy to use Python 3 Pandas Extension providing a comprehensive set of Technical Analysis indicators."
    "Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."
    "This is the classic/community maintained version."
)

__all__ = [
    "AllStrategy",
    "AnalysisIndicators",
    "CANGLE_AGG",
    "Category",
    "CommonStrategy",
    "Imports",
    "RATE",
    "Strategy",
    "above",
    "above_value",
    "apply_fill",
    "apply_offset",
    "av",
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
    "yf",
    "zero",
]


def __dir__() -> list[str]:
    from pandas_ta_classic._indicator_loader import _INDICATOR_TO_CATEGORY

    names = set(globals().keys())
    names.update(_INDICATOR_TO_CATEGORY.keys())
    names.add("ALL_PATTERNS")
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
        _find_indicator_func,
        _INDICATOR_TO_CATEGORY,
    )

    # Regular indicators in Category → return the function
    cat = _INDICATOR_TO_CATEGORY.get(name)
    if cat is not None:
        try:
            func = _find_indicator_func(name)
        except ModuleNotFoundError:
            raise AttributeError(f"module 'pandas_ta_classic' has no attribute '{name}'")
        setattr(sys.modules[__name__], name, func)  # cache in module dict
        return func

    # ALL_PATTERNS: canonical public name for the candle pattern name list
    if name == "ALL_PATTERNS":
        from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS

        setattr(sys.modules[__name__], name, ALL_PATTERNS)
        return ALL_PATTERNS

    # CDL_PATTERN_NAMES: deprecated alias — use ALL_PATTERNS
    # Not cached via setattr: caching would resolve future lookups directly off
    # the module dict, bypassing __getattr__ (and the warning) after first use.
    if name == "CDL_PATTERN_NAMES":
        import warnings

        warnings.warn(
            "CDL_PATTERN_NAMES is deprecated; use ALL_PATTERNS instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from pandas_ta_classic.candles.cdl_pattern import ALL_PATTERNS

        return ALL_PATTERNS

    # get_time / EXCHANGE_TZ: deprecated as public top-level names. These are
    # wall-clock / exchange-timezone helpers unrelated to computing indicators;
    # they remain available (internally used by the Strategy run timer) but the
    # public exports emit a FutureWarning. Internal code imports them from
    # pandas_ta_classic._meta / pandas_ta_classic.utils directly (no warning).
    # Not cached via setattr, same reason as CDL_PATTERN_NAMES above — the
    # warning must fire on every access, not just the first.
    if name in ("get_time", "EXCHANGE_TZ"):
        import warnings

        warnings.warn(
            f"pandas_ta_classic.{name} is deprecated and will be removed in a "
            "future release; time/exchange-clock helpers are out of scope for a "
            "technical-analysis library.",
            FutureWarning,
            stacklevel=2,
        )
        if name == "get_time":
            from pandas_ta_classic.utils import get_time as _obj
        else:
            from pandas_ta_classic._meta import EXCHANGE_TZ as _obj
        return _obj

    # Individual candle-pattern submodules (cdl_*) not tracked in Category
    # → return the submodule (mimics old `from candles import *` behaviour)
    if name.startswith("cdl_"):
        try:
            mod = importlib.import_module(f"pandas_ta_classic.candles.{name}")
            setattr(sys.modules[__name__], name, mod)  # cache in module dict
            return mod
        except ModuleNotFoundError:
            pass
        except ImportError:
            raise

    raise AttributeError(f"module 'pandas_ta_classic' has no attribute '{name}'")
