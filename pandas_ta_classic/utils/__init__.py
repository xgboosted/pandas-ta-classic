from typing import NoReturn

from ._candles import candle_color
from ._core import (
    apply_fill,
    apply_offset,
    degenerate_div,
    degenerate_zero,
    get_drift,
    get_offset,
    is_datetime_ordered,
    is_percent,
    recent_maximum_index,
    recent_minimum_index,
    signed_series,
    tal_ma,
    unsigned_differences,
    verify_series,
)
from ._math import (
    combination,
    df_error_analysis,
    fibonacci,
    linear_regression,
    np_rolling_moments,
    pascals_triangle,
    symmetric_triangle,
    weights,
    zero,
)

# volatility is re-exported for utils.volatility() access but kept out of
# __all__ so it never shadows the pandas_ta_classic.volatility subpackage.
from ._metrics import (
    cagr,
    calmar_ratio,
    downside_deviation,
    jensens_alpha,
    log_max_drawdown,
    max_drawdown,
    optimal_leverage,
    pure_profit_score,
    sharpe_ratio,
    sortino_ratio,
    volatility as volatility,
)
from ._signals import (
    above,
    above_value,
    below,
    below_value,
    cross,
    cross_value,
    crossover,
    lag,
    signals,
)
from ._time import df_year_to_date, final_time, get_time, to_utc, total_time

__all__ = [
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
    "degenerate_div",
    "degenerate_zero",
    "df_error_analysis",
    "df_year_to_date",
    "downside_deviation",
    "fibonacci",
    "final_time",
    "get_drift",
    "get_offset",
    "get_time",
    "is_datetime_ordered",
    "is_percent",
    "jensens_alpha",
    "lag",
    "linear_regression",
    "log_max_drawdown",
    "max_drawdown",
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
    "verify_series",
    "weights",
    "zero",
]

# Removed in 0.9.0 without a deprecation step (AGENTS.md rule 11). Each name
# answers with its replacement instead of a bare AttributeError.
_REMOVED = {
    "non_zero_range": (
        "non_zero_range() was removed in 0.9.0: it replaced a zero difference with sys.float_info.epsilon, "
        "which turned a degenerate window into an epsilon-scale or epsilon-inflated value. "
        "Take the exact difference (high - low) and mask the divisor instead: (num / den).mask(den == 0, 0.0)"
    ),
}


def __getattr__(name: str) -> NoReturn:
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
