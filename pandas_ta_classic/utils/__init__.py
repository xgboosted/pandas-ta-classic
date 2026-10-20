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
from ._time import df_year_to_date, final_time, to_utc, total_time

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

# Names removed in 0.9.0, each with what to use instead. Reading one raises
# AttributeError with this text rather than Python's bare message, and stays an
# AttributeError so hasattr() is False and getattr(..., default) returns the
# default. Each namespace answers only for the names it actually had: this
# module (_REMOVED), the top-level package, pandas_ta_classic.candles and the
# df.ta accessor pick theirs from here. A `from ... import name` still gets
# Python's plain ImportError: the import machinery drops the message.
_DATA_FETCHING_REMOVED = (
    "was removed in 0.9.0 (deprecated in 0.8.32): data fetching is out of scope for a technical-analysis library. "
    "Fetch OHLCV with yfinance or alpha-vantage directly and pass the DataFrame in; examples/fetch_market_data.py shows both"
)
_REMOVED_MESSAGES = {
    "non_zero_range": (
        "non_zero_range() was removed in 0.9.0: it replaced a zero difference with sys.float_info.epsilon, "
        "which turned a degenerate window into an epsilon-scale or epsilon-inflated value. "
        "Take the exact difference (high - low) and mask the divisor instead: (num / den).mask(den == 0, 0.0)"
    ),
    "get_time": (
        "get_time() was removed in 0.9.0: it added a fixed exchange offset to the host's local time, so it was right "
        "only on a UTC host and ignored daylight saving. Use datetime.now(timezone.utc) and convert it with "
        ".astimezone(zoneinfo.ZoneInfo('America/New_York')) or another time zone; df.ta.last_run is a UTC datetime"
    ),
    "EXCHANGE_TZ": (
        "EXCHANGE_TZ was removed in 0.9.0 together with get_time(): its fixed offsets ignored daylight saving. "
        "Use an IANA time zone instead, e.g. zoneinfo.ZoneInfo('America/New_York')"
    ),
    "CDL_PATTERN_NAMES": "CDL_PATTERN_NAMES was removed in 0.9.0 (deprecated in 0.8.32); use ta.ALL_PATTERNS",
    "constants": (
        "df.ta.constants() was removed in 0.9.0 (deprecated in 0.8.32): horizontal chart lines are out of scope. "
        "Add a constant column directly, e.g. df['0'] = 0"
    ),
    "data": f"pandas_ta_classic.utils.data {_DATA_FETCHING_REMOVED}",
    "yf": f"ta.yf() {_DATA_FETCHING_REMOVED}",
    "av": f"ta.av() {_DATA_FETCHING_REMOVED}",
    "ticker": f"df.ta.ticker() {_DATA_FETCHING_REMOVED}",
}
_REMOVED = {name: _REMOVED_MESSAGES[name] for name in ("non_zero_range", "get_time", "data")}


def __getattr__(name: str) -> NoReturn:
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
