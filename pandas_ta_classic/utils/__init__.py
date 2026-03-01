# -*- coding: utf-8 -*-
from ._candles import *
from ._core import *
from ._core import (
    _build_dataframe,
    _finalize,
    _get_min_periods,
    _get_tal_mode,
    _sliding_weighted_ma,
    _sma_seed,
    _swap_fast_slow,
)
from ._math import *
from ._signals import *
from ._time import *
from ._metrics import *
from ._numba import *
from .data import *

__all__ = [
    # _candles
    "candle_color",
    "high_low_range",
    "real_body",
    # _core
    "_build_dataframe",
    "_finalize",
    "_get_min_periods",
    "_get_tal_mode",
    "_sliding_weighted_ma",
    "_sma_seed",
    "_swap_fast_slow",
    "get_drift",
    "get_offset",
    "is_datetime_ordered",
    "is_percent",
    "non_zero_range",
    "recent_maximum_index",
    "recent_minimum_index",
    "signed_series",
    "tal_ma",
    "unsigned_differences",
    "verify_series",
    # _math
    "combination",
    "df_error_analysis",
    "erf",
    "fibonacci",
    "geometric_mean",
    "linear_regression",
    "log_geometric_mean",
    "pascals_triangle",
    "symmetric_triangle",
    "weights",
    "zero",
    # _signals
    "above",
    "above_value",
    "below",
    "below_value",
    "cross",
    "cross_value",
    "signals",
    # _time
    "df_dates",
    "df_month_to_date",
    "df_quarter_to_date",
    "df_year_to_date",
    "final_time",
    "get_time",
    "to_utc",
    "total_time",
    # _metrics
    "cagr",
    "calmar_ratio",
    "downside_deviation",
    "jensens_alpha",
    "log_max_drawdown",
    "max_drawdown",
    "optimal_leverage",
    "pure_profit_score",
    "sharpe_ratio",
    "sortino_ratio",
    "volatility",
    # _numba
    "_rsx_loop",
    "_jma_loop",
    "_hwc_loop",
    "_schaff_tc_loop",
    "_schaff_tc_loop2",
    "_ebsw_loop",
    "_qqe_loop",
    "_lrsi_loop",
    "_pmax_loop",
    "_fisher_loop",
    "_psar_loop",
    "_mcgd_loop",
    "_hwma_loop",
    "_supertrend_loop",
    "_vidya_loop",
    "_ssf2_loop",
    "_ssf3_loop",
    "_hilbert_transform_loop",
    "_sarext_loop",
    # data
    "av",
    "yf",
]
