# -*- coding: utf-8 -*-
# MESA Adaptive Moving Average (MAMA)
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.cycles._hilbert import hilbert_result
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_offset,
    verify_series,
)


try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _mama_loop(
    close_arr: np.ndarray,
    smooth_price: np.ndarray,
    in_phase: np.ndarray,
    quadrature: np.ndarray,
    m: int,
    ht_start: int,
    fastlimit: float,
    slowlimit: float,
) -> tuple:
    """MAMA / FAMA adaptive filter loop (TA-Lib compatible).

    Uses instantaneous phase ``atan(Q1/I1)`` (not dc_phase) and
    ``delta_phase = prevPhase - phase`` matching TA-Lib's convention.
    TA-Lib initialises MAMA/FAMA to ``smoothPrice[ht_start]`` and
    only runs the Hilbert phase computation from bar ``ht_start``.
    """
    mama_arr = np.full(m, np.nan)
    fama_arr = np.full(m, np.nan)

    rad2deg = 180.0 / np.pi
    prev_phase = 0.0

    # TA-Lib initialises mama/fama to the WMA smooth price at the
    # start of the Hilbert warmup (the "todayValue" after the WMA loop).
    mama_val = smooth_price[ht_start] if ht_start < m else close_arr[0]
    fama_val = mama_val

    # TA-Lib lookback for MAMA is 32 (ht_start + 20 convergence bars).
    lookback = 32

    for i in range(ht_start, m):
        # Instantaneous phase from Q1 / I1 (TA-Lib MAMA convention).
        ip = in_phase[i]
        qd = quadrature[i]
        if np.isnan(ip) or np.isnan(qd):
            phase = 0.0
        elif ip != 0.0:
            phase = np.arctan(qd / ip) * rad2deg
        else:
            phase = 0.0

        # TA-Lib: delta_phase = prevPhase - phase (positive when slowing)
        delta_phase = prev_phase - phase
        prev_phase = phase

        if delta_phase < 1.0:
            delta_phase = 1.0

        alpha = fastlimit / delta_phase
        if alpha < slowlimit:
            alpha = slowlimit
        if alpha > fastlimit:
            alpha = fastlimit

        mama_val = alpha * close_arr[i] + (1.0 - alpha) * mama_val
        fama_val = 0.5 * alpha * mama_val + (1.0 - 0.5 * alpha) * fama_val

        if i >= lookback:
            mama_arr[i] = mama_val
            fama_arr[i] = fama_val

    return mama_arr, fama_arr


def mama(
    close: Series,
    fastlimit: Optional[float] = None,
    slowlimit: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: MESA Adaptive Moving Average (MAMA)"""
    # Validate Arguments
    fastlimit = float(fastlimit) if fastlimit and fastlimit > 0 else 0.5
    slowlimit = float(slowlimit) if slowlimit and slowlimit > 0 else 0.05
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MAMA as taMAMA

        mama_arr, fama_arr = taMAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
        mama_s = Series(mama_arr, index=close.index)
        fama_s = Series(fama_arr, index=close.index)
    else:
        ht = hilbert_result(close)
        c_arr = close.to_numpy(dtype=float)
        m = c_arr.shape[0]
        # WMA(4) smooth price at ht_start — matches TA-Lib "todayValue"
        # after its WMA warmup loop.
        ht_start = 12
        if m > ht_start and ht_start >= 3:
            sp_init = (
                4.0 * c_arr[ht_start]
                + 3.0 * c_arr[ht_start - 1]
                + 2.0 * c_arr[ht_start - 2]
                + c_arr[ht_start - 3]
            ) / 10.0
        else:
            sp_init = c_arr[0]
        # Build a tiny smooth_price array containing just the init value.
        sp_arr = np.zeros(m)
        sp_arr[ht_start] = sp_init
        mama_out, fama_out = _mama_loop(
            c_arr,
            sp_arr,
            ht["in_phase"],
            ht["quadrature"],
            m,
            ht_start,
            fastlimit,
            slowlimit,
        )
        mama_s = Series(mama_out, index=close.index)
        fama_s = Series(fama_out, index=close.index)

    _params = f"_{fastlimit}_{slowlimit}"
    return _build_dataframe(
        {f"MAMA{_params}": mama_s, f"FAMA{_params}": fama_s},
        f"MAMA{_params}",
        "overlap",
        offset,
        **kwargs,
    )


mama.__doc__ = """MESA Adaptive Moving Average (MAMA)

MAMA adapts to price movement based on the rate of change of the Hilbert
Transform phase.  It uses a fast limit and slow limit to bound the
smoothing factor.  FAMA (Following Adaptive Moving Average) is a further
smoothed version of MAMA.

Sources:
    John F. Ehlers, "MESA and Trading Market Cycles"

Calculation:
    Default Inputs:
        fastlimit=0.5, slowlimit=0.05
    alpha = fastlimit / delta_phase  (clamped to [slowlimit, fastlimit])
    MAMA = alpha * close + (1 - alpha) * MAMA[1]
    FAMA = 0.5 * alpha * MAMA + (1 - 0.5 * alpha) * FAMA[1]

Args:
    close (pd.Series): Series of 'close's
    fastlimit (float): Upper bound for the adaptive alpha. Default: 0.5
    slowlimit (float): Lower bound for the adaptive alpha. Default: 0.05
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: MAMA and FAMA columns.
"""
