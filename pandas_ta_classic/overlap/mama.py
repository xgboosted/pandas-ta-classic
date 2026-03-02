# -*- coding: utf-8 -*-
# MESA Adaptive Moving Average (MAMA)
from typing import Any, Optional

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._numba import _mama_talib_loop


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
        c_arr = close.to_numpy(dtype=float)
        m = c_arr.shape[0]
        mama_out, fama_out = _mama_talib_loop(c_arr, m, fastlimit, slowlimit)
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
