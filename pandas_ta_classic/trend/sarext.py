# Parabolic SAR Extended (SAREXT)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
    zero,
)
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_float, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


def _sarext_falling(high: Series, low: Series, drift: int = 1) -> bool:
    up = high - high.shift(drift)
    dn = low.shift(drift) - low
    _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
    return _dmn > 0


@njit(cache=True)
def _sarext_loop(
    h_arr,
    l_arr,
    m,
    falling,
    sar,
    ep,
    af0_long,
    af_long,
    max_af_long,
    af0_short,
    af_short,
    max_af_short,
    offset_on_reverse,
):
    # Same state machine as TA-Lib's SAREXT: each bar first publishes the SAR
    # carried into it (reversing when price crosses it, restarting from the
    # prior extreme point and applying offset_on_reverse away from price),
    # then projects the SAR for the next bar, clamped by this and the
    # previous bar's range.
    long_arr = np.full(m, np.nan)
    short_arr = np.full(m, np.nan)
    af_arr = np.full(m, np.nan)
    af0_long = min(af0_long, max_af_long)
    af_long = min(af_long, max_af_long)
    af0_short = min(af0_short, max_af_short)
    af_short = min(af_short, max_af_short)
    afl = af0_long
    afs = af0_short
    # TA-Lib seeds the "previous bar" with bar 1 itself, so the first
    # projection is clamped by bar 1's range only.
    new_high = h_arr[1]
    new_low = l_arr[1]

    for row in range(1, m):
        prev_high = new_high
        prev_low = new_low
        new_high = h_arr[row]
        new_low = l_arr[row]

        if not falling:
            if new_low <= sar:
                falling = True
                sar = max(ep, prev_high, new_high)
                if offset_on_reverse != 0.0:
                    sar += sar * offset_on_reverse
                short_arr[row] = sar
                afs = af0_short
                ep = new_low
                sar = max(sar + afs * (ep - sar), prev_high, new_high)
            else:
                long_arr[row] = sar
                if new_high > ep:
                    ep = new_high
                    afl = min(afl + af_long, max_af_long)
                sar = min(sar + afl * (ep - sar), prev_low, new_low)
        else:
            if new_high >= sar:
                falling = False
                sar = min(ep, prev_low, new_low)
                if offset_on_reverse != 0.0:
                    sar -= sar * offset_on_reverse
                long_arr[row] = sar
                afl = af0_long
                ep = new_high
                sar = min(sar + afl * (ep - sar), prev_low, new_low)
            else:
                short_arr[row] = sar
                if new_low < ep:
                    ep = new_low
                    afs = min(afs + af_short, max_af_short)
                sar = max(sar + afs * (ep - sar), prev_high, new_high)
        af_arr[row] = afs if falling else afl

    return long_arr, short_arr, af_arr


def _sarext_native_result(
    high,
    low,
    startvalue,
    af0_long,
    af_long,
    max_af_long,
    af0_short,
    af_short,
    max_af_short,
    offsetonreverse,
):
    """Run the native SAREXT computation and return a signed Series."""
    # TA-Lib: a non-zero startvalue fixes the first direction by its sign and
    # |startvalue| is the first SAR; otherwise the direction comes from the
    # first bar's directional movement and the SAR from the previous bar.
    if startvalue > 0:
        falling, sar, ep = False, startvalue, high.iloc[1]
    elif startvalue < 0:
        falling, sar, ep = True, abs(startvalue), low.iloc[1]
    else:
        falling = bool(_sarext_falling(high.iloc[:2], low.iloc[:2]))
        sar = high.iloc[0] if falling else low.iloc[0]
        ep = low.iloc[1] if falling else high.iloc[1]
    m = high.shape[0]
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    long_arr, short_arr, _af_arr = _sarext_loop(
        h_arr,
        l_arr,
        m,
        falling,
        sar,
        ep,
        af0_long,
        af_long,
        max_af_long,
        af0_short,
        af_short,
        max_af_short,
        offsetonreverse,
    )
    result = np.where(
        ~np.isnan(long_arr),
        long_arr,
        np.where(~np.isnan(short_arr), -short_arr, np.nan),
    )
    return Series(result, index=high.index)


@nan_on_short_input
@skip_leading_nan("high", "low")
def sarext(
    high: Series,
    low: Series,
    startvalue: float | None = None,
    offsetonreverse: float | None = None,
    accelerationinitlong: float | None = None,
    accelerationlong: float | None = None,
    accelerationmaxlong: float | None = None,
    accelerationinitshort: float | None = None,
    accelerationshort: float | None = None,
    accelerationmaxshort: float | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Parabolic SAR Extended (SAREXT)"""
    # Validate Arguments
    high = verify_series(high, 2)
    low = verify_series(low, 2)
    startvalue = _number(startvalue, 0.0, "startvalue")
    offsetonreverse = _number(offsetonreverse, 0.0, "offsetonreverse", ge=0)
    af0_long = _pos_float(accelerationinitlong, 0.02, "accelerationinitlong")
    af_long = _pos_float(accelerationlong, 0.02, "accelerationlong")
    max_af_long = _pos_float(accelerationmaxlong, 0.2, "accelerationmaxlong")
    af0_short = _pos_float(accelerationinitshort, 0.02, "accelerationinitshort")
    af_short = _pos_float(accelerationshort, 0.02, "accelerationshort")
    max_af_short = _pos_float(accelerationmaxshort, 0.2, "accelerationmaxshort")
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if high is None or low is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import SAREXT as TASAREXT

        sarext_ = TASAREXT(
            high,
            low,
            startvalue=startvalue,
            offsetonreverse=offsetonreverse,
            accelerationinitlong=af0_long,
            accelerationlong=af_long,
            accelerationmaxlong=max_af_long,
            accelerationinitshort=af0_short,
            accelerationshort=af_short,
            accelerationmaxshort=max_af_short,
        )
    else:
        sarext_ = _sarext_native_result(
            high,
            low,
            startvalue,
            af0_long,
            af_long,
            max_af_long,
            af0_short,
            af_short,
            max_af_short,
            offsetonreverse,
        )

    # Offset
    sarext_ = apply_offset(sarext_, offset)

    sarext_ = apply_fill(sarext_, **kwargs)

    # Name and Categorize it
    sarext_.name = "SAREXT"
    sarext_.category = "trend"

    return sarext_


sarext.__doc__ = """Parabolic SAR Extended (SAREXT)

The Parabolic SAR Extended is an enhanced version of the Parabolic SAR that
allows separate acceleration factor settings for long and short positions,
plus an optional offset applied when a reversal occurs.

Sources:
    https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    startvalue (float): Starting SAR value (0 = use first high/low). Default: 0
    offsetonreverse (float): Fractional offset added to SAR on reversal. Default: 0
    accelerationinitlong (float): Initial AF for long positions. Default: 0.02
    accelerationlong (float): AF increment for long positions. Default: 0.02
    accelerationmaxlong (float): Maximum AF for long positions. Default: 0.2
    accelerationinitshort (float): Initial AF for short positions. Default: 0.02
    accelerationshort (float): AF increment for short positions. Default: 0.02
    accelerationmaxshort (float): Maximum AF for short positions. Default: 0.2
    talib (bool): Use TA-Lib if installed. Default: False
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: SAREXT values (positive = long SAR, negative = short SAR).
"""
