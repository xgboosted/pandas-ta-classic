# -*- coding: utf-8 -*-
# Parabolic SAR (PSAR)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
    zero,
)
from pandas_ta_classic.utils._njit import njit


def _pos_float(val, default):
    return float(val) if val and val > 0 else default


def _psar_falling(high: Series, low: Series, drift: int = 1) -> bool:
    """Returns True when the last -DM is positive (falling market)."""
    up = high - high.shift(drift)
    dn = low.shift(drift) - low
    _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
    return _dmn > 0


@njit(cache=True)
def _psar_loop(h_arr, l_arr, m, falling, sar, ep, af0, max_af):
    long_arr = np.full(m, np.nan)
    short_arr = np.full(m, np.nan)
    af_arr = np.full(m, np.nan)
    reversal_arr = np.zeros(m)
    af_arr[0] = af0
    af = af0
    for row in range(1, m):
        h_ = h_arr[row]
        l_ = l_arr[row]
        if falling:
            _sar = sar + af * (ep - sar)
            if l_ < ep:
                ep = l_
                af = min(af + af0, max_af)
            # Guard row==1: row-2 would be -1 (last element) without the clamp.
            # Guard must be applied before reversal check to match TA-Lib behaviour:
            # the guarded SAR (not the raw projected value) determines the reversal.
            _sar = max(h_arr[row - 1], h_arr[max(0, row - 2)], _sar)
            reverse = h_ > _sar
        else:
            _sar = sar + af * (ep - sar)
            if h_ > ep:
                ep = h_
                af = min(af + af0, max_af)
            # Guard row==1: row-2 would be -1 (last element) without the clamp.
            _sar = min(l_arr[row - 1], l_arr[max(0, row - 2)], _sar)
            reverse = l_ < _sar
        if reverse:
            _sar = ep
            af = af0
            falling = not falling  # Must come before next line
            ep = l_ if falling else h_
        sar = _sar  # Update SAR
        if falling:
            short_arr[row] = sar
        else:
            long_arr[row] = sar
        af_arr[row] = af
        reversal_arr[row] = 1.0 if reverse else 0.0
    return long_arr, short_arr, af_arr, reversal_arr


def psar(
    high: Series,
    low: Series,
    close: Optional[Series] = None,
    af0: Optional[float] = None,
    af: Optional[float] = None,
    max_af: Optional[float] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Parabolic Stop and Reverse (PSAR)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    af = _pos_float(af, 0.02)
    af0 = _pos_float(af0, af)
    max_af = _pos_float(max_af, 0.2)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else False

    if high is None or low is None:
        return None

    if Imports["talib"] and mode_tal:
        from talib import SAR as _SAR

        sar = _SAR(high, low, acceleration=af0, maximum=max_af)
        sar_s = Series(sar, index=high.index)
        sar_s = apply_offset(sar_s, offset)
        _params = f"_{af0}_{max_af}"
        # Split into long (below price) and short (above price) using close or mid
        ref = high  # fallback: use high as reference
        long = sar_s.where(sar_s < ref, other=np.nan)
        short = sar_s.where(sar_s >= ref, other=np.nan)
        _af_s = Series(np.nan, index=high.index)
        _rev_s = Series(0.0, index=high.index)
        data = {
            f"PSARl{_params}": long,
            f"PSARs{_params}": short,
            f"PSARaf{_params}": _af_s,
            f"PSARr{_params}": _rev_s,
        }
        psardf = DataFrame(data)
        psardf.name = f"PSAR{_params}"
        psardf.category = "trend"
        return psardf

    # Falling if the first NaN -DM is positive
    falling = _psar_falling(high.iloc[:2], low.iloc[:2]) if len(high) > 1 else False
    if falling:
        sar = high.iloc[0]
        ep = low.iloc[1] if len(low) > 1 else low.iloc[0]
    else:
        sar = low.iloc[0]
        ep = high.iloc[1] if len(high) > 1 else high.iloc[0]

    if close is not None:
        close = verify_series(close)
        if close is None:
            return None
        sar = close.iloc[0]

    # Calculate Result
    m = high.shape[0]
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    long_arr, short_arr, af_arr, reversal_arr = _psar_loop(
        h_arr, l_arr, m, falling, sar, ep, af0, max_af
    )
    import numpy as _np
    # Combine to a single SAR series then reclassify using close when available.
    # This matches TA-Lib's convention (SAR < close → long, SAR >= close → short)
    # and avoids off-by-one splits at reversal bars when using the falling flag alone.
    _combined = _np.where(~_np.isnan(long_arr), long_arr, short_arr)
    if close is not None:
        close_arr = close.to_numpy(dtype=float)
        long_arr = _np.where(_combined < close_arr, _combined, _np.nan)
        short_arr = _np.where(_combined >= close_arr, _combined, _np.nan)
    long = Series(long_arr, index=high.index)
    short = Series(short_arr, index=high.index)
    _af = Series(af_arr, index=high.index)
    reversal = Series(reversal_arr, index=high.index)

    # Offset
    _af, long, short, reversal = apply_offset([_af, long, short, reversal], offset)

    _af, long, short, reversal = apply_fill([_af, long, short, reversal], **kwargs)

    # Prepare DataFrame to return
    _params = f"_{af0}_{max_af}"
    data = {
        f"PSARl{_params}": long,
        f"PSARs{_params}": short,
        f"PSARaf{_params}": _af,
        f"PSARr{_params}": reversal,
    }
    psardf = DataFrame(data)
    psardf.name = f"PSAR{_params}"
    psardf.category = long.category = short.category = "trend"

    return psardf


psar.__doc__ = """Parabolic Stop and Reverse (psar)

Parabolic Stop and Reverse (PSAR) was developed by J. Wells Wilder, that is used
to determine trend direction and it's potential reversals in price. PSAR uses a
trailing stop and reverse method called "SAR," or stop and reverse, to identify
possible entries and exits. It is also known as SAR.

PSAR indicator typically appears on a chart as a series of dots, either above or
below an asset's price, depending on the direction the price is moving. A dot is
placed below the price when it is trending upward, and above the price when it
is trending downward.

Sources:
    https://www.tradingview.com/pine-script-reference/#fun_sar
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=66&Name=Parabolic

Calculation:
    Default Inputs:
        af0=0.02, af=0.02, max_af=0.2

    See Source links

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series, optional): Series of 'close's. Optional
    af0 (float): Initial Acceleration Factor. Default: 0.02
    af (float): Acceleration Factor. Default: 0.02
    max_af (float): Maximum Acceleration Factor. Default: 0.2
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version (SAR only, long/short/af/reversal columns not available). Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: long, short, af, and reversal columns.
"""
