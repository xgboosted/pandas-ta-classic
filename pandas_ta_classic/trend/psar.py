# -*- coding: utf-8 -*-
# Parabolic SAR (PSAR)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.utils import get_offset, verify_series, zero
from pandas_ta_classic.utils._njit import njit


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
            reverse = h_ > _sar
            if l_ < ep:
                ep = l_
                af = min(af + af0, max_af)
            # Guard row==1: row-2 would be -1 (last element) without the clamp.
            _sar = max(h_arr[row - 1], h_arr[max(0, row - 2)], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = l_ < _sar
            if h_ > ep:
                ep = h_
                af = min(af + af0, max_af)
            # Guard row==1: row-2 would be -1 (last element) without the clamp.
            _sar = min(l_arr[row - 1], l_arr[max(0, row - 2)], _sar)
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
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Parabolic Stop and Reverse (PSAR)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    af = float(af) if af and af > 0 else 0.02
    af0 = float(af0) if af0 and af0 > 0 else af
    max_af = float(max_af) if max_af and max_af > 0 else 0.2
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    def _falling(high: Series, low: Series, drift: int = 1) -> bool:
        """Returns the last -DM value"""
        # Not to be confused with ta.falling()
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
        return _dmn > 0

    # Falling if the first NaN -DM is positive
    falling = _falling(high.iloc[:2], low.iloc[:2]) if len(high) > 1 else False
    if falling:
        sar = high.iloc[0]
        ep = low.iloc[1] if len(low) > 1 else low.iloc[0]
    else:
        sar = low.iloc[0]
        ep = high.iloc[1] if len(high) > 1 else high.iloc[0]

    if close is not None:
        close = verify_series(close)
        sar = close.iloc[0]

    # Calculate Result
    m = high.shape[0]
    h_arr = high.to_numpy(dtype=float)
    l_arr = low.to_numpy(dtype=float)
    long_arr, short_arr, af_arr, reversal_arr = _psar_loop(
        h_arr, l_arr, m, falling, sar, ep, af0, max_af
    )
    long = Series(long_arr, index=high.index)
    short = Series(short_arr, index=high.index)
    _af = Series(af_arr, index=high.index)
    reversal = Series(reversal_arr, index=high.index)

    # Offset
    if offset != 0:
        _af = _af.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)
        reversal = reversal.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        _af.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
        reversal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                _af.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                _af.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                long.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                long.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                short.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                short.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                reversal.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                reversal.bfill(inplace=True)

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
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: long, short, af, and reversal columns.
"""
