# -*- coding: utf-8 -*-
# Parabolic SAR (PSAR)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan
from pandas_ta_classic.utils import get_offset, verify_series, zero


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
    falling = _falling(high.iloc[:2], low.iloc[:2])
    if falling:
        sar = high.iloc[0]
        ep = low.iloc[0]
    else:
        sar = low.iloc[0]
        ep = high.iloc[0]

    if close is not None:
        close = verify_series(close)
        sar = close.iloc[0]

    # Use raw numpy arrays to avoid pandas iloc overhead in the loop.
    h_arr = high.to_numpy()
    l_arr = low.to_numpy()
    m = h_arr.shape[0]
    long_arr = np.full(m, npNaN)
    short_arr = np.full(m, npNaN)
    reversal_arr = np.zeros(m)
    af_arr = np.full(m, npNaN)
    af_arr[0] = af0
    if m > 1:
        af_arr[1] = af0

    # Calculate Result
    for row in range(1, m):
        high_ = h_arr[row]
        low_ = l_arr[row]

        if falling:
            _sar = sar + af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                af = min(af + af0, max_af)

            _sar = max(h_arr[row - 1], h_arr[row - 2], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                af = min(af + af0, max_af)

            _sar = min(l_arr[row - 1], l_arr[row - 2], _sar)

        if reverse:
            _sar = ep
            af = af0
            falling = not falling  # Must come before next line
            ep = low_ if falling else high_

        sar = _sar  # Update SAR

        # Separate long/short sar based on falling
        if falling:
            short_arr[row] = sar
        else:
            long_arr[row] = sar

        af_arr[row] = af
        reversal_arr[row] = int(reverse)

    long = Series(long_arr, index=high.index)
    short = Series(short_arr, index=high.index)
    reversal = Series(reversal_arr, index=high.index)
    _af = Series(af_arr, index=high.index)

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
        fill = kwargs["fill_method"]
        if fill == "ffill":
            _af.ffill(inplace=True)
            long.ffill(inplace=True)
            short.ffill(inplace=True)
            reversal.ffill(inplace=True)
        elif fill == "bfill":
            _af.bfill(inplace=True)
            long.bfill(inplace=True)
            short.bfill(inplace=True)
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
