# -*- coding: utf-8 -*-
# Price Max (PMAX)
from typing import Any, Optional
from numpy import nan as npNaN
from pandas import Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def pmax(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    multiplier: Optional[float] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: PMAX (Price Max)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 10
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    # Calculate ATR
    atr_value = atr(high, low, close, length=length)
    if atr_value is None:
        return None

    # Calculate moving average of close
    ma_value = ma(mamode, close, length=length)
    if ma_value is None:
        return None

    # Calculate PMAX bands
    pmax_up = ma_value - (multiplier * atr_value)
    pmax_down = ma_value + (multiplier * atr_value)

    # Convert to numpy arrays for faster iteration
    close_arr = close.to_numpy()
    pmax_up_arr = pmax_up.to_numpy(copy=True)
    pmax_down_arr = pmax_down.to_numpy(copy=True)

    # Initialize arrays
    n = len(close)
    trend_arr = [1] * n  # Start with uptrend
    pmax_arr = [npNaN] + [0.0] * (n - 1)

    # Iterate using numpy arrays (much faster than pandas .iloc)
    for i in range(1, n):
        # Update upper band: if price was above upper band, maintain higher of current or previous
        if close_arr[i - 1] > pmax_up_arr[i - 1]:
            pmax_up_arr[i] = max(pmax_up_arr[i], pmax_up_arr[i - 1])

        # Update lower band: if price was below lower band, maintain lower of current or previous
        if close_arr[i - 1] < pmax_down_arr[i - 1]:
            pmax_down_arr[i] = min(pmax_down_arr[i], pmax_down_arr[i - 1])

        # Determine trend: price crosses lower band (uptrend) or upper band (downtrend)
        if close_arr[i] > pmax_down_arr[i - 1]:
            trend_arr[i] = 1
        elif close_arr[i] < pmax_up_arr[i - 1]:
            trend_arr[i] = -1
        else:
            trend_arr[i] = trend_arr[i - 1]  # Maintain previous trend

        # Set PMAX value based on trend
        pmax_arr[i] = pmax_up_arr[i] if trend_arr[i] == 1 else pmax_down_arr[i]

    # Convert back to Series
    pmax = Series(pmax_arr, index=close.index)

    # Offset
    pmax = apply_offset(pmax, offset)

    pmax = apply_fill(pmax, **kwargs)

    # Name and Categorize it
    pmax.name = f"PMAX_{mamode[0].upper()}_{length}_{multiplier}"
    pmax.category = "trend"

    return pmax


pmax.__doc__ = """PMAX (Price Max)

PMAX is a trend-following indicator that combines moving averages with ATR
(Average True Range) to create adaptive support and resistance levels. It helps
identify trend direction and potential reversal points.

Sources:
    https://www.tradingview.com/script/sU9molfV/
    https://www.prorealcode.com/prorealtime-indicators/pmax/

Calculation:
    Default Inputs:
        length=10, multiplier=3.0, mamode='ema'

    ATR = ATR(high, low, close, length)
    MA = MA(close, length, mamode)

    PMAX_UP = MA - (multiplier * ATR)
    PMAX_DOWN = MA + (multiplier * ATR)

    If close > PMAX_DOWN[1]: trend = 1 (uptrend)
    If close < PMAX_UP[1]: trend = -1 (downtrend)

    PMAX = PMAX_UP if trend == 1 else PMAX_DOWN

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): ATR period. Default: 10
    multiplier (float): ATR multiplier. Default: 3.0
    mamode (str): Moving average mode. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
