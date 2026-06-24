# Smart Money Concept Liquidity Sweep (SMC_SWEEP)
from typing import Any, Optional
import numpy as np
from pandas import Series, concat
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def smc_sweep(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    wick_mult: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Smart Money Concept Liquidity Sweep"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 15
    wick_mult = float(wick_mult) if wick_mult and wick_mult > 0 else 1.5
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    swing_low = low.rolling(window=length).min().shift(1)
    swing_high = high.rolling(window=length).max().shift(1)

    body = (close - open_).abs()
    oc = concat([open_, close], axis=1)
    lower_wick = oc.min(axis=1) - low
    upper_wick = high - oc.max(axis=1)

    bull_sweep = np.where(
        (low < swing_low) & (close > swing_low) & (close > open_) & (lower_wick > body * wick_mult),
        1, 0,
    )
    bear_sweep = np.where(
        (high > swing_high) & (close < swing_high) & (close < open_) & (upper_wick > body * wick_mult),
        -1, 0,
    )

    result = Series(bull_sweep + bear_sweep, index=close.index)

    # Offset
    result = apply_offset(result, offset)
    result = apply_fill(result, **kwargs)

    # Name and Categorize
    _props = f"_{length}_{round(wick_mult, 4)}"
    result.name = f"SMC_SWEEP{_props}"
    result.category = "momentum"

    return result


smc_sweep.__doc__ = """Smart Money Concept Liquidity Sweep (SMC_SWEEP)

Identifies when price sweeps below a swing low (or above a swing high),
violently rejects it, leaving a long wick and closing in the opposite direction.

Sources:
    Smart Money Concept / ICT trading methodology

Calculation:
    Default Inputs:
        length=15, wick_mult=1.5
    swing_low  = rolling min of low over length bars (shifted 1)
    swing_high = rolling max of high over length bars (shifted 1)
    body       = abs(close - open)
    lower_wick = min(open, close) - low
    upper_wick = high - max(open, close)
    Bull: low < swing_low AND close > swing_low AND close > open AND lower_wick > body * wick_mult → +1
    Bear: high > swing_high AND close < swing_high AND close < open AND upper_wick > body * wick_mult → -1

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): Swing high/low lookback period. Default: 15
    wick_mult (float): Wick-to-body ratio multiplier. Default: 1.5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: 1 (Bullish Sweep), -1 (Bearish Sweep), 0 (None).
"""
