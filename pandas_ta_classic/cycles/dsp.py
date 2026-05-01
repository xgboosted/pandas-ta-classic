# -*- coding: utf-8 -*-
# Detrended Synthetic Price (DSP)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def dsp(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Detrended Synthetic Price (DSP)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    # Calculate EMA
    ema_value = ema(close, length=length)
    if ema_value is None:
        return None

    # Detrend by subtracting EMA
    dsp = close - ema_value

    # Offset
    dsp = apply_offset(dsp, offset)

    dsp = apply_fill(dsp, **kwargs)

    # Name and Categorize it
    dsp.name = f"DSP_{length}"
    dsp.category = "cycles"

    return dsp


dsp.__doc__ = """Detrended Synthetic Price (DSP)

Detrended Synthetic Price removes the trend component from price data to reveal
the cyclical component. It's useful for cycle analysis and identifying periodic
patterns in price movement.

Sources:
    https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf
    Cycle Analytics for Traders by John F. Ehlers

Calculation:
    Default Inputs:
        length=14

    EMA = EMA(close, length)
    DSP = close - EMA

Args:
    close (pd.Series): Series of 'close's
    length (int): The EMA period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
