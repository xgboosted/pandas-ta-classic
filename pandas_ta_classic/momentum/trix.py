# -*- coding: utf-8 -*-
# TRIX (TRIX)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)


def trix(
    close: Series,
    length: Optional[int] = None,
    signal: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Trix (TRIX)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    signal = int(signal) if signal and signal > 0 else 9
    scalar = float(scalar) if scalar else 100
    close = verify_series(close, max(length, signal))
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    ema1 = ema(close=close, length=length, **kwargs)
    ema2 = ema(close=ema1, length=length, **kwargs)
    ema3 = ema(close=ema2, length=length, **kwargs)
    if ema1 is None or ema2 is None or ema3 is None:
        return None
    trix = scalar * ema3.pct_change(drift)

    trix_signal = trix.rolling(signal).mean()

    # Offset
    trix, trix_signal = apply_offset([trix, trix_signal], offset)

    trix, trix_signal = apply_fill([trix, trix_signal], **kwargs)

    # Name & Category
    trix.name = f"TRIX_{length}_{signal}"
    trix_signal.name = f"TRIXs_{length}_{signal}"
    trix.category = trix_signal.category = "momentum"

    # Prepare DataFrame to return
    df = DataFrame({trix.name: trix, trix_signal.name: trix_signal})
    df.name = f"TRIX_{length}_{signal}"
    df.category = "momentum"

    return df


trix.__doc__ = """Trix (TRIX)

TRIX is a momentum oscillator to identify divergences.

Sources:
    https://www.tradingview.com/wiki/TRIX

Calculation:
    Default Inputs:
        length=18, drift=1
    EMA = Exponential Moving Average
    ROC = Rate of Change
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TRIX = 100 * ROC(ema3, drift)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 18
    signal (int): It's period. Default: 9
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
