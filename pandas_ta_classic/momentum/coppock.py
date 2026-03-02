# Coppock Curve (COPC)
from typing import Any, Optional
from pandas import Series
from .roc import roc
from pandas_ta_classic.overlap.wma import wma
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def coppock(
    close: Series,
    length: Optional[int] = None,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Coppock Curve (COPC)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    fast = int(fast) if fast and fast > 0 else 11
    slow = int(slow) if slow and slow > 0 else 14
    close = verify_series(close, max(length, fast, slow))
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    total_roc = roc(close, fast) + roc(close, slow)
    coppock = wma(total_roc, length)

    return _finalize(
        coppock, offset, f"COPC_{fast}_{slow}_{length}", "momentum", **kwargs
    )


coppock.__doc__ = """Coppock Curve (COPC)

Coppock Curve (originally called the "Trendex Model") is a momentum indicator
is designed for use on a monthly time scale.  Although designed for monthly
use, a daily calculation over the same period can be made, converting the
periods to 294-day and 231-day rate of changes, and a 210-day weighted
moving average.

Sources:
    https://en.wikipedia.org/wiki/Coppock_curve

Calculation:
    Default Inputs:
        length=10, fast=11, slow=14
    SMA = Simple Moving Average
    MAD = Mean Absolute Deviation
    tp = typical_price = hlc3 = (high + low + close) / 3
    mean_tp = SMA(tp, length)
    mad_tp = MAD(tp, length)
    CCI = (tp - mean_tp) / (c * mad_tp)

Args:
    close (pd.Series): Series of 'close's
    length (int): WMA period. Default: 10
    fast (int): Fast ROC period. Default: 11
    slow (int): Slow ROC period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
