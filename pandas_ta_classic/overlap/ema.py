# Exponential Moving Average (EMA)
from typing import Any, Optional
import numpy as np
from pandas import Series
from pandas_ta_classic import Imports

npNaN = np.nan
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def ema(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Exponential Moving Average (EMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    adjust = kwargs.pop("adjust", False)
    sma = kwargs.pop("sma", True)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import EMA

        ema = EMA(close, length)
    else:
        if sma:
            close = close.copy()
            # Find the first valid (non-NaN) position so the SMA seed is
            # computed from exactly `length` consecutive valid values — matching
            # TA-Lib's EMA lookback behaviour for chained/lagged inputs.
            first_valid = close.first_valid_index()
            fv_pos = None if first_valid is None else close.index.get_loc(first_valid)
            if fv_pos is not None:
                sma_nth = close.iloc[fv_pos : fv_pos + length].mean()
                close.iloc[: fv_pos + length - 1] = npNaN
                close.iloc[fv_pos + length - 1] = sma_nth
        ema = close.ewm(span=length, adjust=adjust).mean()

    # Offset
    ema = apply_offset(ema, offset)
    ema = apply_fill(ema, **kwargs)

    # Name & Category
    ema.name = f"EMA_{length}"
    ema.category = "overlap"

    return ema


ema.__doc__ = """Exponential Moving Average (EMA)

The Exponential Moving Average is more responsive moving average compared to the
Simple Moving Average (SMA).  The weights are determined by alpha which is
proportional to it's length.  There are several different methods of calculating
EMA.  One method uses just the standard definition of EMA and another uses the
SMA to generate the initial value for the rest of the calculation.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

Calculation:
    Default Inputs:
        length=10, adjust=False, sma=True
    if sma:
        sma_nth = close[0:length].sum() / length
        close[:length - 1] = np.NaN
        close.iloc[length - 1] = sma_nth
    EMA = close.ewm(span=length, adjust=adjust).mean()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool, optional): Default: False
    sma (bool, optional): If True, uses SMA for initial value. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


def _ema_chain(close: Series, length: int, depth: int, **kwargs):
    """Repeatedly apply EMA, stripping leading NaN between stages.

    This matches TA-Lib's lookback of ``depth * (length - 1)`` for
    indicators like DEMA (depth=2), TEMA (depth=3), and T3 (depth=6).

    Parameters
    ----------
    close : pd.Series
        Input price series.
    length : int
        EMA period.
    depth : int
        Number of EMA passes (2 for DEMA, 3 for TEMA, 6 for T3).
    **kwargs
        Forwarded to each ``ema()`` call.

    Returns
    -------
    list of pd.Series or None
        The ``depth`` EMA results in order (e1, e2, …).  Each result
        retains the full index of the original *close*.  Returns *None*
        if any intermediate EMA call fails.
    """
    results = []
    feed = close
    for _ in range(depth):
        e = ema(close=feed, length=length, talib=False, **kwargs)
        if e is None:
            return None
        results.append(e)
        # Strip leading NaN so the next EMA seeds at the right bar.
        fvi = e.first_valid_index()
        if fvi is None:
            return results  # all-NaN — caller can check
        feed = e.loc[fvi:]
    return results
