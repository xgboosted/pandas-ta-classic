# -*- coding: utf-8 -*-
# Moving Average Convergence Divergence (MACD)
from typing import Any, Optional
import numpy as np
from pandas import concat, DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    signals,
    verify_series,
)


def macd(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Moving Average, Convergence/Divergence (MACD)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    as_mode = kwargs.setdefault("asmode", False)

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MACD

        macd, signalma, histogram = MACD(close, fast, slow, signal)
    else:

        def _ema_aligned(arr, m, period, seed_end):
            """EMA with SMA seed at a specific index, matching TA-Lib."""
            result = np.full(m, np.nan)
            k = 2.0 / (period + 1)
            start = seed_end - period + 1
            if start < 0 or seed_end >= m:
                return result
            result[seed_end] = arr[start : seed_end + 1].mean()
            for i in range(seed_end + 1, m):
                result[i] = k * arr[i] + (1 - k) * result[i - 1]
            return result

        c_arr = close.to_numpy(dtype=float)
        m = c_arr.shape[0]
        # TA-Lib seeds the fast EMA at its own lookback (fast-1) and the slow
        # EMA at slow-1; by slow-1 the fast EMA has already been running for
        # (slow - fast) additional bars, not reseeded with a fresh SMA.
        fast_start = fast - 1
        slow_start = slow - 1
        fast_ema = _ema_aligned(c_arr, m, fast, fast_start)
        slow_ema = _ema_aligned(c_arr, m, slow, slow_start)
        macd_arr = fast_ema - slow_ema
        sig_start = slow_start + signal - 1
        sig_ema = _ema_aligned(macd_arr, m, signal, sig_start)

        macd = Series(macd_arr, index=close.index)
        signalma = Series(sig_ema, index=close.index)
        histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ema(close=macd.loc[macd.first_valid_index() :,], length=signal)
        if signalma is None:
            return None
        histogram = macd - signalma

    # Offset
    macd, histogram, signalma = apply_offset([macd, histogram, signalma], offset)

    macd, histogram, signalma = apply_fill([macd, histogram, signalma], **kwargs)

    # Name and Categorize it
    _asmode = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    macd.name = f"MACD{_asmode}{_props}"
    histogram.name = f"MACD{_asmode}h{_props}"
    signalma.name = f"MACD{_asmode}s{_props}"
    macd.category = histogram.category = signalma.category = "momentum"

    # Prepare DataFrame to return
    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    df = DataFrame(data)
    df.name = f"MACD{_asmode}{_props}"
    df.category = macd.category

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = concat(
            [
                df,
                signals(
                    indicator=histogram,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", True),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
                signals(
                    indicator=macd,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return df


macd.__doc__ = """Moving Average Convergence Divergence (MACD)

The MACD is a popular indicator to that is used to identify a security's trend.
While APO and MACD are the same calculation, MACD also returns two more series
called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
the difference of MACD and Signal.

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
    AS Mode: https://tr.tradingview.com/script/YFlKXHnP/

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

    if asmode:
        MACD = MACD - Signal
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period. Default: 12
    slow (int): The long period. Default: 26
    signal (int): The signal period. Default: 9
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    asmode (value, optional): When True, enables AS version of MACD.
        Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns.
"""
