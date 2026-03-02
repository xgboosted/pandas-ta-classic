# -*- coding: utf-8 -*-
# Quantitative Qualitative Estimation (QQE)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan

from .rsi import rsi
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_drift,
    get_offset,
    verify_series,
)


def qqe(
    close: Series,
    length: Optional[int] = None,
    smooth: Optional[int] = None,
    factor: Optional[float] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Quantitative Qualitative Estimation (QQE)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    smooth = int(smooth) if smooth and smooth > 0 else 5
    factor = float(factor) if factor else 4.236
    wilders_length = 2 * length - 1
    mamode = mamode if isinstance(mamode, str) else "ema"
    close = verify_series(close, max(length, smooth, wilders_length))
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    rsi_ = rsi(close, length)
    _mode = mamode.lower()[0] if mamode != "ema" else ""
    rsi_ma = ma(mamode, rsi_, length=smooth)
    if rsi_ma is None:
        return None

    # RSI MA True Range
    rsi_ma_tr = rsi_ma.diff(drift).abs()

    # Double Smooth the RSI MA True Range using Wilder's Length with a default
    # width of 4.236.
    smoothed_rsi_tr_ma = ma("ema", rsi_ma_tr, length=wilders_length)
    if smoothed_rsi_tr_ma is None:
        return None
    _dar_ma = ma("ema", smoothed_rsi_tr_ma, length=wilders_length)
    if _dar_ma is None:
        return None
    dar = factor * _dar_ma

    # Create the Upper and Lower Bands around RSI MA.
    upperband = rsi_ma + dar
    lowerband = rsi_ma - dar

    m = close.size
    idx = close.index

    from pandas_ta_classic.utils._numba import _qqe_loop

    rsi_arr = rsi_ma.to_numpy()
    ub_arr = upperband.to_numpy()
    lb_arr = lowerband.to_numpy()

    long_arr, short_arr, trend_arr, qqe_arr, qqe_long_arr, qqe_short_arr = _qqe_loop(
        rsi_arr, ub_arr, lb_arr, m
    )

    long = Series(long_arr, index=idx)
    short = Series(short_arr, index=idx)
    qqe = Series(qqe_arr, index=idx)
    qqe_long = Series(qqe_long_arr, index=idx)
    qqe_short = Series(qqe_short_arr, index=idx)

    # Offset + Name + Category + DataFrame
    _props = f"{_mode}_{length}_{smooth}_{factor}"
    return _build_dataframe(
        {
            f"QQE{_props}": qqe,
            f"QQE{_props}_RSI{_mode.upper()}MA": rsi_ma,
            f"QQEl{_props}": qqe_long,
            f"QQEs{_props}": qqe_short,
            f"QQEb_l{_props}": long,
            f"QQEb_s{_props}": short,
        },
        f"QQE{_props}",
        "momentum",
        offset,
        **kwargs,
    )


qqe.__doc__ = """Quantitative Qualitative Estimation (QQE)

The Quantitative Qualitative Estimation (QQE) is similar to SuperTrend but uses a Smoothed RSI with an upper and lower bands. The band width is a combination of a one period True Range of the Smoothed RSI which is double smoothed using Wilder's smoothing length (2 * rsiLength - 1) and multiplied by the default factor of 4.236. A Long trend is determined when the Smoothed RSI crosses the previous upperband and a Short trend when the Smoothed RSI crosses the previous lowerband.

Based on QQE.mq5 by EarnForex Copyright © 2010, based on version by Tim Hyder (2008), based on version by Roman Ignatov (2006)

Sources:
    https://www.tradingview.com/script/IYfA9R2k-QQE-MT4/
    https://www.tradingpedia.com/forex-trading-indicators/quantitative-qualitative-estimation
    https://www.prorealcode.com/prorealtime-indicators/qqe-quantitative-qualitative-estimation/

Calculation:
    Default Inputs:
        length=14, smooth=5, factor=4.236, mamode="ema", drift=1

Args:
    close (pd.Series): Series of 'close's
    length (int): RSI period. Default: 14
    smooth (int): RSI smoothing period. Default: 5
    factor (float): QQE Factor. Default: 4.236
    mamode (str): See ```help(ta.ma)```. Default: 'sma'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: QQE, RSI_MA (basis), QQEl (sparse long signal),
        QQEs (sparse short signal), QQEb_l (continuous long band),
        and QQEb_s (continuous short band) columns.
"""
