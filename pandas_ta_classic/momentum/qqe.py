# -*- coding: utf-8 -*-
# Quantitative Qualitative Estimation (QQE)
from typing import Any, Optional
import numpy as np
from pandas import DataFrame, Series

npNaN = np.nan

from .rsi import rsi
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import get_drift, get_offset, verify_series


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
    rsi_arr = rsi_ma.to_numpy()
    ub_arr = upperband.to_numpy()
    lb_arr = lowerband.to_numpy()

    long_arr = np.zeros(m)
    short_arr = np.zeros(m)
    trend_arr = np.ones(m)
    qqe_arr = np.empty(m)
    qqe_arr[0] = rsi_arr[0]
    qqe_long_arr = np.full(m, npNaN)
    qqe_short_arr = np.full(m, npNaN)

    for i in range(1, m):
        c_rsi = rsi_arr[i]
        p_rsi = rsi_arr[i - 1]
        c_long = long_arr[i - 1]
        c_short = short_arr[i - 1]
        p_long = long_arr[i - 2] if i >= 2 else 0.0
        p_short = short_arr[i - 2] if i >= 2 else 0.0

        # Long Line
        if p_rsi > c_long and c_rsi > c_long:
            long_arr[i] = max(c_long, lb_arr[i])
        else:
            long_arr[i] = lb_arr[i]

        # Short Line
        if p_rsi < c_short and c_rsi < c_short:
            short_arr[i] = min(c_short, ub_arr[i])
        else:
            short_arr[i] = ub_arr[i]

        # Trend & QQE Calculation
        if (c_rsi > c_short and p_rsi < p_short) or (
            c_rsi <= c_short and p_rsi >= p_short
        ):
            trend_arr[i] = 1.0
            qqe_arr[i] = qqe_long_arr[i] = long_arr[i]
        elif (c_rsi > c_long and p_rsi < p_long) or (
            c_rsi <= c_long and p_rsi >= p_long
        ):
            trend_arr[i] = -1.0
            qqe_arr[i] = qqe_short_arr[i] = short_arr[i]
        else:
            trend_arr[i] = trend_arr[i - 1]
            if trend_arr[i] == 1.0:
                qqe_arr[i] = qqe_long_arr[i] = long_arr[i]
            else:
                qqe_arr[i] = qqe_short_arr[i] = short_arr[i]

    idx = close.index
    long = Series(long_arr, index=idx)
    short = Series(short_arr, index=idx)
    trend = Series(trend_arr, index=idx)
    qqe = Series(qqe_arr, index=idx)
    qqe_long = Series(qqe_long_arr, index=idx)
    qqe_short = Series(qqe_short_arr, index=idx)

    # Offset
    if offset != 0:
        rsi_ma = rsi_ma.shift(offset)
        qqe = qqe.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rsi_ma.fillna(kwargs["fillna"], inplace=True)
        qqe.fillna(kwargs["fillna"], inplace=True)
        qqe_long.fillna(kwargs["fillna"], inplace=True)
        qqe_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                rsi_ma.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                rsi_ma.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                qqe.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                qqe.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                qqe_long.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                qqe_long.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                qqe_short.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                qqe_short.bfill(inplace=True)

    # Name and Categorize it
    _props = f"{_mode}_{length}_{smooth}_{factor}"
    qqe.name = f"QQE{_props}"
    rsi_ma.name = f"QQE{_props}_RSI{_mode.upper()}MA"
    qqe_long.name = f"QQEl{_props}"
    qqe_short.name = f"QQEs{_props}"
    qqe.category = rsi_ma.category = "momentum"
    qqe_long.category = qqe_short.category = qqe.category

    # Prepare DataFrame to return
    data = {
        qqe.name: qqe,
        rsi_ma.name: rsi_ma,
        # long.name: long, short.name: short
        qqe_long.name: qqe_long,
        qqe_short.name: qqe_short,
    }
    df = DataFrame(data)
    df.name = f"QQE{_props}"
    df.category = qqe.category

    return df


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
    pd.DataFrame: QQE, RSI_MA (basis), QQEl (long), and QQEs (short) columns.
"""
