# -*- coding: utf-8 -*-
# Average Directional Movement Index Rating (ADXR)
from typing import Any, Optional

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.trend.adx import adx
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _build_dataframe,
    get_drift,
    get_offset,
    verify_series,
)


def adxr(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    lensig: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Average Directional Movement Index Rating (ADXR)"""
    # Validate Arguments
    length = length if length and length > 0 else 14
    lensig = lensig if lensig and lensig > 0 else length
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ADXR as taADXR

        adxr_series = Series(
            taADXR(high, low, close, timeperiod=length), index=close.index
        )
        # Also compute DMP/DMN via native adx for the DataFrame
        adx_df = adx(
            high,
            low,
            close,
            length=length,
            lensig=lensig,
            scalar=scalar,
            mamode=mamode,
            drift=drift,
            talib=False,
            **kwargs,
        )
        if adx_df is None:
            return None
        dmp = adx_df.iloc[:, 1]
        dmn = adx_df.iloc[:, 2]
    else:
        adx_df = adx(
            high,
            low,
            close,
            length=length,
            lensig=lensig,
            scalar=scalar,
            mamode=mamode,
            drift=drift,
            talib=False,
            **kwargs,
        )
        if adx_df is None:
            return None
        adx_series = adx_df.iloc[:, 0]
        dmp = adx_df.iloc[:, 1]
        dmn = adx_df.iloc[:, 2]
        adxr_series = (adx_series + adx_series.shift(length - 1)) / 2

    # Offset, Name and Categorize it
    return _build_dataframe(
        {
            f"ADXR_{lensig}": adxr_series,
            f"DMP_{length}": dmp,
            f"DMN_{length}": dmn,
        },
        f"ADXR_{lensig}",
        "trend",
        offset,
        **kwargs,
    )


adxr.__doc__ = """Average Directional Movement Index Rating (ADXR)

ADXR is the average of the current ADX and the ADX from one period
(length) ago.  It smooths the ADX and is used to confirm trend strength.

Sources:
    https://www.investopedia.com/terms/a/adxr.asp
    TA Lib

Calculation:
    Default Inputs:
        length=14
    ADX = Average Directional Index
    ADXR = (ADX + ADX.shift(length)) / 2

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    lensig (int): Signal Length. Default: length
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ``help(ta.ma)``. Default: 'rma'
    drift (int): The difference period. Default: 1
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: adxr, dmp, dmn columns.
"""
