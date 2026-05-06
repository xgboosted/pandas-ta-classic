# -*- coding: utf-8 -*-
# Average Directional Movement Index Rating (ADXR)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.trend.adx import adx
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def adxr(
    high: Series,
    low: Series,
    close: Series,
    length: Optional[int] = None,
    lensig: Optional[int] = None,
    scalar: Optional[float] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
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

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    adx_df = adx(
        high,
        low,
        close,
        length=length,
        lensig=lensig,
        scalar=scalar,
        mamode=mamode,
        drift=drift,
    )
    if adx_df is None:
        return None

    adx_series = adx_df.iloc[:, 0]
    dmp = adx_df.iloc[:, 1]
    dmn = adx_df.iloc[:, 2]

    adxr_series = (adx_series + adx_series.shift(length - 1)) / 2

    # Offset
    adxr_series, dmp, dmn = apply_offset([adxr_series, dmp, dmn], offset)

    # Handle fills
    adxr_series, dmp, dmn = apply_fill([adxr_series, dmp, dmn], **kwargs)

    # Name and Categorize it
    adxr_series.name = f"ADXR_{lensig}"
    dmp.name = f"DMP_{length}"
    dmn.name = f"DMN_{length}"

    data = {adxr_series.name: adxr_series, dmp.name: dmp, dmn.name: dmn}
    df = DataFrame(data)
    df.name = f"ADXR_{lensig}"
    df.category = "trend"

    return df


adxr.__doc__ = """Average Directional Movement Index Rating (ADXR)

ADXR is the average of the current ADX and the ADX from one period
(length - 1) ago. It smooths the ADX and is used to confirm trend strength.

Sources:
    https://www.investopedia.com/terms/a/adxr.asp
    TA Lib

Calculation:
    Default Inputs:
        length=14
    ADX = Average Directional Index
    ADXR = (ADX + ADX.shift(length - 1)) / 2

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 14
    lensig (int): Signal length. Default: length
    scalar (float): How much to magnify. Default: 100
    mamode (str): See ``help(ta.ma)``. Default: 'rma'
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: adxr, dmp, dmn columns.
"""
