# -*- coding: utf-8 -*-
# Pearson Correlation Coefficient (CORREL)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import (
    _get_tal_mode,
    _get_min_periods,
    _finalize,
    get_offset,
    verify_series,
)


def correl(
    close: Series,
    benchmark: Optional[Series] = None,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Pearson Correlation Coefficient"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    min_periods = _get_min_periods(kwargs, length)
    close = verify_series(close, max(length, min_periods))
    benchmark = verify_series(benchmark, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = _get_tal_mode(talib)

    if close is None or benchmark is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import CORREL as taCORREL

        result = Series(
            taCORREL(close, benchmark, timeperiod=length), index=close.index
        )
    else:
        result = close.rolling(length, min_periods=min_periods).corr(benchmark)

    return _finalize(result, offset, f"CORREL_{length}", "statistics", **kwargs)


correl.__doc__ = """Pearson Correlation Coefficient (CORREL)

The Pearson Correlation Coefficient measures the linear relationship between
two series over a rolling window.  Values range from -1 (perfect negative
correlation) to +1 (perfect positive correlation).

Sources:
    https://www.investopedia.com/terms/c/correlationcoefficient.asp

Calculation:
    Default Inputs:
        length=30
    CORREL = close.rolling(length).corr(benchmark)

Args:
    close (pd.Series): Series of 'close's
    benchmark (pd.Series): Series of benchmark 'close's
    length (int): It's period. Default: 30
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
