# -*- coding: utf-8 -*-
# Beta (BETA)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic import Imports
from pandas_ta_classic.utils import get_offset, verify_series


def beta(
    close: Series,
    benchmark: Optional[Series] = None,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Beta"""
    # Validate Arguments
    length = int(length) if length and length > 1 else 30
    min_periods = (
        int(kwargs["min_periods"])
        if "min_periods" in kwargs and kwargs["min_periods"] is not None
        else length
    )
    close = verify_series(close, max(length, min_periods))
    benchmark = verify_series(benchmark, max(length, min_periods))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None or benchmark is None:
        return None

    # Calculate Result
    # Beta uses returns (pct_change), not raw prices.
    # Standard financial beta = Cov(close_ret, bench_ret) / Var(bench_ret)
    close_ret = close / close.shift(1) - 1
    bench_ret = benchmark / benchmark.shift(1) - 1

    if Imports["talib"] and mode_tal:
        from talib import BETA as taBETA

        # TA-Lib BETA(A, B) = Cov(A_ret, B_ret) / Var(A_ret), so the
        # benchmark (independent variable) must be passed as the first arg.
        result = Series(taBETA(benchmark, close, timeperiod=length), index=close.index)
    else:
        cov = close_ret.rolling(length, min_periods=min_periods).cov(bench_ret)
        var = bench_ret.rolling(length, min_periods=min_periods).var()
        result = cov / var

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            result.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            result.bfill(inplace=True)

    # Name and Categorize it
    result.name = f"BETA_{length}"
    result.category = "statistics"

    return result


beta.__doc__ = """Beta (BETA)

Beta measures the sensitivity of a security's returns to the returns of a
benchmark.  A beta of 1 means the security moves with the benchmark;
above 1 means more volatile, below 1 means less volatile.

Sources:
    https://www.investopedia.com/terms/b/beta.asp

Calculation:
    Default Inputs:
        length=30
    BETA = COV(close, benchmark, length) / VAR(benchmark, length)

Args:
    close (pd.Series): Series of 'close's
    benchmark (pd.Series): Series of benchmark 'close's
    length (int): It's period. Default: 30 (TA-Lib default: 5)
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
