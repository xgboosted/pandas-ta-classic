# -*- coding: utf-8 -*-
# Donchian Channels (DONCHIAN)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.utils import (
    _get_min_periods,
    _build_dataframe,
    get_offset,
    verify_series,
)


def donchian(
    high: Series,
    low: Series,
    lower_length: Optional[int] = None,
    upper_length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Donchian Channels (DC)"""
    # Validate arguments
    lower_length = int(lower_length) if lower_length and lower_length > 0 else 20
    upper_length = int(upper_length) if upper_length and upper_length > 0 else 20
    lower_min_periods = _get_min_periods(kwargs, lower_length, "lower_min_periods")
    upper_min_periods = _get_min_periods(kwargs, upper_length, "upper_min_periods")
    _length = max(lower_length, lower_min_periods, upper_length, upper_min_periods)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    lower = low.rolling(lower_length, min_periods=lower_min_periods).min()
    upper = high.rolling(upper_length, min_periods=upper_min_periods).max()
    mid = 0.5 * (lower + upper)

    _props = f"_{lower_length}_{upper_length}"
    return _build_dataframe(
        {f"DCL{_props}": lower, f"DCM{_props}": mid, f"DCU{_props}": upper},
        f"DC{_props}",
        "volatility",
        offset,
        **kwargs,
    )


donchian.__doc__ = """Donchian Channels (DC)

Donchian Channels are used to measure volatility, similar to
Bollinger Bands and Keltner Channels.

Sources:
    https://www.tradingview.com/wiki/Donchian_Channels_(DC)

Calculation:
    Default Inputs:
        lower_length=upper_length=20
    LOWER = low.rolling(lower_length).min()
    UPPER = high.rolling(upper_length).max()
    MID = 0.5 * (LOWER + UPPER)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    lower_length (int): The short period. Default: 20
    upper_length (int): The short period. Default: 20
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, mid, upper columns.
"""
