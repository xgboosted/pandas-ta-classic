# -*- coding: utf-8 -*-
# Rate of Change Ratio * 100 (ROCR100)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import get_offset, verify_series


def rocr100(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Rate of Change Ratio * 100 (ROCR100)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ROCR100 as TAROCR100

        rocr100_ = TAROCR100(close, length)
    else:
        rocr100_ = 100 * (close / close.shift(length))

    # Offset
    if offset != 0:
        rocr100_ = rocr100_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rocr100_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                rocr100_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                rocr100_.bfill(inplace=True)

    # Name and Categorize it
    rocr100_.name = f"ROCR100_{length}"
    rocr100_.category = "momentum"

    return rocr100_


rocr100.__doc__ = """Rate of Change Ratio * 100 (ROCR100)

Rate of Change Ratio * 100 measures the ratio of the current price to the
price n periods ago, scaled by 100. A value of 100 means no change.

ROCR100 = 100 * (close / close[n])

Sources:
    https://www.investopedia.com/terms/r/rateofchange.asp

Args:
    close (pd.Series): Close price series.
    length (int): The period. Default: 10
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: ROCR100 values.
"""
