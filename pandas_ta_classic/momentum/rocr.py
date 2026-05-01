# -*- coding: utf-8 -*-
# Rate of Change Ratio (ROCR)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def rocr(
    close: Series,
    length: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Rate of Change Ratio (ROCR)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ROCR as TAROCR

        rocr_ = TAROCR(close, length)
    else:
        rocr_ = close / close.shift(length)

    # Offset
    rocr_ = apply_offset(rocr_, offset)

    rocr_ = apply_fill(rocr_, **kwargs)

    # Name and Categorize it
    rocr_.name = f"ROCR_{length}"
    rocr_.category = "momentum"

    return rocr_


rocr.__doc__ = """Rate of Change Ratio (ROCR)

Rate of Change Ratio measures the ratio of the current price to the price
n periods ago.

ROCR = close / close[n]

Sources:
    https://www.investopedia.com/terms/r/rateofchange.asp

Args:
    close (pd.Series): Close price series.
    length (int): The period. Default: 10
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: ROCR values.
"""
