# -*- coding: utf-8 -*-
# Historical Volatility (HVOL)
from typing import Any, Optional

from numpy import log as npLog, sqrt as npSqrt
from pandas import Series

from pandas_ta_classic.utils import get_offset, verify_series


def hvol(
    close: Series,
    length: Optional[int] = None,
    annualization: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Historical Volatility (HVOL)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 20
    # Annualization factor: 252 for daily, 52 for weekly, 12 for monthly
    annualization = float(annualization) if annualization and annualization > 0 else 252
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    log_returns = npLog(close / close.shift(1))
    hvol_ = 100 * log_returns.rolling(length).std(ddof=1) * npSqrt(annualization)

    # Offset
    if offset != 0:
        hvol_ = hvol_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        hvol_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                hvol_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                hvol_.bfill(inplace=True)

    # Name and Categorize it
    hvol_.name = f"HVOL_{length}"
    hvol_.category = "volatility"

    return hvol_


hvol.__doc__ = """Historical Volatility (HVOL)

Historical Volatility is the annualized standard deviation of logarithmic
daily returns over a given period. It measures how much the price has varied
historically, expressed as an annualized percentage.

log_return = log(close / close[1])
HVOL = 100 * StdDev(log_return, length) * sqrt(annualization)

Sources:
    https://www.investopedia.com/terms/h/historicalvolatility.asp

Args:
    close (pd.Series): Close price series.
    length (int): Lookback period for std dev. Default: 20
    annualization (float): Annualization factor. Default: 252 (trading days/year)
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: HVOL values (annualized %).
"""
