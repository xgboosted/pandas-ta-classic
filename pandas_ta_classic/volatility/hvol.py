# Historical Volatility (HVOL)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_float, _pos_int


def hvol(
    close: Series,
    length: int | None = None,
    annualization: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Historical Volatility (HVOL)"""
    # Validate Arguments
    length = _pos_int(length, 20, "length")
    # Annualization factor: 252 for daily, 52 for weekly, 12 for monthly
    annualization = _pos_float(annualization, 252, "annualization")
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    log_returns = np.log(close / close.shift(1))
    hvol_ = 100 * log_returns.rolling(length).std(ddof=1) * np.sqrt(annualization)

    # Offset
    hvol_ = apply_offset(hvol_, offset)

    hvol_ = apply_fill(hvol_, **kwargs)

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

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: HVOL values (annualized %).
"""
