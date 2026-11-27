# Historical Volatility (HVOL)
from typing import Annotated, Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import Bound, _name_tokens, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("close")
def hvol(
    close: Series,
    length: Annotated[int, Bound(gt=1)] = 20,
    annualization: Annotated[float, Bound(gt=0)] = 252,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Historical Volatility (HVOL)"""
    # Validate Arguments
    # Annualization factor: 252 for daily, 52 for weekly, 12 for monthly
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
    hvol_.name = f"HVOL_{length}" + _name_tokens(("annualization", annualization, 252))
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
    length (int): Lookback period for std dev, at least 2. Default: 20
    annualization (float): Annualization factor. Default: 252 (trading days/year)
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: HVOL values (annualized %).
"""
