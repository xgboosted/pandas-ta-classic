# -*- coding: utf-8 -*-
# Historical Volatility – annualised (tulipy: VOLATILITY)
from typing import Any, Optional

from numpy import log as npLog, sqrt as npSqrt
from pandas import Series

from pandas_ta_classic.utils import get_offset, verify_series


def avolume(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Annualised Historical Volatility (tulipy: VOLATILITY)

    Standard deviation of logarithmic returns, scaled by sqrt(252) to
    annualise.  Uses population standard deviation (ddof=0), matching
    tulipy's implementation.  Values are fractions (e.g. 0.09 = 9 % p.a.),
    unlike ta.hvol which returns percentages.

    tulipy name: VOLATILITY.
    """
    length = int(length) if length and length > 0 else 20
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    log_returns = npLog(close / close.shift(1))
    result = log_returns.rolling(length).std(ddof=0) * npSqrt(252)

    if offset != 0:
        result = result.shift(offset)

    result.name = f"AVOLUME_{length}"
    result.category = "volatility"
    return result


avolume.__doc__ = """Annualised Historical Volatility (VOLATILITY)

Population standard deviation of log-returns annualised by sqrt(252).

Formula:
    log_ret = log(close / prev_close)
    VOLATILITY = std(log_ret[-length:], ddof=0) * sqrt(252)

Equivalent to tulipy.volatility(close, period=length).
See ta.hvol for the percentage version using sample std (ddof=1).

tulipy name: VOLATILITY.

Args:
    close (pd.Series): Series of 'close' prices
    length (int): Lookback period. Default: 20
    offset (int): Periods to offset. Default: 0

Returns:
    pd.Series
"""
