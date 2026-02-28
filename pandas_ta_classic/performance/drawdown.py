# -*- coding: utf-8 -*-
# Drawdown (DRAWDOWN)
from typing import Any, Optional
from numpy import log as nplog
from numpy import seterr
from pandas import DataFrame, Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def drawdown(
    close: Series, offset: Optional[int] = None, **kwargs: Any
) -> Optional[DataFrame]:
    """Indicator: Drawdown (DD)"""
    # Validate Arguments
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    max_close = close.cummax()
    dd = max_close - close
    dd_pct = 1 - (close / max_close)

    _np_err = seterr()
    seterr(divide="ignore", invalid="ignore")
    dd_log = nplog(max_close) - nplog(close)
    seterr(divide=_np_err["divide"], invalid=_np_err["invalid"])

    # Offset
    dd = apply_offset(dd, offset, **kwargs)
    dd_pct = apply_offset(dd_pct, offset, **kwargs)
    dd_log = apply_offset(dd_log, offset, **kwargs)

    # Name and Categorize it
    dd.name = "DD"
    dd_pct.name = f"{dd.name}_PCT"
    dd_log.name = f"{dd.name}_LOG"
    dd.category = dd_pct.category = dd_log.category = "performance"

    # Prepare DataFrame to return
    data = {dd.name: dd, dd_pct.name: dd_pct, dd_log.name: dd_log}
    df = DataFrame(data)
    df.name = dd.name
    df.category = dd.category

    return df


drawdown.__doc__ = """Drawdown (DD)

Drawdown is a peak-to-trough decline during a specific period for an investment,
trading account, or fund. It is usually quoted as the percentage between the
peak and the subsequent trough.

Sources:
    https://www.investopedia.com/terms/d/drawdown.asp

Calculation:
    PEAKDD = close.cummax()
    DD = PEAKDD - close
    DD% = 1 - (close / PEAKDD)
    DDlog = log(PEAKDD / close)

Args:
    close (pd.Series): Series of 'close's.
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: drawdown, drawdown percent, drawdown log columns
"""
