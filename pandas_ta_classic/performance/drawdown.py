# -*- coding: utf-8 -*-
# Drawdown (DRAWDOWN)
from typing import Any, Optional
from numpy import log as nplog
from numpy import seterr
from pandas import DataFrame, Series
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


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

    # Offset + Name + Category + DataFrame
    return _build_dataframe(
        {"DD": dd, "DD_PCT": dd_pct, "DD_LOG": dd_log},
        "DD",
        "performance",
        offset,
        **kwargs,
    )


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
