# -*- coding: utf-8 -*-
# Log Return (LOG_RETURN)
from typing import Any, Optional
from numpy import log as nplog
from pandas import Series
from pandas_ta_classic.utils import apply_offset, get_offset, verify_series


def log_return(
    close: Series,
    length: Optional[int] = None,
    cumulative: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Log Return"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 1
    cumulative = bool(cumulative) if cumulative is not None and cumulative else False
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    if cumulative:
        # log_return = nplog(close).diff(length).cumsum()
        log_return = nplog(close / close.iloc[0])
    else:
        log_return = nplog(close / close.shift(length))  # nplog(close).diff(length)

    # Offset
    log_return = apply_offset(log_return, offset, **kwargs)

    # Name & Category
    log_return.name = f"{'CUM' if cumulative else ''}LOGRET_{length}"
    log_return.category = "performance"

    return log_return


log_return.__doc__ = """Log Return

Calculates the logarithmic return of a Series.
See also: help(df.ta.log_return) for additional **kwargs a valid 'df'.

Sources:
    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

Calculation:
    Default Inputs:
        length=1, cumulative=False
    LOGRET = log( close.diff(periods=length) )
    CUMLOGRET = LOGRET.cumsum() if cumulative

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    cumulative (bool): If True, returns the cumulative returns. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
