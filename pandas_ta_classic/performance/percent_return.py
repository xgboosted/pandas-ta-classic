# -*- coding: utf-8 -*-
# Percent Return (PERCENT_RETURN)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def percent_return(
    close: Series,
    length: Optional[int] = None,
    cumulative: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Percent Return"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 1
    cumulative = bool(cumulative) if cumulative is not None and cumulative else False
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    if cumulative:
        pct_return = (close / close.iloc[0]) - 1
    else:
        pct_return = close.pct_change(length)  # (close / close.shift(length)) - 1

    return _finalize(
        pct_return,
        offset,
        f"{'CUM' if cumulative else ''}PCTRET_{length}",
        "performance",
        **kwargs,
    )


percent_return.__doc__ = """Percent Return

Calculates the percent return of a Series.
See also: help(df.ta.percent_return) for additional **kwargs a valid 'df'.

Sources:
    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

Calculation:
    Default Inputs:
        length=1, cumulative=False
    PCTRET = close.pct_change(length)
    CUMPCTRET = PCTRET.cumsum() if cumulative

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
