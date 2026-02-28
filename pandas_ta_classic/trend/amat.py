# -*- coding: utf-8 -*-
# Archer Moving Averages Trends (AMAT)
from typing import Any, Optional
from pandas import DataFrame, Series
from .long_run import long_run
from .short_run import short_run
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import get_offset, verify_series


def amat(
    close: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    lookback: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Archer Moving Averages Trends (AMAT)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 8
    slow = int(slow) if slow and slow > 0 else 21
    lookback = int(lookback) if lookback and lookback > 0 else 2
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, max(fast, slow, lookback))
    offset = get_offset(offset)
    if "length" in kwargs:
        kwargs.pop("length")

    if close is None:
        return None

    # # Calculate Result
    fast_ma = ma(mamode, close, length=fast, **kwargs)
    slow_ma = ma(mamode, close, length=slow, **kwargs)

    mas_long = long_run(fast_ma, slow_ma, length=lookback)
    mas_short = short_run(fast_ma, slow_ma, length=lookback)

    # Offset
    if offset != 0:
        mas_long = mas_long.shift(offset)
        mas_short = mas_short.shift(offset)

    # # Handle fills
    if "fillna" in kwargs:
        mas_long.fillna(kwargs["fillna"], inplace=True)
        mas_short.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                mas_long.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                mas_long.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                mas_short.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                mas_short.bfill(inplace=True)

    # Prepare DataFrame to return
    amatdf = DataFrame(
        {
            f"AMAT{mamode[0]}_LR_{fast}_{slow}_{lookback}": mas_long,
            f"AMAT{mamode[0]}_SR_{fast}_{slow}_{lookback}": mas_short,
        }
    )

    # Name and Categorize it
    amatdf.name = f"AMAT{mamode[0]}_{fast}_{slow}_{lookback}"
    amatdf.category = "trend"

    return amatdf


amat.__doc__ = """Archer Moving Averages Trends (AMAT)

The Archer Moving Averages Trends indicator identifies trend direction by comparing
fast and slow moving averages. It generates long and short run signals based on the
relationship between the two moving averages over a lookback period.

Sources:
    https://www.tradingview.com/script/nhQe8QJ0-Archer-Moving-Averages-Trends/

Calculation:
    Default Inputs:
        fast=8, slow=21, lookback=2, mamode="ema"
    
    FAST_MA = MA(close, fast, mamode)
    SLOW_MA = MA(close, slow, mamode)
    
    AMAT_LR = LONG_RUN(FAST_MA, SLOW_MA, lookback)
    AMAT_SR = SHORT_RUN(FAST_MA, SLOW_MA, lookback)

Args:
    close (pd.Series): Series of 'close's
    fast (int): Fast MA period. Default: 8
    slow (int): Slow MA period. Default: 21
    lookback (int): Lookback period for trend detection. Default: 2
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: AMAT_LR and AMAT_SR columns.
"""
