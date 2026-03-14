# -*- coding: utf-8 -*-
# Archer On Balance Volume (AOBV)
from typing import Any, Optional
from pandas import DataFrame, Series
from .obv import obv
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.trend import long_run, short_run
from pandas_ta_classic.utils import get_offset, verify_series


def aobv(
    close: Series,
    volume: Series,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    max_lookback: Optional[int] = None,
    min_lookback: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Archer On Balance Volume (AOBV)"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 4
    slow = int(slow) if slow and slow > 0 else 12
    max_lookback = int(max_lookback) if max_lookback and max_lookback > 0 else 2
    min_lookback = int(min_lookback) if min_lookback and min_lookback > 0 else 2
    if slow < fast:
        fast, slow = slow, fast
    mamode = mamode if isinstance(mamode, str) else "ema"
    _length = max(fast, slow, max_lookback, min_lookback)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    offset = get_offset(offset)
    if "length" in kwargs:
        kwargs.pop("length")
    run_length = kwargs.pop("run_length", 2)

    if close is None or volume is None:
        return None

    # Calculate Result
    obv_ = obv(close=close, volume=volume, **kwargs)
    maf = ma(mamode, obv_, length=fast, **kwargs)
    if maf is None:
        return None
    mas = ma(mamode, obv_, length=slow, **kwargs)
    if mas is None:
        return None

    # When MAs are long and short
    obv_long = long_run(maf, mas, length=run_length)
    obv_short = short_run(maf, mas, length=run_length)

    # Offset
    if offset != 0:
        obv_ = obv_.shift(offset)
        maf = maf.shift(offset)
        mas = mas.shift(offset)
        obv_long = obv_long.shift(offset)
        obv_short = obv_short.shift(offset)

    # # Handle fills
    if "fillna" in kwargs:
        obv_.fillna(kwargs["fillna"], inplace=True)
        maf.fillna(kwargs["fillna"], inplace=True)
        mas.fillna(kwargs["fillna"], inplace=True)
        obv_long.fillna(kwargs["fillna"], inplace=True)
        obv_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                obv_.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                obv_.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                maf.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                maf.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                mas.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                mas.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                obv_long.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                obv_long.bfill(inplace=True)
        if "fill_method" in kwargs:

            if kwargs["fill_method"] == "ffill":

                obv_short.ffill(inplace=True)

            elif kwargs["fill_method"] == "bfill":

                obv_short.bfill(inplace=True)

    # Prepare DataFrame to return
    _mode = mamode.lower()[0] if len(mamode) else ""
    data = {
        obv_.name: obv_,
        f"OBV_min_{min_lookback}": obv_.rolling(min_lookback).min(),
        f"OBV_max_{max_lookback}": obv_.rolling(max_lookback).max(),
        f"OBV{_mode}_{fast}": maf,
        f"OBV{_mode}_{slow}": mas,
        f"AOBV_LR_{run_length}": obv_long,
        f"AOBV_SR_{run_length}": obv_short,
    }
    aobvdf = DataFrame(data)

    # Name and Categorize it
    aobvdf.name = (
        f"AOBV{_mode}_{fast}_{slow}_{min_lookback}_{max_lookback}_{run_length}"
    )
    aobvdf.category = "volume"

    return aobvdf


aobv.__doc__ = """Archer On Balance Volume (AOBV)

Archer On Balance Volume enhances the traditional OBV indicator by applying moving
averages and detecting long/short run trends. It provides multiple signals including
OBV with min/max bounds, fast/slow moving averages of OBV, and trend direction signals.

Sources:
    Derived from OBV (On Balance Volume)
    https://www.investopedia.com/terms/o/onbalancevolume.asp

Calculation:
    Default Inputs:
        fast=4, slow=12, max_lookback=2, min_lookback=2, mamode="ema", run_length=2
    
    OBV = On Balance Volume(close, volume)
    OBV_MIN = ROLLING_MIN(OBV, min_lookback)
    OBV_MAX = ROLLING_MAX(OBV, max_lookback)
    FAST_MA = MA(OBV, fast, mamode)
    SLOW_MA = MA(OBV, slow, mamode)
    AOBV_LR = LONG_RUN(FAST_MA, SLOW_MA, run_length)
    AOBV_SR = SHORT_RUN(FAST_MA, SLOW_MA, run_length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    fast (int): Fast MA period. Default: 4
    slow (int): Slow MA period. Default: 12
    max_lookback (int): Max lookback period. Default: 2
    min_lookback (int): Min lookback period. Default: 2
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    run_length (int, optional): Lookback for long/short run. Default: 2
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: OBV, OBV_min, OBV_max, fast MA, slow MA, AOBV_LR, AOBV_SR columns.
"""
