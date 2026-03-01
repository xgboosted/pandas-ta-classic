# -*- coding: utf-8 -*-
# Directional Movement (DM)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    _build_dataframe,
    get_drift,
    get_offset,
    verify_series,
    zero,
)


def dm(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: DM"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high)
    low = verify_series(low)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None:
        return None

    if Imports["talib"] and mode_tal:
        from talib import MINUS_DM, PLUS_DM

        pos = PLUS_DM(high, low, length)
        neg = MINUS_DM(high, low, length)
    else:
        up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos_ = ((up > dn) & (up > 0)) * up
        neg_ = ((dn > up) & (dn > 0)) * dn

        pos_ = pos_.apply(zero)
        neg_ = neg_.apply(zero)

        # Not the same values as TA Lib's -+DM (Good First Issue)
        pos = ma(mamode, pos_, length=length)
        neg = ma(mamode, neg_, length=length)
        if pos is None or neg is None:
            return None

    # Offset + Name + Category + DataFrame
    _params = f"_{length}"
    return _build_dataframe(
        {f"DMP{_params}": pos, f"DMN{_params}": neg},
        f"DM{_params}",
        "trend",
        offset,
        **kwargs,
    )


dm.__doc__ = """Directional Movement (DM)

The Directional Movement was developed by J. Welles Wilder in 1978 attempts to
determine which direction the price of an asset is moving. It compares prior
highs and lows to yield to two series +DM and -DM.

Sources:
    https://www.tradingview.com/pine-script-reference/#fun_dmi
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=24&Name=Directional_Movement_Index

Calculation:
    Default Inputs:
        length=14, mamode="rma", drift=1
            up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos_ = ((up > dn) & (up > 0)) * up
        neg_ = ((dn > up) & (dn > 0)) * dn

        pos_ = pos_.apply(zero)
        neg_ = neg_.apply(zero)

        # Not the same values as TA Lib's -+DM
        pos = ma(mamode, pos_, length=length)
        neg = ma(mamode, neg_, length=length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    mamode (str): See ```help(ta.ma)```.  Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Returns:
    pd.DataFrame: DMP (+DM) and DMN (-DM) columns.
"""
