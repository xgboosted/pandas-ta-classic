# -*- coding: utf-8 -*-
# Stochastic Fast (STOCHF)
from typing import Any, Optional

from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    non_zero_range,
    verify_series,
)


def stochf(
    high: Series,
    low: Series,
    close: Series,
    fastk: Optional[int] = None,
    fastd: Optional[int] = None,
    mamode: Optional[str] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Stochastic Fast (STOCHF)"""
    # Validate Arguments
    fastk = fastk if fastk and fastk > 0 else 5
    fastd = fastd if fastd and fastd > 0 else 3
    _length = max(fastk, fastd)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import STOCHF as TASTOCHF

        fastk_, fastd_ = TASTOCHF(high, low, close, fastk, fastd)
    else:
        lowest_low = low.rolling(fastk).min()
        highest_high = high.rolling(fastk).max()

        fastk_ = 100 * (close - lowest_low) / non_zero_range(highest_high, lowest_low)
        fastk_first_valid = fastk_.first_valid_index()
        if fastk_first_valid is None:
            fastd_ = fastk_.copy()
        else:
            fastd_ = ma(mamode, fastk_.loc[fastk_first_valid:,], length=fastd)
            if fastd_ is None:
                return None

    # Offset
    fastk_, fastd_ = apply_offset([fastk_, fastd_], offset)

    fastk_, fastd_ = apply_fill([fastk_, fastd_], **kwargs)

    # Name and Categorize it
    _name = "STOCHF"
    _params = f"_{fastk}_{fastd}"
    fastk_.name = f"{_name}k{_params}"
    fastd_.name = f"{_name}d{_params}"
    fastk_.category = fastd_.category = "momentum"

    data = {fastk_.name: fastk_, fastd_.name: fastd_}
    df = DataFrame(data)
    df.name = f"{_name}{_params}"
    df.category = fastk_.category

    return df


stochf.__doc__ = """Stochastic Fast (STOCHF)

The Stochastic Fast oscillator is a faster variant of the classic Stochastic
oscillator. It uses a shorter smoothing period for %D, making it more
responsive to price changes.

%FastK = 100 * (Close - Lowest Low[n]) / (Highest High[n] - Lowest Low[n])
%FastD = MA(%FastK, fastd_period)

Sources:
    https://www.investopedia.com/terms/s/stochasticoscillator.asp

Args:
    high (pd.Series): High price series.
    low (pd.Series): Low price series.
    close (pd.Series): Close price series.
    fastk (int): Fast %K period. Default: 5
    fastd (int): Fast %D smoothing period. Default: 3
    mamode (str): MA type for %D. Default: 'sma'
    talib (bool): Use TA-Lib if installed. Default: True
    offset (int): Result offset. Default: 0

Returns:
    pd.DataFrame: FastK and FastD columns.
"""
