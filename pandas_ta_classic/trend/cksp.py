# -*- coding: utf-8 -*-
# Chande Kroll Stop (CKSP)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.volatility import atr
from pandas_ta_classic.utils import _build_dataframe, get_offset, verify_series


def cksp(
    high: Series,
    low: Series,
    close: Series,
    p: Optional[int] = None,
    x: Optional[float] = None,
    q: Optional[int] = None,
    tvmode: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Chande Kroll Stop (CKSP)"""
    # Validate Arguments
    # TV defaults=(10,1,9), book defaults = (10,3,20)
    p = int(p) if p and p > 0 else 10
    x = float(x) if x and x > 0 else 1 if tvmode is True else 3
    q = int(q) if q and q > 0 else 9 if tvmode is True else 20
    _length = max(p, q, x)

    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    if high is None or low is None or close is None:
        return None

    offset = get_offset(offset)
    tvmode = tvmode if isinstance(tvmode, bool) else True
    mamode = "rma" if tvmode else "sma"

    # Calculate Result
    atr_ = atr(high=high, low=low, close=close, length=p, mamode=mamode)

    long_stop_ = high.rolling(p).max() - x * atr_
    long_stop = long_stop_.rolling(q).max()

    short_stop_ = low.rolling(p).min() + x * atr_
    short_stop = short_stop_.rolling(q).min()

    # Offset, Name and Categorize it
    _props = f"_{p}_{x}_{q}"
    return _build_dataframe(
        {f"CKSPl{_props}": long_stop, f"CKSPs{_props}": short_stop},
        f"CKSP{_props}",
        "trend",
        offset,
        **kwargs,
    )


cksp.__doc__ = """Chande Kroll Stop (CKSP)

The Tushar Chande and Stanley Kroll in their book
“The New Technical Trader”. It is a trend-following indicator,
identifying your stop by calculating the average true range of
the recent market volatility. The indicator defaults to the implementation
found on tradingview but it provides the original book implementation as well,
which differs by the default periods and moving average mode. While the trading
view implementation uses the Welles Wilder moving average, the book uses a
simple moving average.

Sources:
    https://www.multicharts.com/discussion/viewtopic.php?t=48914
    "The New Technical Trader", Wikey 1st ed. ISBN 9780471597803, page 95

Calculation:
    Default Inputs:
        p=10, x=1, q=9, tvmode=True
    ATR = Average True Range

    LS0 = high.rolling(p).max() - x * ATR(length=p)
    LS = LS0.rolling(q).max()

    SS0 = high.rolling(p).min() + x * ATR(length=p)
    SS = SS0.rolling(q).min()

Args:
    close (pd.Series): Series of 'close's
    p (int): ATR and first stop period. Default: 10 in both modes
    x (float): ATR scalar. Default: 1 in TV mode, 3 otherwise
    q (int): Second stop period. Default: 9 in TV mode, 20 otherwise
    tvmode (bool): Trading View or book implementation mode. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: long and short columns.
"""
