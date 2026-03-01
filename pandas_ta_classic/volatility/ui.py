# -*- coding: utf-8 -*-
# Ulcer Index (UI)
from typing import Any, Optional
from numpy import sqrt as npsqrt
from pandas import Series
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def ui(
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Ulcer Index (UI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar and scalar > 0 else 100
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    highest_close = close.rolling(length).max()
    downside = scalar * (close - highest_close)
    downside /= highest_close
    d2 = downside * downside

    everget = kwargs.pop("everget", False)
    if everget:
        # Everget uses SMA instead of SUM for calculation
        ui = (sma(d2, length) / length).apply(npsqrt)
    else:
        ui = (d2.rolling(length).sum() / length).apply(npsqrt)

    return _finalize(
        ui, offset, f"UI{'' if not everget else 'e'}_{length}", "volatility", **kwargs
    )


ui.__doc__ = """Ulcer Index (UI)

The Ulcer Index by Peter Martin measures the downside volatility with the use of
the Quadratic Mean, which has the effect of emphasising large drawdowns.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-ulcer-index.html
    https://en.wikipedia.org/wiki/Ulcer_index
    http://www.tangotools.com/ui/ui.htm

Calculation:
    Default Inputs:
        length=14, scalar=100
    HC = Highest Close
    SMA = Simple Moving Average

    HCN = HC(close, length)
    DOWNSIDE = scalar * (close - HCN) / HCN
    if kwargs["everget"]:
        UI = SQRT(SMA(DOWNSIDE^2, length) / length)
    else:
        UI = SQRT(SUM(DOWNSIDE^2, length) / length)

Args:
    high (pd.Series): Series of 'high's
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 14
    scalar (float): A positive float to scale the bands. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
    everget (value, optional): TradingView's Evergets SMA instead of SUM
        calculation. Default: False

Returns:
    pd.Series: New feature
"""
