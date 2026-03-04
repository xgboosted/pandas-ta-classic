# -*- coding: utf-8 -*-
# Detrend Price Oscillator (DPO)
from typing import Any, Optional
from pandas import Series
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import _finalize, get_offset, verify_series


def dpo(
    close: Series,
    length: Optional[int] = None,
    centered: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Detrend Price Oscillator (DPO)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 20
    close = verify_series(close, length)
    offset = get_offset(offset)
    if not kwargs.get("lookahead", True):
        centered = False

    if close is None:
        return None

    # Calculate Result
    t = int(0.5 * length) + 1
    ma = sma(close, length)

    dpo = close - ma.shift(t)
    if centered:
        dpo = (close.shift(t) - ma).shift(-t)

    return _finalize(dpo, offset, f"DPO_{length}", "trend", **kwargs)


dpo.__doc__ = """Detrend Price Oscillator (DPO)

Is an indicator designed to remove trend from price and make it easier to
identify cycles.

Sources:
    https://www.tradingview.com/scripts/detrendedpriceoscillator/
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/dpo
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

Calculation:
    Default Inputs:
        length=20, centered=True
    SMA = Simple Moving Average
    t = int(0.5 * length) + 1

    DPO = close.shift(t) - SMA(close, length)
    if centered:
        DPO = DPO.shift(-t)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    centered (bool): Shift the dpo back by int(0.5 * length) + 1. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
