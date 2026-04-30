# -*- coding: utf-8 -*-
# Forecast Oscillator (FOSC)
from typing import Any, Optional

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import get_offset, verify_series


def fosc(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Forecast Oscillator (FOSC)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    # Forecast = linear regression value at last bar + 1 step ahead (TSF)
    forecast = linreg(close, length=length, tsf=True)
    if forecast is None:
        return None

    fosc_ = 100 * (close - forecast) / close

    # Offset
    if offset != 0:
        fosc_ = fosc_.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        fosc_.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        if "fill_method" in kwargs:
            if kwargs["fill_method"] == "ffill":
                fosc_.ffill(inplace=True)
            elif kwargs["fill_method"] == "bfill":
                fosc_.bfill(inplace=True)

    # Name and Categorize it
    fosc_.name = f"FOSC_{length}"
    fosc_.category = "momentum"

    return fosc_


fosc.__doc__ = """Forecast Oscillator (FOSC)

The Forecast Oscillator computes the percentage difference between the actual
close price and the Time Series Forecast (linear regression projected one step
ahead). Positive values suggest price above forecast (bullish), negative
values suggest price below forecast (bearish).

FOSC = 100 * (Close - TSF[n]) / Close

Where TSF = Time Series Forecast (Linear Regression + 1 step ahead projection)

Sources:
    Tushar Chande, "The New Technical Trader", 1994
    https://library.tradingtechnologies.com/trade/chrt-ti-forecast-osc.html

Args:
    close (pd.Series): Close price series.
    length (int): Lookback period. Default: 14
    offset (int): Result offset. Default: 0

Returns:
    pd.Series: FOSC values.
"""
