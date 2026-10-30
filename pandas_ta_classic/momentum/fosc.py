# Forecast Oscillator (FOSC)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.linreg import linreg
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("close")
def fosc(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Forecast Oscillator (FOSC)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    # The forecast for this bar is the previous bar's TSF (its regression line
    # one step ahead), as in TA-Lib's FOSC and Tulip's fosc. The same bar's TSF
    # made fosc equal cfo (#344).
    forecast = linreg(close, length=length, tsf=True)
    if forecast is None:
        return None
    forecast = forecast.shift(1)

    fosc_ = 100 * (close - forecast) / close

    # Offset
    fosc_ = apply_offset(fosc_, offset)

    fosc_ = apply_fill(fosc_, **kwargs)

    # Name and Categorize it
    fosc_.name = f"FOSC_{length}"
    fosc_.category = "momentum"

    return fosc_


fosc.__doc__ = """Forecast Oscillator (FOSC)

The Forecast Oscillator computes the percentage difference between the actual
close price and the Time Series Forecast (linear regression projected one step
ahead). Positive values suggest price above forecast (bullish), negative
values suggest price below forecast (bearish).

FOSC = 100 * (Close - TSF.shift(1)) / Close

Where TSF = Time Series Forecast (Linear Regression + 1 step ahead projection).
The previous bar's TSF is the forecast for this bar, as in TA-Lib's FOSC and
Tulip's fosc. With the same bar's TSF, fosc equalled cfo.

Sources:
    Tushar Chande, "The New Technical Trader", 1994
    https://library.tradingtechnologies.com/trade/chrt-ti-forecast-osc.html

Args:
    close (pd.Series): Close price series.
    length (int): Lookback period. Default: 14
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: FOSC values.
"""
