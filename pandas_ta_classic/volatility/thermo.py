# -*- coding: utf-8 -*-
# Elder Thermometer (THERMO)
from typing import Any, Optional
from pandas import DataFrame, Series
from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import apply_offset, get_drift, get_offset, verify_series


def thermo(
    high: Series,
    low: Series,
    length: Optional[int] = None,
    long: Optional[float] = None,
    short: Optional[float] = None,
    mamode: Optional[str] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Elders Thermometer (THERMO)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    long = float(long) if long and long > 0 else 2
    short = float(short) if short and short > 0 else 0.5
    mamode = mamode if isinstance(mamode, str) else "ema"
    high = verify_series(high, length)
    low = verify_series(low, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    asint = kwargs.pop("asint", True)

    if high is None or low is None:
        return None

    # Calculate Result
    thermoL = (low.shift(drift) - low).abs()
    thermoH = (high - high.shift(drift)).abs()

    thermo = thermoL
    thermo = thermo.where(thermoH < thermoL, thermoH)
    thermo.index = high.index

    thermo_ma = ma(mamode, thermo, length=length)
    if thermo_ma is None:
        return None

    # Create signals
    thermo_long = thermo < (thermo_ma * long)
    thermo_short = thermo > (thermo_ma * short)

    # Binary output, useful for signals
    if asint:
        thermo_long = thermo_long.astype(int)
        thermo_short = thermo_short.astype(int)

    # Offset
    thermo = apply_offset(thermo, offset, **kwargs)
    thermo_ma = apply_offset(thermo_ma, offset, **kwargs)
    thermo_long = apply_offset(thermo_long, offset, **kwargs)
    thermo_short = apply_offset(thermo_short, offset, **kwargs)

    # Name and Categorize it
    _props = f"_{length}_{long}_{short}"
    thermo.name = f"THERMO{_props}"
    thermo_ma.name = f"THERMOma{_props}"
    thermo_long.name = f"THERMOl{_props}"
    thermo_short.name = f"THERMOs{_props}"

    thermo.category = thermo_ma.category = thermo_long.category = (
        thermo_short.category
    ) = "volatility"

    # Prepare Dataframe to return
    data = {
        thermo.name: thermo,
        thermo_ma.name: thermo_ma,
        thermo_long.name: thermo_long,
        thermo_short.name: thermo_short,
    }
    df = DataFrame(data)
    df.name = f"THERMO{_props}"
    df.category = thermo.category

    return df


thermo.__doc__ = """Elders Thermometer (THERMO)

Elder's Thermometer measures price volatility.

Sources:
    https://www.motivewave.com/studies/elders_thermometer.htm
    https://www.tradingview.com/script/HqvTuEMW-Elder-s-Market-Thermometer-LazyBear/

Calculation:
    Default Inputs:
    length=20, drift=1, mamode=EMA, long=2, short=0.5
    EMA = Exponential Moving Average

    thermoL = (low.shift(drift) - low).abs()
    thermoH = (high - high.shift(drift)).abs()

    thermo = np.where(thermoH > thermoL, thermoH, thermoL)
    thermo_ma = ema(thermo, length)

    thermo_long = thermo < (thermo_ma * long)
    thermo_short = thermo > (thermo_ma * short)
    thermo_long = thermo_long.astype(int)
    thermo_short = thermo_short.astype(int)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    long(int): The buy factor
    short(float): The sell factor
    length (int): The  period. Default: 20
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    drift (int): The diff period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: thermo, thermo_ma, thermo_long, thermo_short columns.
"""
