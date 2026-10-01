# Elder Thermometer (THERMO)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.ma import ma
from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_drift,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _pos_float, _pos_int, _str_param, nan_on_short_input


@nan_on_short_input
def thermo(
    high: Series,
    low: Series,
    length: int | None = None,
    long: float | None = None,
    short: float | None = None,
    mamode: str | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Elders Thermometer (THERMO)"""
    # Validate arguments
    length = _pos_int(length, 20, "length")
    long = _pos_float(long, 2, "long")
    short = _pos_float(short, 0.5, "short")
    mamode = _str_param(mamode, "ema", "mamode")
    high = verify_series(high, length)
    low = verify_series(low, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    # ``asint`` is absorbed for strategy-broadcast compatibility.  The signal
    # columns are always float now (NaN must survive on the warm-up bars), so
    # its value no longer changes the dtype.
    _bool_param(kwargs.pop("asint", None), True, "asint")

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

    # Both comparisons are False against a NaN average, so the signals would read
    # 0 on the warm-up bars.  Mask them where the thermometer or its average is
    # not defined yet.
    defined = thermo.notna() & thermo_ma.notna()
    thermo_long = thermo_long.where(defined)
    thermo_short = thermo_short.where(defined)

    # Binary output, useful for signals.  float in both branches: NaN must
    # survive on the warm-up bars, so the flags cannot stay bool (``asint=False``
    # used to leave them object dtype) or int.  ``asint`` is still accepted but
    # no longer changes the dtype.
    thermo_long = thermo_long.astype(float)
    thermo_short = thermo_short.astype(float)

    # Offset
    thermo, thermo_ma, thermo_long, thermo_short = apply_offset([thermo, thermo_ma, thermo_long, thermo_short], offset)

    thermo, thermo_ma, thermo_long, thermo_short = apply_fill([thermo, thermo_ma, thermo_long, thermo_short], **kwargs)

    # Name and Categorize it
    _props = f"_{length}_{long}_{short}"
    thermo.name = f"THERMO{_props}"
    thermo_ma.name = f"THERMOma{_props}"
    thermo_long.name = f"THERMOl{_props}"
    thermo_short.name = f"THERMOs{_props}"

    thermo.category = thermo_ma.category = thermo_long.category = thermo_short.category = "volatility"

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
    thermo_long = thermo_long.where(defined).astype(float)
    thermo_short = thermo_short.where(defined).astype(float)

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
        The two signal columns are float (0.0/1.0/NaN) regardless of ``asint``:
        NaN must survive until thermo_ma exists.
"""
