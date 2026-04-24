# -*- coding: utf-8 -*-
# CPR for Option Trading
from typing import Any, Optional

from pandas import DataFrame, Series, concat
from pandas_ta_classic.trend.cpr import cpr
from pandas_ta_classic.utils import get_offset
from pandas_ta_classic.utils._cpr import (
    calculate_option_strikes,
    detect_cpr_breakout,
    detect_cpr_rejection,
)


def cpr_option(
    open: Series,
    high: Series,
    low: Series,
    close: Series,
    volume: Optional[Series] = None,
    method: str = "classic",
    timeframe: str = "daily",
    interval: Optional[str] = None,
    levels: str = "standard",
    strike_round: int = 50,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: CPR for Option Trading"""
    # Get base CPR with virgin CPR detection enabled for option trading
    enable_virgin_cpr = kwargs.pop("virgin_cpr", True)
    cpr_df = cpr(
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        method=method,
        timeframe=timeframe,
        interval=interval,
        levels=levels,
        width_analysis=True,
        price_position=True,
        virgin_cpr=enable_virgin_cpr,
        offset=0,  # Apply offset at end
        **kwargs,
    )

    if cpr_df is None:
        return None

    # Extract CPR levels
    tc = cpr_df["CPR_TC"]
    pivot = cpr_df["CPR_PIVOT"]
    bc = cpr_df["CPR_BC"]

    # Calculate option strikes
    strikes = calculate_option_strikes(tc, pivot, bc, strike_round)

    # Calculate expected range
    volatility_factor = kwargs.pop("volatility_factor", 1.5)
    if "CPR_WIDTH" in cpr_df.columns:
        cpr_width = cpr_df["CPR_WIDTH"]
        expected_range_high = pivot + (cpr_width * volatility_factor)
        expected_range_low = pivot - (cpr_width * volatility_factor)
    else:
        expected_range_high = None
        expected_range_low = None

    # Detect breakout signals
    breakout_lookback = kwargs.pop("breakout_lookback", 1)
    breakout_call, breakout_put = detect_cpr_breakout(
        close, high, low, tc, bc, lookback=breakout_lookback
    )

    # Detect rejection signals
    rejection_threshold = kwargs.pop("rejection_threshold", 0.2)
    rejection_call, rejection_put = detect_cpr_rejection(
        close, high, low, tc, bc, threshold=rejection_threshold
    )

    # Offset
    offset = get_offset(offset)
    if offset != 0:
        cpr_df = cpr_df.shift(offset)
        for key in strikes:
            strikes[key] = strikes[key].shift(offset)
        if expected_range_high is not None:
            expected_range_high = expected_range_high.shift(offset)
            expected_range_low = expected_range_low.shift(offset)
        breakout_call = breakout_call.shift(offset)
        breakout_put = breakout_put.shift(offset)
        rejection_call = rejection_call.shift(offset)
        rejection_put = rejection_put.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        for key in strikes:
            strikes[key].fillna(kwargs["fillna"], inplace=True)
        if expected_range_high is not None:
            expected_range_high.fillna(kwargs["fillna"], inplace=True)
            expected_range_low.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        if kwargs["fill_method"] == "ffill":
            for key in strikes:
                strikes[key].ffill(inplace=True)
            if expected_range_high is not None:
                expected_range_high.ffill(inplace=True)
                expected_range_low.ffill(inplace=True)
        elif kwargs["fill_method"] == "bfill":
            for key in strikes:
                strikes[key].bfill(inplace=True)
            if expected_range_high is not None:
                expected_range_high.bfill(inplace=True)
                expected_range_low.bfill(inplace=True)

    # Name and Categorize
    strikes["tc_strike"].name = "OPT_TC_STRIKE"
    strikes["pivot_strike"].name = "OPT_PIVOT_STRIKE"
    strikes["bc_strike"].name = "OPT_BC_STRIKE"
    strikes["call_otm1"].name = "OPT_CALL_OTM1"
    strikes["call_otm2"].name = "OPT_CALL_OTM2"
    strikes["put_otm1"].name = "OPT_PUT_OTM1"
    strikes["put_otm2"].name = "OPT_PUT_OTM2"

    for key in strikes:
        strikes[key].category = "trend"

    breakout_call.name = "OPT_BREAKOUT_CALL"
    breakout_put.name = "OPT_BREAKOUT_PUT"
    rejection_call.name = "OPT_REJECTION_CALL"
    rejection_put.name = "OPT_REJECTION_PUT"

    breakout_call.category = breakout_put.category = "trend"
    rejection_call.category = rejection_put.category = "trend"

    if expected_range_high is not None:
        expected_range_high.name = "OPT_RANGE_HIGH"
        expected_range_low.name = "OPT_RANGE_LOW"
        expected_range_high.category = expected_range_low.category = "trend"

    # Prepare DataFrame
    # Start with base CPR data
    data = cpr_df.to_dict("series")

    # Add option strikes
    data[strikes["tc_strike"].name] = strikes["tc_strike"]
    data[strikes["pivot_strike"].name] = strikes["pivot_strike"]
    data[strikes["bc_strike"].name] = strikes["bc_strike"]
    data[strikes["call_otm1"].name] = strikes["call_otm1"]
    data[strikes["call_otm2"].name] = strikes["call_otm2"]
    data[strikes["put_otm1"].name] = strikes["put_otm1"]
    data[strikes["put_otm2"].name] = strikes["put_otm2"]

    # Add signals
    data[breakout_call.name] = breakout_call
    data[breakout_put.name] = breakout_put
    data[rejection_call.name] = rejection_call
    data[rejection_put.name] = rejection_put

    # Add expected range
    if expected_range_high is not None:
        data[expected_range_high.name] = expected_range_high
        data[expected_range_low.name] = expected_range_low

    optcprdf = DataFrame(data)
    optcprdf.name = f"CPR_OPTION"
    optcprdf.category = "trend"

    return optcprdf


cpr_option.__doc__ = """CPR for Option Trading

Extends base CPR with option-specific analysis including strike recommendations,
breakout/rejection signals, and expected range calculations.

Sources:
    https://tradingqna.com/what-is-central-pivot-range-cpr-how-to-trade-using-it/

Examples:
    import pandas_ta_classic as ta
    # Basic option CPR
    df.ta.cpr_option(strike_round=50, append=True)

    # With custom parameters for option signals
    df.ta.cpr_option(method='camarilla', strike_round=100,
                     volatility_factor=2.0, breakout_lookback=2, append=True)

    # Find call/put opportunities
    calls = df[df['OPT_BREAKOUT_CALL'] == True]
    puts = df[df['OPT_REJECTION_PUT'] == True]

Calculation:
    Uses base CPR calculation, then adds:

    Strike Recommendations:
        TC_STRIKE = round(TC / strike_round) * strike_round
        PIVOT_STRIKE = round(PIVOT / strike_round) * strike_round
        BC_STRIKE = round(BC / strike_round) * strike_round
        CALL_OTM1 = TC_STRIKE + strike_round
        CALL_OTM2 = TC_STRIKE + (strike_round * 2)
        PUT_OTM1 = BC_STRIKE - strike_round
        PUT_OTM2 = BC_STRIKE - (strike_round * 2)

    Expected Range:
        RANGE_HIGH = PIVOT + (CPR_WIDTH * volatility_factor)
        RANGE_LOW = PIVOT - (CPR_WIDTH * volatility_factor)

    Breakout Signals:
        BREAKOUT_CALL: Close crosses above TC
        BREAKOUT_PUT: Close crosses below BC

    Rejection Signals:
        REJECTION_CALL: Price touches BC then closes above
        REJECTION_PUT: Price touches TC then closes below

Args:
    open (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series, optional): Series of 'volume's
    method (str): Pivot calculation method.
        Options: 'classic', 'camarilla', 'fibonacci', 'woodie'. Default: 'classic'
    timeframe (str): Time context for CPR calculation.
        Options: 'intraday', 'daily', 'weekly', 'monthly'. Default: 'daily'
    interval (str, optional): For intraday only - data interval.
        Examples: '1min', '5min', '15min', '30min', '1H'. Default: None
    levels (str): Which pivot levels to calculate.
        Options: 'basic', 'standard', 'extended', 'all'. Default: 'standard'
    strike_round (int): Round strikes to nearest value. Default: 50
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    volatility_factor (float): Multiplier for expected range calculation. Default: 1.5
    breakout_lookback (int): Periods to look back for breakout detection. Default: 1
    rejection_threshold (float): Percentage tolerance for rejection detection. Default: 0.2
    virgin_cpr (bool): Enable virgin CPR detection. Default: True
    virgin_lookforward (int): Periods to look forward for virgin CPR detection. Default: 5
    width_narrow (float): Threshold for narrow CPR classification (%). Default: 0.5
    width_wide (float): Threshold for wide CPR classification (%). Default: 1.5
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Base CPR columns plus option-specific columns.
        Base CPR columns: CPR_TC, CPR_PIVOT, CPR_BC, CPR_R1, CPR_S1, etc.
        Strike columns: OPT_TC_STRIKE, OPT_PIVOT_STRIKE, OPT_BC_STRIKE,
                       OPT_CALL_OTM1, OPT_CALL_OTM2, OPT_PUT_OTM1, OPT_PUT_OTM2
        Range columns: OPT_RANGE_HIGH, OPT_RANGE_LOW
        Signal columns: OPT_BREAKOUT_CALL, OPT_BREAKOUT_PUT,
                       OPT_REJECTION_CALL, OPT_REJECTION_PUT
        Analysis columns: CPR_WIDTH, CPR_WIDTH_PCT, CPR_WIDTH_CLASS, CPR_POSITION
"""
