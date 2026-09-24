# Trend Signals (TSIGNALS)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.utils import (
    apply_fill,
    apply_offset,
    get_offset,
    verify_series,
)
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


def _reject_trend_reset(indicator: str, kwargs: dict) -> None:
    """trend_reset was documented but never read; it is removed (AGENTS.md rule 11 exception).

    **kwargs would otherwise swallow it silently.
    """
    if "trend_reset" in kwargs:
        raise TypeError(f"{indicator}() no longer accepts 'trend_reset': it never had an effect; remove the argument")


@nan_on_short_input
def tsignals(
    trend: Series,
    asbool: bool | None = None,
    *,
    trade_offset: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Trend Signals"""
    # Validate Arguments
    trend = verify_series(trend)
    if trend is None:
        return None

    asbool = _bool_param(asbool, False, "asbool")
    _reject_trend_reset("tsignals", kwargs)
    # a negative shift would move entries/exits onto earlier bars (look-ahead)
    trade_offset = _pos_int(trade_offset, 0, "trade_offset", gt=None, ge=0)
    offset = get_offset(offset)

    # Calculate Result
    trends = trend.fillna(0).astype(int)
    trades = trends.diff().shift(trade_offset).fillna(0).astype(int)
    entries = (trades > 0).astype(int)
    exits = (trades < 0).abs().astype(int)

    if asbool:
        trends = trends.astype(bool)
        entries = entries.astype(bool)
        exits = exits.astype(bool)

    data = {
        "TS_Trends": trends,
        "TS_Trades": trades,
        "TS_Entries": entries,
        "TS_Exits": exits,
    }
    df = DataFrame(data, index=trends.index)

    # Offset
    df = apply_offset(df, offset)

    df = apply_fill(df, **kwargs)

    # Name & Category
    df.name = "TS"
    df.category = "trend"

    return df


tsignals.__doc__ = """Trend Signals

Given a Trend, Trend Signals returns the Trend, Trades, Entries and Exits as
boolean integers. When 'asbool=True', it returns Trends, Entries and Exits as
boolean values which is helpful when combined with the vectorbt backtesting
package.

A Trend can be a simple as: 'close' > 'moving average' or something more complex
whose values are boolean or integers (0 or 1).

Examples:
ta.tsignals(close > ta.sma(close, 50), asbool=False)
ta.tsignals(ta.ema(close, 8) > ta.ema(close, 21), asbool=True)

Source: Kevin Johnson

Calculation:
    Default Inputs:
        asbool=False, trade_offset=0

    trades = trends.diff().shift(trade_offset).fillna(0).astype(int)
    entries = (trades > 0).astype(int)
    exits = (trades < 0).abs().astype(int)

Args:
    trend (pd.Series): Series of 'trend's. The trend can be either a boolean or
        integer series of '0's and '1's
    asbool (bool): If True, it converts the Trends, Entries and Exits columns to
        booleans. When boolean, it is also useful for backtesting with
        vectorbt's Portfolio.from_signal(close, entries, exits) Default: False
    trade_offset (value): Value used shift the trade entries/exits Use 1 for
        backtesting and 0 for live. Default: 0
    Note: ``trend_reset`` was removed; it never had an effect, and passing it
    raises TypeError. ``trade_offset`` and ``offset`` are keyword-only.
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame with columns:
    Trends (trend: 1, no trend: 0), Trades (Enter: 1, Exit: -1, Otherwise: 0),
    Entries (entry: 1, nothing: 0), Exits (exit: 1, nothing: 0)
"""
