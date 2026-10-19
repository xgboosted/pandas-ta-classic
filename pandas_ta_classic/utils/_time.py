from time import perf_counter

from pandas import DataFrame, DatetimeIndex, Timestamp

from ._core import _require_input


def df_year_to_date(df: DataFrame) -> DataFrame:
    """Yields the Year-to-Date (YTD) DataFrame; empty when no row is in the current year.

    It used to return the whole DataFrame in that case.
    """
    _require_input(df, "df_year_to_date", "df")
    in_ytd = df.index >= Timestamp.now().strftime("%Y-01-01")
    return df[in_ytd]


def final_time(stime: float) -> str:
    """Human readable elapsed time. Calculates the final time elasped since
    stime and returns a string with microseconds and seconds."""
    _require_input(stime, "final_time", "stime")
    time_diff = perf_counter() - stime
    return f"{time_diff * 1000:2.4f} ms ({time_diff:2.4f} s)"


TIME_RANGE_UNITS = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")


def total_time(df: DataFrame, tf: str = "years") -> float:
    """Calculates the total time of a DataFrame. Difference of the Last and
    First index. Options: 'months', 'weeks', 'days', 'hours', 'minutes'
    and 'seconds'. Default: 'years'.
    Useful for annualization.

    Raises ValueError for any other unit (it used to return years) or an index
    not sorted in ascending order, and TypeError when the index is not
    datetime-like. A single row, or rows sharing one timestamp, span 0.
    """
    _require_input(df, "total_time", "df")
    if tf not in TIME_RANGE_UNITS:
        raise ValueError(f"total_time() tf must be one of {list(TIME_RANGE_UNITS)}, got {tf!r}")
    if not isinstance(df.index, DatetimeIndex):
        raise TypeError(f"total_time() needs a DatetimeIndex, got {type(df.index).__name__}")
    # A descending index gave a negative span, which the metrics that divide by
    # it turned into NaN or a negative growth rate without an error.
    if not df.index.is_monotonic_increasing:
        raise ValueError("total_time() needs an index sorted in ascending order")
    # Every unit is derived from total_seconds() so sub-day spans are not
    # truncated to zero (a 6.5-hour frame is 0.271 days, not 0).  "years" uses
    # calendar days per year (365.25), not trading days (252): the numerator is
    # calendar time, so mixing in a trading-day divisor overstated elapsed years
    # by ~45%.
    total_seconds = (df.index[-1] - df.index[0]).total_seconds()
    TimeFrame = {
        "years": total_seconds / (365.25 * 86400),
        "months": total_seconds / (30.417 * 86400),
        "weeks": total_seconds / (7 * 86400),
        "days": total_seconds / 86400,
        "hours": total_seconds / 3600,
        "minutes": total_seconds / 60,
        "seconds": total_seconds,
    }
    return TimeFrame[tf]


def to_utc(df: DataFrame) -> DataFrame:
    """Either localizes the DataFrame Index to UTC or it applies
    tz_convert to set the Index to UTC.

    Returns a copy; the caller's DataFrame is left unchanged.
    """
    _require_input(df, "to_utc", "df")
    df = df.copy()
    if not df.empty:
        try:
            df.index = df.index.tz_localize("UTC")
        except TypeError:
            df.index = df.index.tz_convert("UTC")
    return df
