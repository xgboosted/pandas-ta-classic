import logging
from typing import Any, Optional

from pandas import DataFrame, Series

from ._core import apply_offset, get_offset, verify_series
from ._math import zero

logger = logging.getLogger(__name__)


def _above_below(
    series_a: Series,
    series_b: Series,
    above: bool = True,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)

    if series_a is None or series_b is None:
        return None

    offset = get_offset(offset)

    series_a = series_a.apply(zero)
    series_b = series_b.apply(zero)

    # Calculate Result
    current = series_a >= series_b if above else series_a <= series_b

    if asint:
        current = current.astype(int)

    # Offset
    current = apply_offset(current, offset)

    # Name & Category
    current.name = f"{series_a.name}_{'A' if above else 'B'}_{series_b.name}"
    current.category = "utility"

    return current


def above(
    series_a: Series,
    series_b: Series,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Series:
    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)


def above_value(
    series_a: Series,
    value: float,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    if not isinstance(value, (int, float, complex)):
        logger.error("value is not a number")
        return None
    series_a = verify_series(series_a)
    if series_a is None:
        return None
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)


def below(
    series_a: Series,
    series_b: Series,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Series:
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)


def below_value(
    series_a: Series,
    value: float,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    if not isinstance(value, (int, float, complex)):
        logger.error("value is not a number")
        return None
    series_a = verify_series(series_a)
    if series_a is None:
        return None
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)


def cross_value(
    series_a: Series,
    value: float,
    above: bool = True,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    series_a = verify_series(series_a)
    if series_a is None:
        return None
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return cross(series_a, series_b, above, asint, offset, **kwargs)


def cross(
    series_a: Series,
    series_b: Series,
    above: bool = True,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)

    if series_a is None or series_b is None:
        return None

    offset = get_offset(offset)

    series_a = series_a.apply(zero)
    series_b = series_b.apply(zero)

    # Calculate Result
    current = series_a > series_b  # current is above
    previous = series_a.shift(1) < series_b.shift(1)  # previous is below
    # above if both are true, below if both are false
    cross = current & previous if above else ~current & ~previous

    if asint:
        cross = cross.astype(int)

    # Offset
    cross = apply_offset(cross, offset)

    # Name & Category
    cross.name = f"{series_a.name}_{'XA' if above else 'XB'}_{series_b.name}"
    cross.category = "utility"

    return cross


def _add_scalar_threshold_signals(df, indicator, value, cross_values, use_above, offset):
    """Add scalar-value threshold signals to *df* in-place.

    When *cross_values* is ``True`` two cross columns are added (above=True
    and above=False).  Otherwise a single above/below column is appended.

    Args:
        df (DataFrame): Target frame; mutated in-place.
        indicator (Series): The indicator series.
        value (float): Scalar threshold value.
        cross_values (bool): Emit cross columns rather than a simple level flag.
        use_above (bool): ``True`` → above-value; ``False`` → below-value when
            *cross_values* is ``False``.
        offset (int): Series offset forwarded to the signal helpers.
    """
    if cross_values:
        s_start = cross_value(indicator, value, above=True, offset=offset)
        s_end = cross_value(indicator, value, above=False, offset=offset)
        df[s_start.name] = s_start
        df[s_end.name] = s_end
    elif use_above:
        s = above_value(indicator, value, offset=offset)
        df[s.name] = s
    else:
        s = below_value(indicator, value, offset=offset)
        df[s.name] = s


def _add_series_signals(df, indicator, xserie, cross_series, is_above, offset):
    """Add series-comparison signals to *df* in-place.

    No-op when *xserie* is ``None`` or fails :func:`verify_series`.

    Args:
        df (DataFrame): Target frame; mutated in-place.
        indicator (Series): The indicator series.
        xserie (Series | None): Comparison series.
        cross_series (bool): Emit cross columns rather than above/below flags.
        is_above (bool): Direction of the comparison.
        offset (int): Series offset forwarded to the signal helpers.
    """
    if xserie is not None:
        xserie_v = verify_series(xserie)
        if xserie_v is not None:
            if cross_series:
                s = cross(indicator, xserie_v, above=is_above, offset=offset)
            elif is_above:
                s = above(indicator, xserie_v, offset=offset)
            else:
                s = below(indicator, xserie_v, offset=offset)
            df[s.name] = s


def signals(
    indicator: Series,
    xa: Optional[float],
    xb: Optional[float],
    cross_values: bool,
    xserie: Optional[Series],
    xserie_a: Optional[Series],
    xserie_b: Optional[Series],
    cross_series: bool,
    offset: Optional[int],
) -> DataFrame:
    df = DataFrame()

    if xa is not None and isinstance(xa, (int, float)):
        _add_scalar_threshold_signals(df, indicator, xa, cross_values, True, offset)

    if xb is not None and isinstance(xb, (int, float)):
        _add_scalar_threshold_signals(df, indicator, xb, cross_values, False, offset)

    # xserie is the default value for both xserie_a and xserie_b
    if xserie_a is None:
        xserie_a = xserie
    if xserie_b is None:
        xserie_b = xserie

    _add_series_signals(df, indicator, xserie_a, cross_series, True, offset)
    _add_series_signals(df, indicator, xserie_b, cross_series, False, offset)

    return df


def crossover(
    series_a: Series,
    series_b: Series,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Crossover: series_a crosses above series_b (tulipy: CROSSOVER).

    Returns 1 on bars where series_a crosses from below to above series_b,
    0 otherwise.  Equivalent to cross(series_a, series_b, above=True).
    """
    result = cross(series_a, series_b, above=True, asint=asint, offset=offset)
    if result is None:
        return None
    result.name = f"{series_a.name}_XA_{series_b.name}"
    result.category = "utility"
    return result


def crossany(
    series_a: Series,
    series_b: Series,
    asint: bool = True,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Cross in either direction (tulipy: CROSSANY).

    Returns 1 on any bar where series_a and series_b cross (either up or
    down), 0 otherwise.
    """
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    if series_a is None or series_b is None:
        return None
    offset = get_offset(offset)

    xa = cross(series_a, series_b, above=True, asint=False)
    xb = cross(series_a, series_b, above=False, asint=False)
    result = xa | xb
    if asint:
        result = result.astype(int)

    result = apply_offset(result, offset)

    result.name = f"{series_a.name}_X_{series_b.name}"
    result.category = "utility"
    return result


def lag(
    close: Series,
    period: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Lag / Shift (tulipy: LAG).

    Returns close shifted back by *period* bars.  Equivalent to
    close.shift(period).
    """
    period = int(period) if period and period > 0 else 1
    close = verify_series(close)
    offset = get_offset(offset)
    if close is None:
        return None

    result = close.shift(period)

    result = apply_offset(result, offset)

    result.name = f"LAG_{period}"
    result.category = "utility"
    return result
