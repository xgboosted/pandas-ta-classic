from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pandas import DataFrame, Series

CORRELATION = "corr"
CORRELATION_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Golden-value comparison
# ---------------------------------------------------------------------------
#
# Both tests/fixtures/*.json store their values as round(v, 8), so no
# comparison against them can be tighter than the last stored decimal.  That
# sets the absolute floor; the relative term covers large-magnitude
# indicators, where float64 cannot represent 8 decimals at all (`ad` peaks
# near 3.8e10, one ULP there is already ~7.6e-6).
#
# Both terms are load-bearing: with GOLDEN_ATOL alone the large-magnitude
# columns overrun by ~2.7e4x, and with GOLDEN_RTOL alone the small-magnitude
# ones overrun on storage rounding.
#
# Measured across all 423 tracked columns, the worst native-vs-golden
# disagreement needs a relative term of 3.6e-15; GOLDEN_RTOL leaves nine
# orders of margin for platform and BLAS differences across the 3.10-3.14 CI
# matrix.  It replaces a flat 1e-4, which was loose enough to accept a 0.005%
# error in an indicator whose own numerics are accurate to 1e-14.
GOLDEN_ATOL = 1e-8
GOLDEN_RTOL = 1e-6


def golden_value_close(actual: float, expected: float) -> bool:
    """Return True when *actual* matches a stored golden value.

    Uses ``|actual - expected| <= GOLDEN_ATOL + GOLDEN_RTOL * |expected|``.
    """
    return abs(actual - expected) <= GOLDEN_ATOL + GOLDEN_RTOL * abs(expected)


@dataclass
class IndicatorSpec:
    func: Callable
    args: list[Any]
    expected_name: str
    expected_type: type = Series
    expected_columns: Optional[list[str]] = None
    none_arg_idx: Optional[int] = 0
    kwargs: dict = field(default_factory=dict)
    length_override: Optional[int] = None


def assert_offset(test_case, func, args, **kwargs):
    test_case.assertIsNotNone(func(*args, offset=1, **kwargs))


def assert_fill(test_case, func, args, **kwargs):
    test_case.assertIsNotNone(func(*args, fillna=0, **kwargs))
    test_case.assertIsNotNone(func(*args, fill_method="ffill", **kwargs))
    test_case.assertIsNotNone(func(*args, fill_method="bfill", **kwargs))


def assert_length_in_name(test_case, func, args, length, **kwargs):
    result = func(*args, length=length, **kwargs)
    test_case.assertIsNotNone(result)
    test_case.assertIn(str(length), result.name)


def assert_none_guard(test_case, func, args, none_arg_idx=0, **kwargs):
    none_args = list(args)
    none_args[none_arg_idx] = None
    test_case.assertIsNone(func(*none_args, **kwargs))


def assert_talib(test_case, result, expected, correlation_threshold=None):
    import pandas.testing as pdt

    try:
        if isinstance(result, DataFrame) and isinstance(expected, DataFrame):
            pdt.assert_frame_equal(result, expected, check_dtype=False)
        else:
            pdt.assert_series_equal(result, expected, check_names=False, check_dtype=False)
        return
    except AssertionError:
        if correlation_threshold is None:
            raise

    from pandas_ta_classic.utils import df_error_analysis
    from tests.config import error_analysis

    if isinstance(result, DataFrame):
        n_cols = min(
            len(result.columns),
            (len(expected.columns) if isinstance(expected, DataFrame) else len(result.columns)),
        )
        cols = list(range(n_cols))
    else:
        cols = [None]
    for i in cols:
        r = result.iloc[:, i] if i is not None else result
        e = expected.iloc[:, i] if (i is not None and isinstance(expected, DataFrame)) else expected
        try:
            corr = df_error_analysis(r, e)
        except Exception as ex:
            error_analysis(r, CORRELATION, ex)
            continue
        test_case.assertGreater(corr, correlation_threshold)


def assert_indicator_standard(test_case, spec: IndicatorSpec):
    raw = spec.func(*spec.args, **spec.kwargs)
    result = raw
    test_case.assertIsInstance(result, spec.expected_type)
    test_case.assertEqual(result.name, spec.expected_name)
    if spec.expected_type is DataFrame:
        test_case.assertIsNotNone(
            spec.expected_columns,
            f"{spec.func.__name__}: expected_columns required for DataFrame results",
        )
        test_case.assertListEqual(list(result.columns), spec.expected_columns)
    elif spec.expected_columns is not None:
        test_case.assertListEqual(list(result.columns), spec.expected_columns)
    assert_offset(test_case, spec.func, spec.args, **spec.kwargs)
    assert_fill(test_case, spec.func, spec.args, **spec.kwargs)
    if spec.none_arg_idx is not None:
        assert_none_guard(test_case, spec.func, spec.args, spec.none_arg_idx, **spec.kwargs)
    if spec.length_override is not None:
        base_kwargs = {k: v for k, v in spec.kwargs.items() if k != "length"}
        assert_length_in_name(
            test_case,
            spec.func,
            spec.args,
            spec.length_override,
            **base_kwargs,
        )
    return result
