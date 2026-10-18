import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

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
# orders of margin for platform and BLAS differences across the 3.11-3.15 CI
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
    expected_columns: list[str] | None = None
    none_arg_idx: int | None = 0
    kwargs: dict = field(default_factory=dict)
    length_override: int | None = None


def _assert_same_result(test_case, actual, expected, msg):
    test_case.assertIsInstance(actual, type(expected), msg)
    try:
        if isinstance(expected, DataFrame):
            assert_frame_equal(actual, expected)
        else:
            assert_series_equal(actual, expected)
    except AssertionError as ex:
        raise test_case.failureException(f"{msg}\n{ex}") from None


def assert_offset(test_case, func, args, **kwargs):
    """``offset=1`` returns the default result shifted forward by one bar."""
    expected = func(*args, **kwargs).shift(1)
    _assert_same_result(test_case, func(*args, offset=1, **kwargs), expected, f"{func.__name__}: offset=1 must shift the result by one bar")


def assert_fill(test_case, func, args, **kwargs):
    """``fillna`` and ``fill_method`` match the pandas operation on the default result."""
    base = func(*args, **kwargs)
    name = func.__name__
    _assert_same_result(test_case, func(*args, fillna=0, **kwargs), base.fillna(0), f"{name}: fillna=0 must equal result.fillna(0)")
    _assert_same_result(test_case, func(*args, fill_method="ffill", **kwargs), base.ffill(), f"{name}: fill_method='ffill' must equal result.ffill()")
    _assert_same_result(test_case, func(*args, fill_method="bfill", **kwargs), base.bfill(), f"{name}: fill_method='bfill' must equal result.bfill()")


def assert_length_in_name(test_case, func, args, length, **kwargs):
    result = func(*args, length=length, **kwargs)
    test_case.assertIsNotNone(result)
    test_case.assertIn(str(length), result.name)


def assert_all_nan(test_case, result, index=None):
    """Input shorter than the window yields an all-NaN Series/DataFrame, not None (issue #145, case B).

    When *index* is given, the result must be aligned to it.
    """
    test_case.assertIsInstance(result, (Series, DataFrame), f"expected an all-NaN result, got {type(result).__name__}")
    if index is not None:
        test_case.assertTrue(result.index.equals(index), "all-NaN result must keep the input index")
    values = result.to_numpy(dtype=float)
    test_case.assertTrue(values.size == 0 or bool(np.isnan(values).all()), "short-input result must be all NaN")


def assert_none_guard(test_case, func, args, none_arg_idx=0, **kwargs):
    none_args = list(args)
    none_args[none_arg_idx] = None
    test_case.assertIsNone(func(*none_args, **kwargs))


# ---------------------------------------------------------------------------
# TA-Lib comparison
# ---------------------------------------------------------------------------
#
# Native and TA-Lib results agree to rounding: measured over all 71
# assert_talib calls on SPY_D, the worst column (`adosc`, whose values reach
# 6e8) needs |actual - expected| <= 2e-10 * (1 + |expected|) with TA-Lib 0.8.0
# (1.2e-10 with 0.7.1), 1.9% of TALIB_ATOL + TALIB_RTOL * |expected|.  That
# leaves ~50x margin for other TA-Lib releases and BLAS builds in the CI matrix.
# The absolute term carries the columns that cross zero, where a relative error
# alone is unbounded. The most it has to cover beyond TALIB_RTOL * |expected| is
# 8.2e-11 with TA-Lib 0.8.2 (`linreg_angle`, degrees near 0) and 2.4e-13 with
# 0.7.1 (`stochrsi`), so TALIB_ATOL = 1e-9 keeps ~12x margin without letting a
# column of small values (`macd`, `apo`) pass on its relative error alone.
#
# This replaces a correlation fallback (threshold 0.85-0.99) that a scaled,
# offset or partly wrong result passed, and a `pdt.assert_*_equal` first try
# whose default rtol of 1e-5 was three orders looser than TALIB_RTOL.
TALIB_ATOL = 1e-9
TALIB_RTOL = 1e-8


def assert_talib(test_case, result, expected):
    """*result* equals TA-Lib's *expected* within TALIB_ATOL + TALIB_RTOL * |expected|.

    Columns are compared by position (TA-Lib's names differ), and NaN must sit
    exactly where TA-Lib has NaN, so a different warm-up length fails too.
    """
    actual = result.to_numpy(dtype=float)
    wanted = np.asarray(expected, dtype=float)
    test_case.assertEqual(actual.shape, wanted.shape, "result and TA-Lib output differ in shape")
    if isinstance(expected, (Series, DataFrame)):
        test_case.assertTrue(result.index.equals(expected.index), "result and TA-Lib output differ in index")
    np.testing.assert_allclose(actual, wanted, rtol=TALIB_RTOL, atol=TALIB_ATOL, equal_nan=True)


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
    # vp has no offset: its rows are price bins, not bars
    if "offset" in inspect.signature(spec.func).parameters:
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


def output_columns(result: Any) -> dict[str, Series]:
    """Every output column of an indicator result, keyed uniquely across parts.

    Takes a Series, a DataFrame, or a sequence of either (some indicators
    return a tuple); ``None`` parts are skipped.
    """
    parts = result if isinstance(result, (tuple, list)) else (result,)
    columns: dict[str, Series] = {}
    for position, part in enumerate(parts):
        if part is None:
            continue
        if isinstance(part, Series):
            columns[f"[{position}]{part.name}"] = part
        else:
            for column in part.columns:
                columns[f"[{position}]{column}"] = part[column]
    return columns
