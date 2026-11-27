import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from tests.config import get_sample_data

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


def assert_all_nan(result, index=None):
    """Input shorter than the window yields an all-NaN Series/DataFrame, not None (issue #145, case B).

    When *index* is given, the result must be aligned to it.
    """
    assert isinstance(result, (Series, DataFrame)), f"expected an all-NaN result, got {type(result).__name__}"
    if index is not None:
        assert result.index.equals(index), "all-NaN result must keep the input index"
    values = result.to_numpy(dtype=float)
    assert values.size == 0 or bool(np.isnan(values).all()), "short-input result must be all NaN"


@contextmanager
def assert_logs(caplog: pytest.LogCaptureFixture, logger: str, level: str) -> Iterator[list[str]]:
    """Like ``unittest.TestCase.assertLogs``: the block must log at least one record
    of *level* or above on *logger* or one of its children.

    Yields the list of messages, filled in when the block exits.
    """
    levelno = logging.getLevelNamesMapping()[level]
    messages: list[str] = []
    start = len(caplog.records)
    with caplog.at_level(levelno, logger=logger):
        yield messages
    for record in caplog.records[start:]:
        if record.levelno >= levelno and (record.name == logger or record.name.startswith(f"{logger}.")):
            messages.append(record.getMessage())
    assert messages, f"no {level} or higher record logged on {logger!r}"


# ---------------------------------------------------------------------------
# TA-Lib comparison
# ---------------------------------------------------------------------------
#
# Used by tests/oracle/test_oracle_registry.py. Native and TA-Lib results agree
# to rounding. Measured over its ORACLE and VARIANTS cases on every frame (TA-Lib
# 0.8.3; Windows and Linux, Python 3.11 to 3.15, numba on and off), 87 of 107
# cases stay within a tenth of TALIB_ATOL * min(1, scale) + TALIB_RTOL * |expected|;
# the median case reads TA-Lib's value exactly. The other 20 cancel digits where
# a value is small against its column, and TOLERANCE in that module gives each a
# factor with the bar that needs it. The absolute term carries the columns that
# cross zero, where a relative error alone is unbounded; it is scaled down for
# columns smaller than 1, so a column of small values cannot pass on it alone.
#
# This replaces a correlation fallback (threshold 0.85-0.99) that a scaled,
# offset or partly wrong result passed, a `pdt.assert_*_equal` first try whose
# default rtol of 1e-5 was six orders looser than TALIB_RTOL, and TALIB_ATOL = 1e-9,
# TALIB_RTOL = 1e-8, which most cases used to a fiftieth or less.
TALIB_ATOL = 1e-11
TALIB_RTOL = 1e-11


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


class AccessorValueCase:
    """Base of the accessor value checks: df.ta.<indicator>(append=True) appends
    a column with at least one valid row and a finite last value.

    Loads the sample data once; each test gets a fresh copy.
    """

    @classmethod
    def setup_class(cls):
        cls._base = get_sample_data()

    @classmethod
    def teardown_class(cls):
        del cls._base

    def fresh(self) -> pd.DataFrame:
        """Return an independent copy of the sample data."""
        return self._base.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_appended(self, df: pd.DataFrame, col_name: str) -> None:
        """Assert *col_name* exists, has valid rows, and a finite last value."""
        assert col_name in df.columns, f"Expected column {col_name!r} not found in DataFrame after append"
        series = df[col_name]
        valid = series.dropna()
        assert not valid.empty, f"Column {col_name!r} is all-NaN after append"
        last = float(valid.iloc[-1])
        assert math.isfinite(last), f"Column {col_name!r} last value is not finite: {last}"

    def _check_last_col(self, df: pd.DataFrame) -> None:
        """Check the last column of *df* (for indicators that add a single column)."""
        col = df.columns[-1]
        self._check_appended(df, col)
