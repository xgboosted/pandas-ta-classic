"""
Priority 1 — Golden fixture tests.

Each test asserts two things for every column of every tracked indicator:
  1. The last non-NaN value matches the stored golden value within
     GOLDEN_ATOL + GOLDEN_RTOL * |golden| (see tests/assertions.py).
  2. The number of non-NaN rows matches the stored count exactly.

To regenerate fixtures after an intentional algorithm change run:
    python -m tests.fixtures.generate_fixtures
"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.assertions import golden_value_close as _approx_equal
from tests.config import get_sample_data
from tests.fixtures.tracked_indicators import compute_all as _compute_all

# ---------------------------------------------------------------------------
# Load fixtures & sample data
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "expected_values.json"
with open(_FIXTURE_PATH) as _fh:
    _FIXTURES: dict[str, dict] = json.load(_fh)

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

# Comparison lives in tests/assertions.py — see there for how GOLDEN_ATOL and
# GOLDEN_RTOL were derived, and why both terms are needed.


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestIndicatorValues:
    """Assert that every tracked indicator produces the expected golden values."""

    @classmethod
    def setup_class(cls):
        cls.df = get_sample_data()
        cls.results = _compute_all(cls.df)

    @classmethod
    def teardown_class(cls):
        del cls.df
        del cls.results

    # ------------------------------------------------------------------
    # Internal helper used by every generated test method
    # ------------------------------------------------------------------

    def _check_fixture(self, fixture_key: str, subtests) -> None:
        """Assert last value and non-NaN count for every column of fixture_key."""

        assert fixture_key in self.results, f"No result computed for {fixture_key!r}"
        result = self.results[fixture_key]
        if result is None:
            pytest.fail(f"{fixture_key!r} returned None")

        # Normalise to DataFrame so we iterate columns uniformly
        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name)

        expected_fixture = _FIXTURES[fixture_key]

        for col, expected in expected_fixture.items():
            with subtests.test(col=col):
                assert col in result.columns, f"Column {col!r} missing from {fixture_key!r} result"

                series = result[col]
                actual_non_nan = int(series.notna().sum())
                assert actual_non_nan == expected["non_nan_count"], f"{fixture_key!r}[{col!r}]: non_nan_count " f"{actual_non_nan} != {expected['non_nan_count']}"

                expected_last = expected["last_value"]
                if expected_last is None:
                    continue  # column was all-NaN when fixtures were generated; the next column is still checked

                clean = series.dropna()
                assert not clean.empty, f"{fixture_key!r}[{col!r}]: result is all-NaN but fixture has a value"
                actual_last = float(clean.iloc[-1])
                assert not (math.isnan(actual_last) or math.isinf(actual_last)), f"{fixture_key!r}[{col!r}]: last value is NaN/Inf"
                assert _approx_equal(actual_last, expected_last), f"{fixture_key!r}[{col!r}]: last value {actual_last:.8f} " f"differs from golden {expected_last:.8f}"


# ---------------------------------------------------------------------------
# One test method per fixture key, generated from expected_values.json as
# test_snapshots.py does for its snapshots, so the test set cannot drift from
# the fixture file.
# ---------------------------------------------------------------------------


def _make_fixture_test(fixture_key: str) -> Callable[[TestIndicatorValues, Any], None]:
    def test(self: TestIndicatorValues, subtests) -> None:
        self._check_fixture(fixture_key, subtests)

    test.__name__ = f"test_{fixture_key}"
    test.__doc__ = f"Golden last value and non-NaN count for {fixture_key!r}."
    return test


def _attach_fixture_tests() -> None:
    for fixture_key in _FIXTURES:
        setattr(TestIndicatorValues, f"test_{fixture_key}", _make_fixture_test(fixture_key))


_attach_fixture_tests()
