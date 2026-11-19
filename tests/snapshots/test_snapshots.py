"""
Priority 4 — Snapshot regression tests.

Each test re-computes an indicator on SPY_D.csv and compares spot-check
values at 24 fixed positional indices against stored golden snapshots,
spanning index 50 through the final bar at 5221.  Spacing is tighter near the
warmup edge, where initialisation bugs live.  Checkpoints where an indicator
is still NaN are stored as null and skipped.

This catches algorithm regressions that only affect the interior of the
series (e.g. EMA initialisation, window boundary handling) — complementing
test_golden_values.py which checks only the last value and non-NaN count.

Run the full suite:
    pytest tests/snapshots/test_snapshots.py
"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from tests.assertions import golden_value_close as _approx_equal

# The indicator set is imported rather than restated: tracked_indicators.py
# builds all 223 tracked results that test_golden_values.py checks, with the same parameters the
# generators use, and a second hand-maintained copy here had silently drifted
# to a 43-entry subset, leaving 180 stored snapshots asserted by nothing.
from tests.config import get_sample_data
from tests.fixtures.tracked_indicators import compute_all as _compute_all

# ---------------------------------------------------------------------------
# Load snapshots
# ---------------------------------------------------------------------------

_SNAP_PATH = Path(__file__).parents[1] / "fixtures" / "regression_snapshots.json"
with open(_SNAP_PATH) as _fh:
    _SNAPSHOTS: dict[str, dict] = json.load(_fh)

# ---------------------------------------------------------------------------
# Test class
#
# Tolerance is shared with test_golden_values.py: snapshots carry the same
# round(v, 8) quantisation, so the same floor applies.
# ---------------------------------------------------------------------------


class TestRegressionSnapshots:
    """Spot-check indicator values at 24 fixed positions across SPY_D.csv."""

    @classmethod
    def setup_class(cls):
        cls.df = get_sample_data()
        cls.results = _compute_all(cls.df)

    @classmethod
    def teardown_class(cls):
        del cls.df
        del cls.results

    def test_every_tracked_result_has_a_snapshot(self):
        """The tests are generated from regression_snapshots.json, so a result missing there would go unchecked."""
        missing, stale = set(self.results) - set(_SNAPSHOTS), set(_SNAPSHOTS) - set(self.results)
        assert not missing and not stale, f"tracked without a snapshot: {sorted(missing)}; snapshot without a result: {sorted(stale)}"

    def _check_snapshot(self, fixture_key: str, subtests) -> None:
        assert fixture_key in self.results, f"No result computed for {fixture_key!r}"
        result = self.results[fixture_key]
        assert result is not None, f"{fixture_key!r} returned None"

        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name)

        col_snaps = _SNAPSHOTS[fixture_key]

        for col, checkpoints in col_snaps.items():
            with subtests.test(col=col):
                assert col in result.columns, f"Column {col!r} missing from {fixture_key!r}"
                series = result[col]

                for idx_str, expected_val in checkpoints.items():
                    idx = int(idx_str)
                    with subtests.test(idx=idx):
                        if idx >= len(series):
                            # The generator stores null for out-of-range
                            # positions.  A few indicators return a series
                            # shorter than the input (vp aggregates into 10
                            # bins, tos_stdevall into 30), so every checkpoint
                            # falls past the end.  Requiring null here turns
                            # the mismatch into a length-regression check.
                            assert expected_val is None, f"{fixture_key!r}[{col!r}] at idx={idx}: result has only " f"{len(series)} rows but snapshot holds {expected_val}"
                        elif expected_val is None:
                            # Snapshot was NaN — actual must also be NaN
                            actual = series.iloc[idx]
                            assert pd.isna(actual), f"{fixture_key!r}[{col!r}] at idx={idx}: " f"snapshot was NaN but got {actual}"
                        else:
                            actual_raw = series.iloc[idx]
                            assert not pd.isna(actual_raw), f"{fixture_key!r}[{col!r}] at idx={idx}: " f"got NaN but snapshot has value {expected_val}"
                            actual = float(actual_raw)
                            assert not (math.isnan(actual) or math.isinf(actual)), f"{fixture_key!r}[{col!r}] at idx={idx}: value is NaN/Inf"
                            assert _approx_equal(actual, expected_val), f"{fixture_key!r}[{col!r}] at idx={idx}: " f"actual={actual:.8f} != snapshot={expected_val:.8f}"


# ---------------------------------------------------------------------------
# One test method per snapshot key, generated from the JSON itself.
#
# These used to be hand-written, which meant the test set and the snapshot
# file could drift apart -- and they had, badly: 180 of the 223 stored
# snapshots were asserted by nothing at all, including 55 of the 57
# indicators that have no independent oracle and are therefore protected by
# the snapshot alone.  Generating from _SNAPSHOTS makes that drift
# impossible.
# ---------------------------------------------------------------------------


def _make_snapshot_test(fixture_key: str) -> Callable[[TestRegressionSnapshots, Any], None]:
    def test(self: TestRegressionSnapshots, subtests) -> None:
        self._check_snapshot(fixture_key, subtests)

    test.__name__ = f"test_{fixture_key}"
    test.__doc__ = f"Snapshot regression for {fixture_key!r}."
    return test


def _attach_snapshot_tests() -> None:
    for fixture_key in sorted(_SNAPSHOTS):
        setattr(TestRegressionSnapshots, f"test_{fixture_key}", _make_snapshot_test(fixture_key))


_attach_snapshot_tests()
