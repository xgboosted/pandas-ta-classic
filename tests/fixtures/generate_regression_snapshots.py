"""
Generate regression snapshot values for test_regression.py.

Run this script manually — and ONLY — after intentional algorithm changes:

    python -m tests.fixtures.generate_regression_snapshots

Writes tests/fixtures/regression_snapshots.json.  Snapshots are taken from
this package's own output, so they prove nothing about correctness; their
only job is to make an unintended change visible.  That job requires the
file to be frozen: never invoke this from a test run or a test target, or a
regression will silently overwrite the very snapshot meant to catch it.
Review ``git diff tests/fixtures/regression_snapshots.json`` and commit it
alongside the algorithm change that justifies it.

Snapshots store per-column values at fixed positional indices spread across
the SPY_D.csv time series.  This catches algorithm regressions that only
affect the interior of a series (e.g. EMA initialisation, window edge
handling) rather than just the endpoint.

The first checkpoint sits past the warmup period of every tracked indicator;
positions where an indicator is still NaN serialise as null and are skipped
by the test.  Spacing is tighter near the warmup edge, where initialisation
bugs live, and then even across the rest of the series.  SPY_D.csv has 5222
rows, so the last checkpoint is its final bar — the previous set stopped at
3000 and left the final 2221 rows checked only by the endpoint assertion in
test_indicator_values.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# Re-use the same indicator compute function from generate_fixtures to stay
# in sync with the full set of 43 tracked indicators.
from tests.fixtures.generate_fixtures import _load, _indicators

# ---------------------------------------------------------------------------
# Snapshot configuration
# ---------------------------------------------------------------------------

_CHECKPOINTS = [
    50,
    100,
    200,
    350,
    500,
    750,
    1000,
    1250,
    1500,
    1800,
    2100,
    2400,
    2700,
    3000,
    3300,
    3600,
    3900,
    4200,
    4500,
    4800,
    5000,
    5100,
    5200,
    5221,
]
_OUT_PATH = Path(__file__).parent / "regression_snapshots.json"


def _value_at(series: pd.Series, idx: int) -> float | None:
    """Return the value at positional index *idx*, or None if NaN / out of range."""
    if idx >= len(series):
        return None
    val = series.iloc[idx]
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return round(f, 8)


def _snapshots_for(result) -> dict[str, dict[str, float | None]]:
    """Return {col_name: {str(idx): value_or_null}} for each column."""
    if result is None:
        return {}
    if isinstance(result, pd.Series):
        result = result.to_frame(name=result.name)
    out: dict[str, dict] = {}
    for col in result.columns:
        series = result[col]
        out[col] = {str(idx): _value_at(series, idx) for idx in _CHECKPOINTS}
    return out


def generate() -> None:
    df = _load()
    indicators = _indicators(df)

    # Merge-mode: load existing snapshots (if any) so optional-dependency keys
    # (e.g. cmo_14 when tulipy is absent) are preserved across regenerations.
    if _OUT_PATH.exists():
        with open(_OUT_PATH) as fh:
            snapshots: dict[str, dict] = json.load(fh)
    else:
        snapshots = {}

    for key, ref, actual in indicators:
        # For regression indicators, ref IS the native result (actual is None).
        # For reference indicators, actual is the native result.
        result = actual if actual is not None else ref
        col_snaps = _snapshots_for(result)
        if col_snaps:
            snapshots[key] = col_snaps
            cols = list(col_snaps.keys())
            print(f"  OK    {key!r:<32} cols={cols}")
        else:
            print(f"  SKIP  {key!r:<32} (returned None)")

    with open(_OUT_PATH, "w") as fh:
        json.dump(snapshots, fh, indent=2)
    # ASCII only -- see the matching note in generate_fixtures.generate().
    print(f"\nWrote {len(snapshots)} regression snapshots -> {_OUT_PATH}")


if __name__ == "__main__":
    generate()
