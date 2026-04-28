"""
Generate regression snapshot values for test_regression.py.

Run this script manually after intentional algorithm changes:

    python -m tests.fixtures.generate_regression_snapshots

Writes tests/fixtures/regression_snapshots.json. Commit the updated file
alongside the algorithm change so CI continues to pass.

Snapshots store per-column values at five fixed positional indices spread
across the SPY_D.csv time series.  This catches algorithm regressions that
only affect the interior of a series (e.g. EMA initialisation, window edge
handling) rather than just the endpoint.

Checkpoint indices (positional, 0-based): 50, 200, 500, 1500, 3000
All are well past the warmup period for every tracked indicator.
"""

import json
import math
from pathlib import Path

import pandas as pd

# Re-use the same indicator compute function from generate_fixtures to stay
# in sync with the full set of 43 tracked indicators.
from tests.fixtures.generate_fixtures import _load, _indicators  # noqa: E402

# ---------------------------------------------------------------------------
# Snapshot configuration
# ---------------------------------------------------------------------------

_CHECKPOINTS = [50, 200, 500, 1500, 3000]
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

    snapshots: dict[str, dict] = {}
    for key, result in indicators:
        col_snaps = _snapshots_for(result)
        if col_snaps:
            snapshots[key] = col_snaps
            cols = list(col_snaps.keys())
            print(f"  OK    {key!r:<32} cols={cols}")
        else:
            print(f"  SKIP  {key!r:<32} (returned None)")

    with open(_OUT_PATH, "w") as fh:
        json.dump(snapshots, fh, indent=2)
    print(f"\nWrote {len(snapshots)} regression snapshots → {_OUT_PATH}")


if __name__ == "__main__":
    generate()
