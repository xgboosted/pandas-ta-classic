"""Gaps in the data, in every shape a feed produces them, for every indicator.

``test_interior_nan_contract.py`` pins one shape: a single bar whose ``close``
is missing. Feeds lose data in more ways than that, and each reaches a
different branch:

* ``whole_bar``     -- every column of one bar is NaN (a session that is missing).
* ``five_bars``     -- a run of missing sessions (an outage).
* ``volume_only``   -- prices arrived, volume did not.
* ``high_low_only`` -- a quote-only bar: close but no range.
* ``last_bar``      -- the newest bar is not filled yet (live data that lags).

For the interior shapes the contract is the one the single-bar test states:
bars before the gap are untouched, and once the gap has left every window and
recursion the result equals the clean one again -- except for running totals
(``CUMULATIVE``), which lose the bar's contribution for good, and whole-series
fits. For ``last_bar``, every earlier bar is untouched.

``test_bar_based_indicators_ignore_the_calendar`` covers the other kind of gap,
missing *rows*: a weekend, a holiday, a halted day with no row at all. An
indicator that counts bars must give the same numbers whatever the timestamps
are; only the ones that anchor to the calendar (``TIME_BASED``) may differ.

Indicators are discovered through ``ta.Category``. A change of behaviour that is
decided rather than a defect goes into a table here, with the reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_leading_nan_contract import INDICATORS, _call as _frame

GAP = 600
TAIL = 200  # compared bars at the end, 700+ bars after the gap
CUMULATIVE = {"ad", "aobv", "nvi", "obv", "pvi", "pvt", "wad"}
WHOLE_SERIES = {"tos_stdevall"}
# Non-causal by design (see test_lookahead.py): bars before a gap depend on later bars.
LOOKAHEAD = {"dpo", "ichimoku", "tos_stdevall"}
# Anchored to calendar periods, so the timestamps are input.
TIME_BASED = {"vwap", "cpr"}

GAPS = {
    "whole_bar": (slice(GAP, GAP + 1), ("open", "high", "low", "close", "volume")),
    "five_bars": (slice(GAP, GAP + 5), ("open", "high", "low", "close", "volume")),
    "volume_only": (slice(GAP, GAP + 1), ("volume",)),
    "high_low_only": (slice(GAP, GAP + 1), ("high", "low")),
}

# (indicator, gap) pairs that break the contract today, as strict xfails: one
# that starts passing fails, so the table cannot go stale.
_FISHER_RECURSION = (
    "OPEN DEFECT: fisher's recursion v = 0.66 * pos + 0.67 * v takes one NaN from high/low and "
    "stays NaN for the rest of the series; close-only gaps never reach it because fisher reads no close"
)
OPEN_FINDINGS: dict[tuple[str, str], str] = {
    ("fisher", "whole_bar"): _FISHER_RECURSION,
    ("fisher", "five_bars"): _FISHER_RECURSION,
    ("fisher", "high_low_only"): _FISHER_RECURSION,
}


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df.iloc[-1500:]


def _call(name: str, df: pd.DataFrame) -> np.ndarray:
    return _frame(name, df).to_numpy(float, na_value=np.nan)


def _params(gaps):
    for gap in gaps:
        for name in INDICATORS:
            reason = OPEN_FINDINGS.get((name, gap))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(name, gap, marks=marks, id=f"{name}-{gap}")


@pytest.mark.parametrize(("name", "gap"), list(_params(GAPS)))
def test_an_interior_gap_is_forgotten(name: str, gap: str, clean: pd.DataFrame) -> None:
    rows, columns = GAPS[gap]
    gapped = clean.copy()
    for column in columns:
        gapped.iloc[rows, gapped.columns.get_loc(column)] = np.nan
    before, after = _call(name, clean), _call(name, gapped)

    if name not in LOOKAHEAD:
        np.testing.assert_array_equal(after[: rows.start], before[: rows.start], err_msg=f"{name}: bars before the gap changed")
    tail_before, tail_after = before[-TAIL:], after[-TAIL:]
    np.testing.assert_array_equal(np.isnan(tail_after), np.isnan(tail_before), err_msg=f"{name}: NaN long after the gap")
    if name in CUMULATIVE | WHOLE_SERIES:
        return
    np.testing.assert_allclose(tail_after, tail_before, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=f"{name}: did not recover from the gap")


@pytest.mark.parametrize(("name", "gap"), list(_params(["last_bar"])))
def test_a_missing_last_bar_changes_no_earlier_bar(name: str, gap: str, clean: pd.DataFrame) -> None:
    lagging = clean.copy()
    lagging.iloc[-1] = np.nan
    before, after = _call(name, clean), _call(name, lagging)
    assert after.shape == before.shape, f"{name}: {after.shape} rows/columns, {before.shape} without the gap"
    if name not in LOOKAHEAD:
        np.testing.assert_array_equal(after[:-1], before[:-1], err_msg=f"{name}: a missing newest bar changed history")


@pytest.mark.parametrize(("name", "gap"), list(_params(["calendar"])))
def test_bar_based_indicators_ignore_the_calendar(name: str, gap: str, clean: pd.DataFrame) -> None:
    """The same bars under a gappy calendar give the same numbers.

    Rows are re-stamped onto a calendar with irregular holes -- single days, a
    week, a month -- so the index has no ``freq`` and the spacing varies, as it
    does for any real exchange.
    """
    if name in TIME_BASED:
        pytest.skip(f"{name} anchors to calendar periods")
    rng = np.random.default_rng(3)
    steps = rng.choice([1, 1, 1, 1, 2, 3, 7, 30], size=len(clean))
    holes = clean.copy()
    holes.index = pd.Timestamp("2000-01-03") + pd.to_timedelta(np.cumsum(steps), unit="D")
    np.testing.assert_array_equal(_call(name, holes), _call(name, clean), err_msg=f"{name}: depends on the timestamps, not just the bars")
