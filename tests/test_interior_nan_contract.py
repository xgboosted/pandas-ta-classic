"""What one missing bar in the middle of a series does to each indicator.

There is no single right answer, so the behaviour is pinned per group and any
indicator that changes group fails here (see "Missing Values" in
docs/indicators.rst):

* RECOVERS (the default): the result is NaN near the gap and, once the gap has
  left every window and recursion, equals the clean-series result again.
* CUMULATIVE: a running total loses the missing bar's contribution for good,
  so later values differ from the clean series by a persistent amount.
* WHOLE_SERIES: a fit over the entire series (``tos_stdevall``) changes
  everywhere when one bar is missing, by a small persistent amount.

No indicator may publish a value it could not compute, or go NaN for good.

Before this contract ``rsx`` published a fabricated 50.0 on every bar after one
missing close, ``ebsw`` was off by about 1.0 on a -1..1 scale for good, ``ha``
kept publishing HA_high/HA_low/HA_close computed without HA_open, and 19
recursive indicators (``macd``, ``kama``, ``jma``, ``mama``, the six ``ht_*``,
...) reported NaN from the gap to the end of the series.
"""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

GAP = 600
TAIL = 200  # compared bars at the end, 700+ bars after the gap
COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
NOT_OHLCV = {"add", "sub", "mult", "div", "above", "above_value", "below", "below_value", "cross", "cross_value"}
NOT_OHLCV |= {"long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma", "vp"}
CUMULATIVE = {"ad", "aobv", "nvi", "obv", "pvi", "pvt", "wad"}
WHOLE_SERIES = {"tos_stdevall"}
# Non-causal by design (see test_lookahead.py): bars before the gap depend on later bars.
LOOKAHEAD = {"dpo", "ichimoku", "tos_stdevall"}
INDICATORS = sorted({name for names in ta.Category.values() for name in names} - NOT_OHLCV)


@pytest.fixture(scope="module")
def frames():
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    clean = df.iloc[-1500:]
    gapped = clean.copy()
    gapped.iloc[GAP, gapped.columns.get_loc("close")] = np.nan
    return clean, gapped


def _call(name, df):
    fn = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(**{p: df[c] for p, c in COLUMNS.items() if p in inspect.signature(fn).parameters})
    frame = result.to_frame() if isinstance(result, pd.Series) else result
    return frame.to_numpy(float, na_value=np.nan)


@pytest.mark.parametrize("name", INDICATORS)
def test_one_missing_bar(name, frames):
    clean, gapped = (_call(name, df) for df in frames)
    if name not in LOOKAHEAD:
        np.testing.assert_array_equal(np.isnan(gapped[:GAP]), np.isnan(clean[:GAP]), err_msg=f"{name}: bars before the gap changed")
    tail_clean, tail_gapped = clean[-TAIL:], gapped[-TAIL:]
    np.testing.assert_array_equal(np.isnan(tail_gapped), np.isnan(tail_clean), err_msg=f"{name}: NaN after the gap has passed")
    if name in CUMULATIVE | WHOLE_SERIES:
        return
    np.testing.assert_allclose(tail_gapped, tail_clean, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=f"{name}: did not recover")
