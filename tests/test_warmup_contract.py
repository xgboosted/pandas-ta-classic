"""No indicator may publish a value on a bar it cannot compute.

Two properties, checked for every registered OHLCV indicator and every column it
returns. Both catch the same mistake: an output array initialised with
``np.zeros``/``np.ones``, or a boolean comparison against a NaN band cast with
``astype(int)``, hands back the initial value as if it were a result.

``test_no_value_before_the_warm_up``
    A column may not hold a value in front of its own warm-up. The signature is a
    short finite run at the very front, followed by the real NaN warm-up run,
    holding a value outside the range the column takes once it is settled.
    Genuinely one-sided columns (``PSARl``/``PSARs``, ``SUPERTl``/``SUPERTs``,
    ``HILOl``/``HILOs``, ``QQEl``/``QQEs``) also start with a short run, but with
    a value inside their settled range, so they pass.

``test_flags_only_where_there_is_a_value``
    A column whose values are all in {-1, 0, 1} states something about the bar,
    so it must be NaN wherever every continuous column of the same result is NaN.

Found by these checks: ``supertrend`` reported ``SUPERT`` 0.0 on bar 0 and an
uptrend over the whole ATR warm-up, ``qqe`` reported ``QQEb_l``/``QQEb_s`` 0.0 on
bar 0 and an uptrend over its 70 warm-up bars, and ``thermo`` published both
signals on the 20 bars before its average exists. ``squeeze``, ``squeeze_pro``
and ``ttm_trend`` had the same defect, fixed in 0.9.0 before this module existed.
"""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
NOT_OHLCV = {"add", "sub", "mult", "div", "above", "above_value", "below", "below_value", "cross", "cross_value"}
NOT_OHLCV |= {"long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma", "vp"}
INDICATORS = sorted({name for names in ta.Category.values() for name in names} - NOT_OHLCV)

# A premature value sits in the first few bars; further in it is a real result.
LEAD = 5

# int8 cannot hold NaN, so these two report 0 on the single bar before the first
# completed period. Changing the dtype is a separate decision (see the CPR
# section of docs/indicators.rst).
FLAG_EXCEPTIONS = {("cpr", "CPR_POSITION"), ("cpr", "CPR_WIDTH_CLASS")}


@pytest.fixture(scope="module")
def frame():
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df


def _call(name, df):
    fn = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(**{p: df[c] for p, c in COLUMNS.items() if p in inspect.signature(fn).parameters})
    if result is None:
        return None
    out = result.to_frame() if isinstance(result, pd.Series) else result
    return out.apply(pd.to_numeric, errors="coerce")


def _columns(name, df):
    """{column: float array} for one indicator, or {} when it returns nothing."""
    out = _call(name, df)
    if out is None:
        return {}
    return {col: out[col].to_numpy(dtype="float64", na_value=np.nan) for col in out.columns}


def _leading_run(mask):
    """(start, length) of the first finite run, or None when the column is all NaN."""
    finite = np.flatnonzero(~mask)
    if finite.size == 0:
        return None
    start = int(finite[0])
    end = start
    while end + 1 < mask.size and not mask[end + 1]:
        end += 1
    return start, end - start + 1


@pytest.mark.parametrize("name", INDICATORS)
def test_no_value_before_the_warm_up(name, frame):
    for col, values in _columns(name, frame).items():
        mask = np.isnan(values)
        run = _leading_run(mask)
        if run is None:
            continue
        start, length = run
        if start >= LEAD or start + length >= mask.size:
            continue  # not at the front, or no warm-up run follows it
        rest = mask[start + length :]
        gap = int(np.argmax(~rest)) if (~rest).any() else rest.size
        if length >= gap:
            continue  # the run outlasts the gap: a real, one-sided series
        settled = values[start + length + gap :]
        settled = settled[~np.isnan(settled)]
        if settled.size == 0:
            continue
        head = values[start : start + length]
        assert settled.min() <= head.min() and head.max() <= settled.max(), (
            f"{name}['{col}']: bars {start}..{start + length - 1} hold {head.tolist()} before "
            f"{gap} warm-up NaN bars, outside the settled range "
            f"[{settled.min()}, {settled.max()}] — an array initial value, not a result"
        )


@pytest.mark.parametrize("name", INDICATORS)
def test_flags_only_where_there_is_a_value(name, frame):
    columns = _columns(name, frame)
    flags, values = {}, {}
    for col, series in columns.items():
        finite = series[~np.isnan(series)]
        target = flags if finite.size and np.isin(finite, (-1.0, 0.0, 1.0)).all() else values
        target[col] = series
    if not flags or not values:
        return  # nothing to compare a flag against
    undefined = np.all([np.isnan(series) for series in values.values()], axis=0)
    for col, series in flags.items():
        if (name, col) in FLAG_EXCEPTIONS:
            continue
        published = undefined & ~np.isnan(series)
        assert not published.any(), (
            f"{name}['{col}']: flag on {int(published.sum())} bar(s) {np.flatnonzero(published)[:8].tolist()} "
            f"where every value column of the result is NaN"
        )
