"""Nullable numeric input computes as float64 input does.

Data read with ``dtype_backend="numpy_nullable"`` or built with ``Float64`` /
``Int64`` columns reaches every indicator. For every registered indicator, the
result on such a frame must equal the result on the same frame in float64,
with ``pd.NA`` treated as NaN.

Before this contract ``cdl_doji`` (and with it ``cdl_pattern``),
``increasing``, ``decreasing`` and ``ttm_trend`` raised ``cannot convert NA to
integer``, ``smc_sweep`` raised ``boolean value of NA is ambiguous`` and
``td_seq`` failed to compile its numba kernel, on ``Float64`` input (``Int64``
too, except for ``cdl_doji`` and ``cdl_pattern``), NA or not: a comparison against a NaN warm-up
value gives ``pd.NA`` in the nullable dtype.

Indicators are discovered through ``ta.Category``, so new ones are covered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.test_leading_nan_contract import INDICATORS, _call

NA_ROW = 300


@pytest.fixture(scope="module")
def frames():
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    df = df.iloc[-600:].astype(float)
    nullable = df.astype({c: "Float64" for c in df.columns if c != "volume"} | {"volume": "Int64"})
    with_na, with_nan = nullable.copy(), df.copy()
    with_na.iloc[NA_ROW] = pd.NA
    with_nan.iloc[NA_ROW] = np.nan
    return df, nullable, with_nan, with_na


@pytest.mark.parametrize("name", INDICATORS)
def test_nullable_input_matches_float64(name, frames):
    df, nullable, with_nan, with_na = frames
    pd.testing.assert_frame_equal(_call(name, nullable), _call(name, df), obj=name)
    pd.testing.assert_frame_equal(_call(name, with_na), _call(name, with_nan), obj=f"{name} with NA")


def _non_ohlcv_calls(c, o, fast, slow, rsi, up):
    # The indicators INDICATORS leaves out take derived series, not OHLCV columns.
    return {
        "add": lambda: ta.add(c, o),
        "sub": lambda: ta.sub(c, o),
        "mult": lambda: ta.mult(c, o),
        "div": lambda: ta.div(c, o),
        "above": lambda: ta.above(c, o),
        "below": lambda: ta.below(c, o),
        "cross": lambda: ta.cross(c, o),
        "above_value": lambda: ta.above_value(c, 400),
        "below_value": lambda: ta.below_value(c, 400),
        "cross_value": lambda: ta.cross_value(c, 400),
        "long_run": lambda: ta.long_run(fast, slow, length=2),
        "short_run": lambda: ta.short_run(fast, slow, length=2),
        "tsignals": lambda: ta.tsignals(up),
        "xsignals": lambda: ta.xsignals(rsi, 70, 30),
        "beta": lambda: ta.beta(c, o),
        "correl": lambda: ta.correl(c, o),
        "mavp": lambda: ta.mavp(c, c * 0 + 5),
        "ma": lambda: ta.ma("ema", c, length=5),
    }


@pytest.fixture(scope="module")
def non_ohlcv(frames):
    df = frames[0]
    c, o = df["close"], df["open"]
    # Derived inputs carry a NaN warm-up, which the nullable copies hold as pd.NA.
    fast, slow, rsi = ta.ema(c, 5), ta.ema(c, 20), ta.rsi(c)
    # A trend that is unknown during the warm-up: boolean with pd.NA there made
    # tsignals raise "Invalid value '0' for dtype 'boolean'"; float64 reads it as NaN.
    warmup = ta.sma(c, 10).isna()
    up = c > ta.sma(c, 10)
    plain = _non_ohlcv_calls(c, o, fast, slow, rsi, up.astype(float).mask(warmup))
    as_f64 = [s.astype("Float64") for s in (c, o, fast, slow, rsi)]
    nullable = _non_ohlcv_calls(*as_f64, up.astype("boolean").mask(warmup))
    return plain, nullable


@pytest.mark.parametrize("name", sorted(_non_ohlcv_calls(*[None] * 6)))
def test_nullable_input_matches_float64_for_derived_inputs(name, non_ohlcv):
    plain, nullable = non_ohlcv
    expected, result = plain[name](), nullable[name]()
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected, obj=name)
    else:
        pd.testing.assert_series_equal(result, expected, obj=name)


@pytest.mark.parametrize("dtype", ["double[pyarrow]", "int64[pyarrow]"])
def test_pyarrow_input_matches_float64(dtype, frames):
    pytest.importorskip("pyarrow")
    df = frames[0]
    # int64[pyarrow] needs whole numbers: prices in cents, as tick data arrives.
    base = (df * 100).round() if dtype.startswith("int") else df
    arrow = base.astype(dtype)
    for name in INDICATORS:
        pd.testing.assert_frame_equal(_call(name, arrow), _call(name, base), obj=f"{name} {dtype}")


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_integer_ticks_match_float64(dtype, frames):
    # Feeds deliver prices in ticks: whole numbers as int64, or as Int64 once a
    # column holds a missing value. Prices included, not only the volume.
    # Values must match exactly. Int64 is converted to float64 like the other
    # nullable dtypes, so its dtype must match too; numpy int64 is left alone,
    # and integer-valued results (ceil, npround, pvol, ...) keep int64 by numpy's rules.
    base = (frames[0] * 100).round()
    ticks = base.astype(dtype)
    for name in INDICATORS:
        pd.testing.assert_frame_equal(_call(name, ticks), _call(name, base), check_dtype=dtype == "Int64", obj=f"{name} {dtype}")


def test_pyarrow_boolean_trend_matches_float64(frames):
    pytest.importorskip("pyarrow")
    c = frames[0]["close"]
    warmup = ta.sma(c, 10).isna()
    up = c > ta.sma(c, 10)
    # bool[pyarrow] is not a numeric dtype to pandas, so it was passed through and
    # tsignals raised ArrowInvalid on the null
    expected = ta.tsignals(up.astype(float).mask(warmup))
    pd.testing.assert_frame_equal(ta.tsignals(up.astype("bool[pyarrow]").mask(warmup)), expected)


@pytest.mark.parametrize("dtype", ["float32"])  # float16 cannot hold SPY volume (max 65504)
def test_narrow_float_input_matches_float64(dtype, frames):
    # A float32 frame (a memory-saving read, a float32 feed) computed in float32:
    # 54 indicators returned float32, 4 mixed float32 and float64 columns, and the
    # values lost digits. It now computes in float64, so the result equals the same
    # values given as float64, dtype included.
    narrow = frames[0].astype(dtype)
    base = narrow.astype("float64")
    for name in INDICATORS:
        pd.testing.assert_frame_equal(_call(name, narrow), _call(name, base), obj=f"{name} {dtype}")
