"""Nullable numeric input computes as float64 input does.

Data read with ``dtype_backend="numpy_nullable"`` or built with ``Float64`` /
``Int64`` columns reaches every indicator. For every registered indicator, the
result on such a frame must equal the result on the same frame in float64,
with ``pd.NA`` treated as NaN.

Before this contract ``cdl_doji`` (and with it ``cdl_pattern``),
``increasing``, ``decreasing`` and ``ttm_trend`` raised ``cannot convert NA to
integer``, ``smc_sweep`` raised ``boolean value of NA is ambiguous`` and
``td_seq`` failed to compile its numba kernel, on any ``Float64`` input, NA or
not: a comparison against a NaN warm-up value gives ``pd.NA`` in the nullable
dtype.

Indicators are discovered through ``ta.Category``, so new ones are covered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
