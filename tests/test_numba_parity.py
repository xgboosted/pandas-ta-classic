"""The numba JIT path and the pure-Python fallback must publish the same numbers.

``@njit`` kernels run compiled when numba is installed and as plain Python
otherwise, so users with the ``performance`` extra run different code from a
default install. This compares every registered indicator and every candle
pattern, at default arguments on the SPY sample, between the JIT run (this
process) and a ``NUMBA_DISABLE_JIT=1`` run (a subprocess, because the flag is
read when numba is imported).

Floating-point reassociation in compiled code allows ulp-level differences
(4.3e-16 relative was the largest measured), so the tolerance is 1e-12.
Skipped when numba is not installed; the ``testing-numba`` CI job installs it.
"""

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

import pandas_ta_classic as ta

ROOT = Path(__file__).parent.parent
COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
# Not a function of one OHLCV frame: two-series math, signal helpers, a required
# benchmark or periods series, and the ma() dispatcher.
SKIP = {"add", "sub", "mult", "div", "above", "above_value", "below", "below_value", "cross", "cross_value"}
SKIP |= {"long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma"}


def compute_all() -> dict:
    """Every indicator and candle pattern at defaults, flattened to float arrays."""
    df = pd.read_csv(ROOT / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in sorted({i for names in ta.Category.values() for i in names} - SKIP):
            fn = getattr(ta, name)
            result = fn(**{p: df[c] for p, c in COLUMNS.items() if p in inspect.signature(fn).parameters})
            frame = result.to_frame() if isinstance(result, pd.Series) else result
            for col in frame.columns:
                out[f"{name}:{col}"] = frame[col].to_numpy(float, na_value=np.nan)
        for pattern in ta.ALL_PATTERNS:
            result = ta.cdl_pattern(df.open, df.high, df.low, df.close, name=pattern)
            out[f"cdl:{pattern}"] = result.iloc[:, 0].to_numpy(float, na_value=np.nan)
    return out


def test_jit_matches_pure_python(tmp_path):
    target = tmp_path / "nojit.npz"
    # python -c puts the working directory (the repo root) on sys.path
    script = f"import numpy as np; from tests.test_numba_parity import compute_all; np.savez({str(target)!r}, **compute_all())"
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1"}
    subprocess.run([sys.executable, "-c", script], check=True, env=env, cwd=ROOT)
    nojit = dict(np.load(target))
    jit = compute_all()
    assert jit.keys() == nojit.keys()
    mismatched = []
    for key, a in jit.items():
        b = nojit[key]
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True):
            mismatched.append(key)
    assert not mismatched, f"JIT and pure-Python results differ: {mismatched}"
