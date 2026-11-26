"""
Oracle tests: pandas-ta-classic native (talib=False) vs a *frozen* tulipy oracle.

tulipy (last release 2020) is unmaintained and ships no wheels for CPython
>=3.12, so it cannot be installed across the full CI matrix.  Its output is
fully deterministic, so the arrays are snapshotted once into
``tests/fixtures/tulipy_oracle.json`` (see ``generate_tulipy_oracle.py``) and
compared here on every Python version — no tulipy install required.

Alignment
---------
tulipy returns front-trimmed arrays (no NaN prefix).  The golden file stores
only the tail of each array; every oracle array is aligned to the **tail** of
the full date index:

    tp_s = pd.Series(arr, index=self.idx[-len(arr):])

All comparisons then use the final <=2000 bars from the intersection of
non-NaN values in both series.

tulipy seeds EMA from the *first value* rather than from SMA(period).  For most
EMA-based indicators on this 5 241-row dataset, warm-up divergence washes out
over a long tail window.

Known formula/seeding differences are asserted explicitly (not skipped).

Regenerating the golden file
----------------------------
Run ``python -m tests.fixtures.generate_tulipy_oracle`` on CPython <3.12 with
tulipy installed, then commit the updated ``tulipy_oracle.json``.
"""

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.config import get_sample_data

_HERE = Path(__file__).parents[1]
_ORACLE_PATH = _HERE / "fixtures" / "tulipy_oracle.json"

ORACLE_AVAILABLE = _ORACLE_PATH.is_file()

# Tulip decides the phase with `abs(rp) > 0.001`, in price units, and falls
# back to +-pi; native msw uses the limit of arctan(ip / rp) instead (#346).
# On SPY_D, |rp| <= 0.001 on these two bars only (rp = -4.5e-4 and -2.8e-4).
_MSW_THRESHOLD_BARS = (3793, 4208)


@pytest.mark.skipif(not ORACLE_AVAILABLE, reason="tulipy_oracle.json fixture missing")
class TestTulipyOracle:
    """Compare pandas-ta-classic native output against the frozen tulipy oracle."""

    @classmethod
    def setup_class(cls):
        df = get_sample_data()
        cls.idx = df.index
        cls.open = df["open"]
        cls.high = df["high"]
        cls.low = df["low"]
        cls.close = df["close"]
        cls.vol = df["volume"]
        # Raw volume value used by the OBV offset assertion.
        cls.v0 = float(df["volume"].to_numpy(dtype=np.float64)[0])

        payload = json.loads(_ORACLE_PATH.read_text(encoding="utf-8"))
        # None (JSON null) -> NaN; arrays are the tail of each tulipy output.
        cls.oracle = {k: np.asarray(v, dtype=np.float64) for k, v in payload["arrays"].items()}

    @classmethod
    def teardown_class(cls):
        del cls.idx, cls.open, cls.high, cls.low, cls.close, cls.vol
        del cls.v0, cls.oracle

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _oracle(self, name):
        """Return the frozen tulipy array for *name*."""
        return self.oracle[name]

    def _compare(self, pt_series, tp_arr, tol=1e-11, name="", known=()):
        """Assert max abs diff <= tol on the last 2000 common bars.

        The frozen oracle array is tail-trimmed; align it to the tail of the
        full date index before comparing. *known* holds the positions of bars
        that differ by design; each must still differ, so the list cannot go stale.
        """
        arr = np.asarray(tp_arr)
        tp_s = pd.Series(arr, index=self.idx[-len(arr) :])
        both = pt_series.dropna().index.intersection(tp_s.index)
        assert len(both) > 50, f"{name}: too few common non-NaN values ({len(both)})"
        for label in self.idx[list(known)]:
            assert abs(pt_series.loc[label] - tp_s.loc[label]) > tol, f"{name}: known difference at {label} is gone"
        both = both.difference(self.idx[list(known)])
        tail = both[-min(2000, len(both)) :]
        diff = np.abs(pt_series.loc[tail].values - tp_s.loc[tail].values)
        assert diff.max() < tol, f"{name}: max abs diff {diff.max():.4e} exceeds {tol:.1e}"

    # ------------------------------------------------------------------
    # Comparisons that do not fit the _compare table below
    # ------------------------------------------------------------------

    def test_dpo(self):
        # DPO formulas match; compare non-NaN values directly to avoid index-shift
        # mismatch from centered lookahead output.
        pt = ta.dpo(self.close, length=20).dropna().to_numpy()
        tp = np.asarray(self._oracle("dpo"))
        n = min(len(pt), len(tp))
        assert n > 50, f"DPO: too few comparable values ({n})"
        diff = np.abs(pt[-n:] - tp[-n:])
        assert diff.max() < 1e-11, f"DPO: max abs diff {diff.max():.4e} exceeds 1e-11"

    def test_macd(self):
        pt = ta.macd(self.close)
        oracle = self._oracle("macd")
        tp_s = pd.Series(np.asarray(oracle), index=self.idx[-len(oracle) :])
        both = pt.iloc[:, 0].dropna().index.intersection(tp_s.index)
        assert len(both) > 50, f"MACD: too few common values ({len(both)})"
        diff = np.abs(pt.iloc[:, 0].loc[both].values - tp_s.loc[both].values)
        assert diff.max() > 0.1, f"MACD: expected seeding divergence not observed (max={diff.max():.4e})"

    def test_obv(self):
        pt = ta.obv(self.close, self.vol)
        tp_arr = self._oracle("obv")
        tp_s = pd.Series(np.asarray(tp_arr), index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        assert len(both) > 50, f"OBV: too few common values ({len(both)})"
        offset = pt.loc[both].values - tp_s.loc[both].values
        assert np.allclose(offset, offset[0]), "OBV: expected constant offset between pandas-ta and tulipy"
        assert round(abs(offset[0] - self.v0), 6) == 0, "OBV: unexpected offset magnitude"

    def test_dx(self):
        # A single outlier bar appears in the 1001-2000 tail window; the final
        # 500 bars are clean (diff 3.55e-14).  Use a custom 500-bar comparison.
        pt = ta.dx(self.high, self.low, self.close, length=14)
        tp_arr = self._oracle("dx")
        tp_s = pd.Series(tp_arr, index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        assert len(both) > 50, f"DX: too few common values ({len(both)})"
        tail = both[-min(500, len(both)) :]
        diff = np.abs(pt.loc[tail].values - tp_s.loc[tail].values)
        assert diff.max() < 5e-13, f"DX: max diff {diff.max():.4e} exceeds 5e-13"

    def test_stderr(self):
        """Tulip's STDERR divides the population deviation by sqrt(period): stderr with ddof=0.

        The default ddof=1 reads sqrt(20 / 19) higher. This test used to compare the median ratio
        with sqrt(19 / 18) to 0.005, which the sample deviation passed.
        """
        self._compare(ta.stderr(self.close, length=20, ddof=0), self._oracle("stderr"), tol=5e-10, name="STDERR")

    def test_msw_divergence_is_tulipys_pi(self):
        """With tulipy's pi constant the native loop reproduces the oracle exactly,
        except where Tulip's absolute threshold decides the phase."""
        msw_module = importlib.import_module("pandas_ta_classic.cycles.msw")
        sine, lead = msw_module._msw_native(self.close.to_numpy(dtype=float), 5, pi=3.1415926)
        known = _MSW_THRESHOLD_BARS
        self._compare(pd.Series(sine, index=self.close.index), self._oracle("msw_sine"), tol=1e-12, name="MSW_sine_tulipy_pi", known=known)
        self._compare(pd.Series(lead, index=self.close.index), self._oracle("msw_lead"), tol=1e-12, name="MSW_lead_tulipy_pi", known=known)


# ---------------------------------------------------------------------------
# Plain comparisons: one generated test_<key> per row.
#
# compute gets the test instance (its open/high/low/close/vol Series) and returns
# the native Series; oracle names the frozen tulipy array, which is multiplied by
# scale; tol and known go to _compare; note keeps what the row needs explaining
# (often the tulipy name) and becomes the test's docstring.
# ---------------------------------------------------------------------------


class _Row(NamedTuple):
    compute: Callable[[Any], pd.Series]
    oracle: str
    note: str = ""
    # Absolute. The default is 10x the largest difference of the rows that read tulipy to
    # 1e-12; the others set the 1-2-5 step above 10x theirs (Windows and Linux, Python
    # 3.11 to 3.15, numba on and off).
    tol: float = 1e-11
    scale: float = 1
    known: tuple[int, ...] = ()


_ROWS: dict[str, _Row] = {
    # Overlap / Moving averages
    "sma": _Row(lambda s: ta.sma(s.close, length=20), "sma"),
    "ema": _Row(lambda s: ta.ema(s.close, length=20, talib=False), "ema"),
    "wma": _Row(lambda s: ta.wma(s.close, length=20), "wma", tol=5e-10),
    "hma": _Row(lambda s: ta.hma(s.close, length=9), "hma", tol=5e-8),
    "zlma": _Row(lambda s: ta.zlma(s.close, length=20), "zlema", note="tulipy name: zlema"),
    "rma": _Row(lambda s: ta.rma(s.close, length=14), "wilders", note="tulipy name: wilders (Wilder's Moving Average)"),
    "dema": _Row(lambda s: ta.dema(s.close, length=20, talib=False), "dema"),
    "tema": _Row(lambda s: ta.tema(s.close, length=20, talib=False), "tema"),
    "kama": _Row(lambda s: ta.kama(s.close, length=10, talib=False), "kama"),
    "vwma": _Row(lambda s: ta.vwma(s.close, s.vol, length=20), "vwma", tol=2e-11),
    "wcp": _Row(lambda s: ta.wcp(s.high, s.low, s.close), "wcprice", note="tulipy name: wcprice (Weighted Close Price)"),
    "qstick": _Row(lambda s: ta.qstick(s.open, s.close, length=10), "qstick"),
    "trima": _Row(
        lambda s: ta.trima(s.close, length=20, talib=False),
        "trima",
        note="The native path (SMA(SMA(close, n/2 + 1), n/2) for even n) matches tulipy.",
        tol=5e-9,
    ),
    # Oscillators / Momentum
    "rsi": _Row(lambda s: ta.rsi(s.close, length=14, talib=False), "rsi"),
    "mom": _Row(lambda s: ta.mom(s.close, length=10), "mom"),
    "roc": _Row(
        lambda s: ta.roc(s.close, length=10),
        "roc",
        note="tulipy roc returns a decimal fraction; ta.roc returns percentage, so the oracle is scaled by 100.",
        scale=100,
    ),
    "rocr": _Row(lambda s: ta.rocr(s.close, length=10), "rocr"),
    "willr": _Row(lambda s: ta.willr(s.high, s.low, s.close, length=14), "willr"),
    "cci": _Row(lambda s: ta.cci(s.high, s.low, s.close, length=14), "cci", tol=5e-9),
    "bop": _Row(lambda s: ta.bop(s.open, s.high, s.low, s.close), "bop"),
    "mfi": _Row(lambda s: ta.mfi(s.high, s.low, s.close, s.vol, length=14), "mfi"),
    "cvi": _Row(lambda s: ta.cvi(s.high, s.low, length=10), "cvi"),
    "ao": _Row(lambda s: ta.ao(s.high, s.low), "ao", note="Awesome Oscillator", tol=2e-11),
    "vhf": _Row(lambda s: ta.vhf(s.close, length=28), "vhf"),
    "wad": _Row(lambda s: ta.wad(s.high, s.low, s.close), "wad"),
    "cmo": _Row(lambda s: ta.cmo(s.close, length=14, talib=False), "cmo"),
    "fosc": _Row(
        lambda s: ta.fosc(s.close, length=14),
        "fosc",
        note="Tulip forecasts this bar with the previous bar's TSF, as TA-Lib does; fosc used the same bar's TSF, which made it equal cfo, until #344.",
        tol=2e-9,
    ),
    "trix": _Row(
        lambda s: ta.trix(s.close, length=18).iloc[:, 0],
        "trix",
        note="EMA-seed divergence results in ~6.2e-3 diff; converges over the full series. tol=0.01 is sufficient for a 5241-bar dataset.",
        tol=0.01,
    ),
    "stochrsi": _Row(
        lambda s: ta.stochrsi(s.close, length=14, k=1, talib=False).iloc[:, 0],
        "stochrsi",
        note="The native unsmoothed %K (k=1) matches tulipy's single-period formula.",
        scale=100,
    ),
    # Stochastic
    "stoch_k": _Row(lambda s: ta.stoch(s.high, s.low, s.close, k=14, d=3, smooth_k=3).iloc[:, 0], "stoch_k"),
    "stoch_d": _Row(lambda s: ta.stoch(s.high, s.low, s.close, k=14, d=3, smooth_k=3).iloc[:, 1], "stoch_d", tol=2e-11),
    # Trend / Directional
    "aroon_down": _Row(lambda s: ta.aroon(s.high, s.low, length=14, talib=False).iloc[:, 0], "aroon_down"),
    "aroon_up": _Row(lambda s: ta.aroon(s.high, s.low, length=14, talib=False).iloc[:, 1], "aroon_up"),
    "aroonosc": _Row(lambda s: ta.aroon(s.high, s.low, length=14, talib=False).iloc[:, 2], "aroonosc"),
    # Volatility
    "atr": _Row(lambda s: ta.atr(s.high, s.low, s.close, length=14, talib=False), "atr"),
    "natr": _Row(
        lambda s: ta.natr(s.high, s.low, s.close, length=14, mamode="rma", talib=False),
        "natr",
        note="tulipy NATR uses Wilder's smoothing (RMA) for ATR internally; the native path must be invoked with mamode='rma' to match.",
    ),
    "bbands_lower": _Row(lambda s: ta.bbands(s.close, length=20).filter(regex=r"^BBL").iloc[:, 0], "bbands_lower", tol=2e-9),
    "bbands_upper": _Row(lambda s: ta.bbands(s.close, length=20).filter(regex=r"^BBU").iloc[:, 0], "bbands_upper", tol=2e-9),
    # Statistics
    "stdev": _Row(lambda s: ta.stdev(s.close, length=20, talib=False), "stddev", note="tulipy name: stddev", tol=1e-9),
    "variance": _Row(lambda s: ta.variance(s.close, length=20, talib=False), "var", note="tulipy name: var", tol=5e-9),
    "linreg": _Row(lambda s: ta.linreg(s.close, length=14, talib=False), "linreg", tol=5e-9),
    "tsf": _Row(lambda s: ta.tsf(s.close, length=14, talib=False), "tsf", tol=5e-9),
    "fosc_stat": _Row(lambda s: ta.fosc(s.close, length=14), "fosc", note="The same comparison as test_fosc; separate test kept for grouping.", tol=2e-9),
    # Fisher Transform
    "fisher_val": _Row(lambda s: ta.fisher(s.high, s.low, length=9).iloc[:, 0], "fisher_val"),
    "fisher_sig": _Row(lambda s: ta.fisher(s.high, s.low, length=9).iloc[:, 1], "fisher_sig"),
    # Market Swing Wave (MSW / Mesa Sine Wave)
    "msw_sine": _Row(
        lambda s: ta.msw(s.close, period=5).iloc[:, 0],
        "msw_sine",
        note="tulipy truncates pi to 3.1415926, so the exact native formula differs from the oracle by up to ~3e-4 (not numpy drift).",
        tol=1e-3,
        known=_MSW_THRESHOLD_BARS,
    ),
    "msw_lead": _Row(lambda s: ta.msw(s.close, period=5).iloc[:, 1], "msw_lead", tol=1e-3, known=_MSW_THRESHOLD_BARS),
    # Added with the tulipy wrapper layer
    "edecay": _Row(lambda s: ta.edecay(s.close, length=5), "edecay"),
    "emv": _Row(lambda s: ta.emv(s.high, s.low, s.vol), "emv", note="tulipy emv takes no period argument; computes a single-bar EMV value"),
    "md": _Row(lambda s: ta.md(s.close, length=5), "md"),
    "lag": _Row(lambda s: ta.lag(s.close, period=3), "lag"),
    "avgprice": _Row(lambda s: ta.avgprice(s.open, s.high, s.low, s.close), "avgprice"),
    "medprice": _Row(lambda s: ta.medprice(s.high, s.low), "medprice"),
    "typprice": _Row(lambda s: ta.typprice(s.high, s.low, s.close), "typprice"),
}


def _make_row_test(key: str, row: _Row) -> Callable[[TestTulipyOracle], None]:
    def test(self: TestTulipyOracle) -> None:
        self._compare(row.compute(self), self._oracle(row.oracle) * row.scale, tol=row.tol, name=key, known=row.known)

    test.__name__ = f"test_{key}"
    test.__doc__ = row.note or f"{key} against the frozen tulipy {row.oracle!r} array."
    return test


def _attach_row_tests() -> None:
    for key, row in _ROWS.items():
        setattr(TestTulipyOracle, f"test_{key}", _make_row_test(key, row))


_attach_row_tests()
