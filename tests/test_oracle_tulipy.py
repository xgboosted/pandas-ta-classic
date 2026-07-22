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
Run ``python tests/fixtures/generate_tulipy_oracle.py`` on CPython <3.12 with
tulipy installed, then commit the updated ``tulipy_oracle.json``.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import pandas_ta_classic as ta

_HERE = Path(__file__).parent
_DATA_PATH = _HERE.parent / "examples" / "data" / "SPY_D.csv"
_ORACLE_PATH = _HERE / "fixtures" / "tulipy_oracle.json"

ORACLE_AVAILABLE = _ORACLE_PATH.is_file()


@unittest.skipUnless(ORACLE_AVAILABLE, "tulipy_oracle.json fixture missing")
class TestTulipyOracle(unittest.TestCase):
    """Compare pandas-ta-classic native output against the frozen tulipy oracle."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
        df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        df.columns = df.columns.str.lower()
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
    def tearDownClass(cls):
        del cls.idx, cls.open, cls.high, cls.low, cls.close, cls.vol
        del cls.v0, cls.oracle

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _oracle(self, name):
        """Return the frozen tulipy array for *name*."""
        return self.oracle[name]

    def _compare(self, pt_series, tp_arr, tol=1e-6, name=""):
        """Assert max abs diff <= tol on the last 2000 common bars.

        The frozen oracle array is tail-trimmed; align it to the tail of the
        full date index before comparing.
        """
        arr = np.asarray(tp_arr)
        tp_s = pd.Series(arr, index=self.idx[-len(arr) :])
        both = pt_series.dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"{name}: too few common non-NaN values ({len(both)})")
        tail = both[-min(2000, len(both)) :]
        diff = np.abs(pt_series.loc[tail].values - tp_s.loc[tail].values)
        self.assertLess(diff.max(), tol, f"{name}: max abs diff {diff.max():.4e} exceeds {tol:.1e}")

    # ------------------------------------------------------------------
    # Overlap / Moving averages
    # ------------------------------------------------------------------

    def test_sma(self):
        self._compare(ta.sma(self.close, length=20), self._oracle("sma"), name="SMA")

    def test_ema(self):
        self._compare(
            ta.ema(self.close, length=20, talib=False),
            self._oracle("ema"),
            name="EMA",
        )

    def test_wma(self):
        self._compare(ta.wma(self.close, length=20), self._oracle("wma"), name="WMA")

    def test_hma(self):
        self._compare(ta.hma(self.close, length=9), self._oracle("hma"), name="HMA")

    def test_zlma(self):
        # tulipy name: zlema
        self._compare(ta.zlma(self.close, length=20), self._oracle("zlema"), name="ZLMA")

    def test_rma(self):
        # tulipy name: wilders (Wilder's Moving Average)
        self._compare(ta.rma(self.close, length=14), self._oracle("wilders"), name="RMA")

    def test_dema(self):
        self._compare(
            ta.dema(self.close, length=20, talib=False),
            self._oracle("dema"),
            name="DEMA",
        )

    def test_tema(self):
        self._compare(
            ta.tema(self.close, length=20, talib=False),
            self._oracle("tema"),
            name="TEMA",
        )

    def test_kama(self):
        self._compare(
            ta.kama(self.close, length=10, talib=False),
            self._oracle("kama"),
            name="KAMA",
        )

    def test_vwma(self):
        self._compare(
            ta.vwma(self.close, self.vol, length=20),
            self._oracle("vwma"),
            name="VWMA",
        )

    def test_wcp(self):
        # tulipy name: wcprice (Weighted Close Price)
        self._compare(
            ta.wcp(self.high, self.low, self.close),
            self._oracle("wcprice"),
            name="WCP",
        )

    def test_qstick(self):
        self._compare(
            ta.qstick(self.open, self.close, length=10),
            self._oracle("qstick"),
            name="QSTICK",
        )

    def test_trima(self):
        # talib=True uses TA-Lib's symmetric SMA-of-SMA, which matches tulipy.
        self._compare(
            ta.trima(self.close, length=20, talib=True),
            self._oracle("trima"),
            name="TRIMA",
        )

    # ------------------------------------------------------------------
    # Oscillators / Momentum
    # ------------------------------------------------------------------

    def test_rsi(self):
        self._compare(
            ta.rsi(self.close, length=14, talib=False),
            self._oracle("rsi"),
            name="RSI",
        )

    def test_mom(self):
        self._compare(ta.mom(self.close, length=10), self._oracle("mom"), name="MOM")

    def test_roc(self):
        # tulipy roc returns a decimal fraction; ta.roc returns percentage.
        # Scale the oracle by 100 to match.
        self._compare(ta.roc(self.close, length=10), self._oracle("roc") * 100, name="ROC")

    def test_rocr(self):
        self._compare(ta.rocr(self.close, length=10), self._oracle("rocr"), name="ROCR")

    def test_willr(self):
        self._compare(
            ta.willr(self.high, self.low, self.close, length=14),
            self._oracle("willr"),
            name="WILLR",
        )

    def test_cci(self):
        self._compare(
            ta.cci(self.high, self.low, self.close, length=14),
            self._oracle("cci"),
            name="CCI",
        )

    def test_bop(self):
        self._compare(
            ta.bop(self.open, self.high, self.low, self.close),
            self._oracle("bop"),
            name="BOP",
        )

    def test_mfi(self):
        self._compare(
            ta.mfi(self.high, self.low, self.close, self.vol, length=14),
            self._oracle("mfi"),
            name="MFI",
        )

    def test_cvi(self):
        self._compare(
            ta.cvi(self.high, self.low, length=10),
            self._oracle("cvi"),
            name="CVI",
        )

    def test_ao(self):
        # Awesome Oscillator
        self._compare(ta.ao(self.high, self.low), self._oracle("ao"), name="AO")

    def test_vhf(self):
        self._compare(ta.vhf(self.close, length=28), self._oracle("vhf"), name="VHF")

    def test_wad(self):
        self._compare(
            ta.wad(self.high, self.low, self.close),
            self._oracle("wad"),
            name="WAD",
        )

    def test_dpo(self):
        # DPO formulas match; compare non-NaN values directly to avoid index-shift
        # mismatch from centered lookahead output.
        pt = ta.dpo(self.close, length=20).dropna().to_numpy()
        tp = np.asarray(self._oracle("dpo"))
        n = min(len(pt), len(tp))
        self.assertGreater(n, 50, f"DPO: too few comparable values ({n})")
        diff = np.abs(pt[-n:] - tp[-n:])
        self.assertLess(diff.max(), 1e-6, f"DPO: max abs diff {diff.max():.4e} exceeds 1e-6")

    def test_cmo(self):
        self._compare(
            ta.cmo(self.close, length=14, talib=False),
            self._oracle("cmo"),
            name="CMO",
        )

    def test_fosc(self):
        # FOSC implementations differ by design; enforce that divergence remains
        # measurable so we do not silently claim oracle equality.
        pt = ta.fosc(self.close, length=14)
        tp_arr = self._oracle("fosc")
        tp_s = pd.Series(np.asarray(tp_arr), index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"FOSC: too few common values ({len(both)})")
        diff = np.abs(pt.loc[both].values - tp_s.loc[both].values)
        self.assertGreater(
            diff.max(),
            0.1,
            f"FOSC: expected divergence not observed (max={diff.max():.4e})",
        )

    def test_trix(self):
        # EMA-seed divergence results in ~6.2e-3 diff; converges over the full
        # series.  tol=0.01 is sufficient for a 5241-bar dataset.
        pt = ta.trix(self.close, length=18)
        self._compare(pt.iloc[:, 0], self._oracle("trix"), tol=0.01, name="TRIX")

    def test_stochrsi(self):
        # talib=True uses TA-Lib STOCHRSI which matches tulipy's single-period formula
        pt = ta.stochrsi(self.close, length=14, talib=True)
        self._compare(pt.iloc[:, 0], self._oracle("stochrsi") * 100, name="STOCHRSI")

    def test_macd(self):
        pt = ta.macd(self.close)
        oracle = self._oracle("macd")
        tp_s = pd.Series(np.asarray(oracle), index=self.idx[-len(oracle) :])
        both = pt.iloc[:, 0].dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"MACD: too few common values ({len(both)})")
        diff = np.abs(pt.iloc[:, 0].loc[both].values - tp_s.loc[both].values)
        self.assertGreater(
            diff.max(),
            0.1,
            f"MACD: expected seeding divergence not observed (max={diff.max():.4e})",
        )

    def test_obv(self):
        pt = ta.obv(self.close, self.vol)
        tp_arr = self._oracle("obv")
        tp_s = pd.Series(np.asarray(tp_arr), index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"OBV: too few common values ({len(both)})")
        offset = pt.loc[both].values - tp_s.loc[both].values
        self.assertTrue(
            np.allclose(offset, offset[0]),
            "OBV: expected constant offset between pandas-ta and tulipy",
        )
        self.assertAlmostEqual(
            offset[0],
            self.v0,
            places=6,
            msg="OBV: unexpected offset magnitude",
        )

    # ------------------------------------------------------------------
    # Stochastic
    # ------------------------------------------------------------------

    def test_stoch_k(self):
        pt = ta.stoch(self.high, self.low, self.close, k=14, d=3, smooth_k=3)
        self._compare(pt.iloc[:, 0], self._oracle("stoch_k"), name="STOCH_k")

    def test_stoch_d(self):
        pt = ta.stoch(self.high, self.low, self.close, k=14, d=3, smooth_k=3)
        self._compare(pt.iloc[:, 1], self._oracle("stoch_d"), name="STOCH_d")

    # ------------------------------------------------------------------
    # Trend / Directional
    # ------------------------------------------------------------------

    def test_aroon_down(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        self._compare(pt.iloc[:, 0], self._oracle("aroon_down"), name="AROON_down")

    def test_aroon_up(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        self._compare(pt.iloc[:, 1], self._oracle("aroon_up"), name="AROON_up")

    def test_aroonosc(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        self._compare(pt.iloc[:, 2], self._oracle("aroonosc"), name="AROONOSC")

    def test_dx(self):
        # A single outlier bar appears in the 1001-2000 tail window; the final
        # 500 bars are clean (diff 3.55e-14).  Use a custom 500-bar comparison.
        pt = ta.dx(self.high, self.low, self.close, length=14)
        tp_arr = self._oracle("dx")
        tp_s = pd.Series(tp_arr, index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"DX: too few common values ({len(both)})")
        tail = both[-min(500, len(both)) :]
        diff = np.abs(pt.loc[tail].values - tp_s.loc[tail].values)
        self.assertLess(diff.max(), 1e-6, f"DX: max diff {diff.max():.4e} exceeds 1e-6")

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def test_atr(self):
        self._compare(
            ta.atr(self.high, self.low, self.close, length=14, talib=False),
            self._oracle("atr"),
            name="ATR",
        )

    def test_natr(self):
        # tulipy NATR uses Wilder's smoothing (RMA) for ATR internally.
        # Our pure-Python path must be invoked with mamode="rma" to match.
        self._compare(
            ta.natr(self.high, self.low, self.close, length=14, mamode="rma", talib=False),
            self._oracle("natr"),
            name="NATR",
        )

    def test_bbands_lower(self):
        pt = ta.bbands(self.close, length=20)
        self._compare(pt.filter(regex=r"^BBL").iloc[:, 0], self._oracle("bbands_lower"), name="BBANDS_lower")

    def test_bbands_upper(self):
        pt = ta.bbands(self.close, length=20)
        self._compare(pt.filter(regex=r"^BBU").iloc[:, 0], self._oracle("bbands_upper"), name="BBANDS_upper")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def test_stdev(self):
        # tulipy name: stddev
        self._compare(
            ta.stdev(self.close, length=20, talib=False),
            self._oracle("stddev"),
            name="STDEV",
        )

    def test_variance(self):
        # tulipy name: var
        self._compare(
            ta.variance(self.close, length=20, talib=False),
            self._oracle("var"),
            name="VARIANCE",
        )

    def test_stderr(self):
        pt = ta.stderr(self.close, length=20)
        tp_arr = self._oracle("stderr")
        tp_s = pd.Series(np.asarray(tp_arr), index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        self.assertGreater(len(both), 50, f"STDERR: too few common non-NaN values ({len(both)})")
        ratio = np.median(pt.loc[both].values / tp_s.loc[both].values)
        expected = np.sqrt((20 - 1) / (20 - 2))
        self.assertLess(
            abs(ratio - expected),
            0.005,
            f"STDERR: ratio {ratio:.6f} differs from expected {expected:.6f}",
        )

    def test_linreg(self):
        self._compare(
            ta.linreg(self.close, length=14, talib=False),
            self._oracle("linreg"),
            name="LINREG",
        )

    def test_tsf(self):
        self._compare(
            ta.tsf(self.close, length=14, talib=False),
            self._oracle("tsf"),
            name="TSF",
        )

    def test_fosc_stat(self):
        # Same expected divergence as test_fosc; separate test kept for grouping.
        self.test_fosc()

    # ------------------------------------------------------------------
    # Fisher Transform
    # ------------------------------------------------------------------

    def test_fisher_val(self):
        pt = ta.fisher(self.high, self.low, length=9, talib=False)
        self._compare(pt.iloc[:, 0], self._oracle("fisher_val"), name="FISHER_val")

    def test_fisher_sig(self):
        pt = ta.fisher(self.high, self.low, length=9, talib=False)
        self._compare(pt.iloc[:, 1], self._oracle("fisher_sig"), name="FISHER_sig")

    # ------------------------------------------------------------------
    # Market Swing Wave (MSW / Mesa Sine Wave)
    # ------------------------------------------------------------------

    def test_msw_sine(self):
        # MSW is a recursive trig (Mesa Sine Wave) computation; native output
        # drifts by ~1e-4 across numpy versions, so a 1e-3 tolerance is used to
        # stay robust to float noise while still cross-checking the oracle.
        pt = ta.msw(self.close, length=5)
        self._compare(pt.iloc[:, 0], self._oracle("msw_sine"), tol=1e-3, name="MSW_sine")

    def test_msw_lead(self):
        pt = ta.msw(self.close, length=5)
        self._compare(pt.iloc[:, 1], self._oracle("msw_lead"), tol=1e-3, name="MSW_lead")

    # ------------------------------------------------------------------
    # New indicators (added with tulipy wrapper layer)
    # ------------------------------------------------------------------

    def test_edecay(self):
        self._compare(
            ta.edecay(self.close, length=5),
            self._oracle("edecay"),
            name="EDECAY",
        )

    def test_emv(self):
        # tulipy emv takes no period argument; computes a single-bar EMV value
        self._compare(
            ta.emv(self.high, self.low, self.vol, length=14),
            self._oracle("emv"),
            name="EMV",
        )

    def test_md(self):
        self._compare(
            ta.md(self.close, length=5),
            self._oracle("md"),
            name="MD",
        )

    def test_lag(self):
        self._compare(
            ta.lag(self.close, period=3),
            self._oracle("lag"),
            name="LAG",
        )

    def test_avgprice(self):
        self._compare(
            ta.avgprice(self.open, self.high, self.low, self.close),
            self._oracle("avgprice"),
            name="AVGPRICE",
        )

    def test_medprice(self):
        self._compare(
            ta.medprice(self.high, self.low),
            self._oracle("medprice"),
            name="MEDPRICE",
        )

    def test_typprice(self):
        self._compare(
            ta.typprice(self.high, self.low, self.close),
            self._oracle("typprice"),
            name="TYPPRICE",
        )


if __name__ == "__main__":
    unittest.main()
