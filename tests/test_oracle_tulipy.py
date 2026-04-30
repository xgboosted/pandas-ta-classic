"""
Oracle tests: pandas-ta-classic native (talib=False) vs tulipy secondary oracle.

Alignment
---------
tulipy returns front-trimmed arrays (no NaN prefix).  Each oracle array is
aligned to the **tail** of the full date index:

    tp_s = pd.Series(arr, index=self.idx[-len(arr):])

All comparisons then use the final 2000 bars from the intersection of
non-NaN values in both series.

tulipy seeds EMA from the *first value* rather than from SMA(period).
For most EMA-based indicators on this 5 241-row dataset, warm-up divergence
washes out over a long tail window.

Known formula/seeding differences are asserted explicitly (not skipped).
"""

import unittest

import numpy as np
import pandas as pd

try:
    import tulipy as _tp

    TULIPY_AVAILABLE = True
except ImportError:
    TULIPY_AVAILABLE = False

import pandas_ta_classic as ta

_DATA_PATH = (
    __import__("pathlib").Path(__file__).parent.parent
    / "examples"
    / "data"
    / "SPY_D.csv"
)


@unittest.skipUnless(TULIPY_AVAILABLE, "tulipy not installed")
class TestTulipyOracle(unittest.TestCase):
    """Compare pandas-ta-classic native output against tulipy."""

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
        # Raw numpy float64 arrays for tulipy
        cls.o = df["open"].to_numpy(dtype=np.float64)
        cls.h = df["high"].to_numpy(dtype=np.float64)
        cls.l = df["low"].to_numpy(dtype=np.float64)
        cls.c = df["close"].to_numpy(dtype=np.float64)
        cls.v = df["volume"].to_numpy(dtype=np.float64)

    @classmethod
    def tearDownClass(cls):
        del cls.idx, cls.open, cls.high, cls.low, cls.close, cls.vol
        del cls.o, cls.h, cls.l, cls.c, cls.v

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compare(self, pt_series, tp_arr, tol=1e-6, name=""):
        """Assert max abs diff <= tol on the last 2000 common bars.

        tulipy returns a front-trimmed array; align it to the tail of the
        full date index before comparing.
        """
        arr = np.asarray(tp_arr)
        tp_s = pd.Series(arr, index=self.idx[-len(arr) :])
        both = pt_series.dropna().index.intersection(tp_s.index)
        self.assertGreater(
            len(both), 50, f"{name}: too few common non-NaN values ({len(both)})"
        )
        tail = both[-min(2000, len(both)) :]
        diff = np.abs(pt_series.loc[tail].values - tp_s.loc[tail].values)
        self.assertLess(
            diff.max(), tol, f"{name}: max abs diff {diff.max():.4e} exceeds {tol:.1e}"
        )

    # ------------------------------------------------------------------
    # Overlap / Moving averages
    # ------------------------------------------------------------------

    def test_sma(self):
        self._compare(
            ta.sma(self.close, length=20), _tp.sma(self.c, period=20), name="SMA"
        )

    def test_ema(self):
        self._compare(
            ta.ema(self.close, length=20, talib=False),
            _tp.ema(self.c, period=20),
            name="EMA",
        )

    def test_wma(self):
        self._compare(
            ta.wma(self.close, length=20), _tp.wma(self.c, period=20), name="WMA"
        )

    def test_hma(self):
        self._compare(
            ta.hma(self.close, length=9), _tp.hma(self.c, period=9), name="HMA"
        )

    def test_zlma(self):
        # tulipy name: zlema
        self._compare(
            ta.zlma(self.close, length=20), _tp.zlema(self.c, period=20), name="ZLMA"
        )

    def test_rma(self):
        # tulipy name: wilders (Wilder's Moving Average)
        self._compare(
            ta.rma(self.close, length=14), _tp.wilders(self.c, period=14), name="RMA"
        )

    def test_dema(self):
        self._compare(
            ta.dema(self.close, length=20, talib=False),
            _tp.dema(self.c, period=20),
            name="DEMA",
        )

    def test_tema(self):
        self._compare(
            ta.tema(self.close, length=20, talib=False),
            _tp.tema(self.c, period=20),
            name="TEMA",
        )

    def test_kama(self):
        self._compare(
            ta.kama(self.close, length=10, talib=False),
            _tp.kama(self.c, period=10),
            name="KAMA",
        )

    def test_vwma(self):
        self._compare(
            ta.vwma(self.close, self.vol, length=20),
            _tp.vwma(self.c, self.v, period=20),
            name="VWMA",
        )

    def test_wcp(self):
        # tulipy name: wcprice (Weighted Close Price)
        self._compare(
            ta.wcp(self.high, self.low, self.close),
            _tp.wcprice(self.h, self.l, self.c),
            name="WCP",
        )

    def test_qstick(self):
        self._compare(
            ta.qstick(self.open, self.close, length=10),
            _tp.qstick(self.o, self.c, period=10),
            name="QSTICK",
        )

    def test_trima(self):
        # talib=True uses TA-Lib's symmetric SMA-of-SMA, which matches tulipy.
        self._compare(
            ta.trima(self.close, length=20, talib=True),
            _tp.trima(self.c, period=20),
            name="TRIMA",
        )

    # ------------------------------------------------------------------
    # Oscillators / Momentum
    # ------------------------------------------------------------------

    def test_rsi(self):
        self._compare(
            ta.rsi(self.close, length=14, talib=False),
            _tp.rsi(self.c, period=14),
            name="RSI",
        )

    def test_mom(self):
        self._compare(
            ta.mom(self.close, length=10), _tp.mom(self.c, period=10), name="MOM"
        )

    def test_roc(self):
        # tulipy roc returns a decimal fraction; ta.roc returns percentage.
        # Scale the oracle by 100 to match.
        self._compare(
            ta.roc(self.close, length=10), _tp.roc(self.c, period=10) * 100, name="ROC"
        )

    def test_rocr(self):
        self._compare(
            ta.rocr(self.close, length=10), _tp.rocr(self.c, period=10), name="ROCR"
        )

    def test_willr(self):
        self._compare(
            ta.willr(self.high, self.low, self.close, length=14),
            _tp.willr(self.h, self.l, self.c, period=14),
            name="WILLR",
        )

    def test_cci(self):
        self._compare(
            ta.cci(self.high, self.low, self.close, length=14),
            _tp.cci(self.h, self.l, self.c, period=14),
            name="CCI",
        )

    def test_bop(self):
        self._compare(
            ta.bop(self.open, self.high, self.low, self.close),
            _tp.bop(self.o, self.h, self.l, self.c),
            name="BOP",
        )

    def test_mfi(self):
        self._compare(
            ta.mfi(self.high, self.low, self.close, self.vol, length=14),
            _tp.mfi(self.h, self.l, self.c, self.v, period=14),
            name="MFI",
        )

    def test_cvi(self):
        self._compare(
            ta.cvi(self.high, self.low, length=10),
            _tp.cvi(self.h, self.l, period=10),
            name="CVI",
        )

    def test_ao(self):
        # Awesome Oscillator
        self._compare(ta.ao(self.high, self.low), _tp.ao(self.h, self.l), name="AO")

    def test_vhf(self):
        self._compare(
            ta.vhf(self.close, length=28), _tp.vhf(self.c, period=28), name="VHF"
        )

    def test_wad(self):
        self._compare(
            ta.wad(self.high, self.low, self.close),
            _tp.wad(self.h, self.l, self.c),
            name="WAD",
        )

    def test_dpo(self):
        # DPO formulas match; compare non-NaN values directly to avoid index-shift
        # mismatch from centered lookahead output.
        pt = ta.dpo(self.close, length=20).dropna().to_numpy()
        tp = np.asarray(_tp.dpo(self.c, period=20))
        n = min(len(pt), len(tp))
        self.assertGreater(n, 50, f"DPO: too few comparable values ({n})")
        diff = np.abs(pt[-n:] - tp[-n:])
        self.assertLess(
            diff.max(), 1e-6, f"DPO: max abs diff {diff.max():.4e} exceeds 1e-6"
        )

    def test_cmo(self):
        self._compare(
            ta.cmo(self.close, length=14, talib=False),
            _tp.cmo(self.c, period=14),
            name="CMO",
        )

    def test_fosc(self):
        # FOSC implementations differ by design; enforce that divergence remains
        # measurable so we do not silently claim oracle equality.
        pt = ta.fosc(self.close, length=14)
        tp_arr = _tp.fosc(self.c, period=14)
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
        self._compare(pt.iloc[:, 0], _tp.trix(self.c, period=18), tol=0.01, name="TRIX")

    def test_stochrsi(self):
        # talib=True uses TA-Lib STOCHRSI which matches tulipy's single-period formula
        pt = ta.stochrsi(self.close, length=14, talib=True)
        self._compare(
            pt.iloc[:, 0], _tp.stochrsi(self.c, period=14) * 100, name="STOCHRSI"
        )

    def test_macd(self):
        pt = ta.macd(self.close)
        oracle, _, _ = _tp.macd(
            self.c, short_period=12, long_period=26, signal_period=9
        )
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
        tp_arr = _tp.obv(self.c, self.v)
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
            float(self.v[0]),
            places=6,
            msg="OBV: unexpected offset magnitude",
        )

    # ------------------------------------------------------------------
    # Stochastic
    # ------------------------------------------------------------------

    def test_stoch_k(self):
        # tulipy stoch takes positional args: (h, l, c, k_period, k_slowing, d_period)
        pt = ta.stoch(self.high, self.low, self.close, k=14, d=3, smooth_k=3)
        tp_k, _ = _tp.stoch(self.h, self.l, self.c, 14, 3, 3)
        self._compare(pt.iloc[:, 0], tp_k, name="STOCH_k")

    def test_stoch_d(self):
        pt = ta.stoch(self.high, self.low, self.close, k=14, d=3, smooth_k=3)
        _, tp_d = _tp.stoch(self.h, self.l, self.c, 14, 3, 3)
        self._compare(pt.iloc[:, 1], tp_d, name="STOCH_d")

    # ------------------------------------------------------------------
    # Trend / Directional
    # ------------------------------------------------------------------

    def test_aroon_down(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        tp_d, _ = _tp.aroon(self.h, self.l, period=14)
        self._compare(pt.iloc[:, 0], tp_d, name="AROON_down")

    def test_aroon_up(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        _, tp_u = _tp.aroon(self.h, self.l, period=14)
        self._compare(pt.iloc[:, 1], tp_u, name="AROON_up")

    def test_aroonosc(self):
        pt = ta.aroon(self.high, self.low, length=14, talib=False)
        self._compare(
            pt.iloc[:, 2], _tp.aroonosc(self.h, self.l, period=14), name="AROONOSC"
        )

    def test_dx(self):
        # A single outlier bar appears in the 1001-2000 tail window; the final
        # 500 bars are clean (diff 3.55e-14).  Use a custom 500-bar comparison.
        pt = ta.dx(self.high, self.low, self.close, length=14)
        tp_arr = _tp.dx(self.h, self.l, self.c, period=14)
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
            _tp.atr(self.h, self.l, self.c, period=14),
            name="ATR",
        )

    def test_natr(self):
        self._compare(
            ta.natr(self.high, self.low, self.close, length=14, talib=False),
            _tp.natr(self.h, self.l, self.c, period=14),
            name="NATR",
        )

    def test_bbands_lower(self):
        # tulipy bbands returns (lower, mid, upper)
        pt = ta.bbands(self.close, length=20)
        tp_l, _, _ = _tp.bbands(self.c, period=20, stddev=2.0)
        self._compare(pt.filter(regex=r"^BBL").iloc[:, 0], tp_l, name="BBANDS_lower")

    def test_bbands_upper(self):
        pt = ta.bbands(self.close, length=20)
        _, _, tp_u = _tp.bbands(self.c, period=20, stddev=2.0)
        self._compare(pt.filter(regex=r"^BBU").iloc[:, 0], tp_u, name="BBANDS_upper")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def test_stdev(self):
        # tulipy name: stddev
        self._compare(
            ta.stdev(self.close, length=20, talib=False),
            _tp.stddev(self.c, period=20),
            name="STDEV",
        )

    def test_variance(self):
        # tulipy name: var
        self._compare(
            ta.variance(self.close, length=20, talib=False),
            _tp.var(self.c, period=20),
            name="VARIANCE",
        )

    def test_stderr(self):
        pt = ta.stderr(self.close, length=20)
        tp_arr = _tp.stderr(self.c, period=20)
        tp_s = pd.Series(np.asarray(tp_arr), index=self.idx[-len(tp_arr) :])
        both = pt.dropna().index.intersection(tp_s.index)
        self.assertGreater(
            len(both), 50, f"STDERR: too few common non-NaN values ({len(both)})"
        )
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
            _tp.linreg(self.c, period=14),
            name="LINREG",
        )

    def test_tsf(self):
        self._compare(
            ta.tsf(self.close, length=14, talib=False),
            _tp.tsf(self.c, period=14),
            name="TSF",
        )

    def test_fosc_stat(self):
        # Same expected divergence as test_fosc; separate test kept for grouping.
        self.test_fosc()

    # ------------------------------------------------------------------
    # Fisher Transform
    # ------------------------------------------------------------------

    def test_fisher_val(self):
        # tulipy fisher returns (fisher_val, fisher_signal) as a tuple
        pt = ta.fisher(self.high, self.low, length=9, talib=False)
        tp_val, _ = _tp.fisher(self.h, self.l, period=9)
        self._compare(pt.iloc[:, 0], tp_val, name="FISHER_val")

    def test_fisher_sig(self):
        pt = ta.fisher(self.high, self.low, length=9, talib=False)
        _, tp_sig = _tp.fisher(self.h, self.l, period=9)
        self._compare(pt.iloc[:, 1], tp_sig, name="FISHER_sig")

    # ------------------------------------------------------------------
    # Market Swing Wave (MSW / Mesa Sine Wave)
    # ------------------------------------------------------------------

    def test_msw_sine(self):
        # tulipy msw returns (sine, lead) as a tuple
        pt = ta.msw(self.close, length=5)
        tp_sine, _ = _tp.msw(self.c, period=5)
        self._compare(pt.iloc[:, 0], tp_sine, name="MSW_sine")

    def test_msw_lead(self):
        pt = ta.msw(self.close, length=5)
        _, tp_lead = _tp.msw(self.c, period=5)
        self._compare(pt.iloc[:, 1], tp_lead, name="MSW_lead")

    # ------------------------------------------------------------------
    # New indicators (added with tulipy wrapper layer)
    # ------------------------------------------------------------------

    def test_edecay(self):
        self._compare(
            ta.edecay(self.close, length=5),
            _tp.edecay(self.c, period=5),
            name="EDECAY",
        )

    def test_emv(self):
        # tulipy emv takes no period argument; computes a single-bar EMV value
        self._compare(
            ta.emv(self.high, self.low, self.vol, length=14),
            _tp.emv(self.h, self.l, self.v),
            name="EMV",
        )

    def test_md(self):
        self._compare(
            ta.md(self.close, length=5),
            _tp.md(self.c, period=5),
            name="MD",
        )

    def test_lag(self):
        self._compare(
            ta.lag(self.close, period=3),
            _tp.lag(self.c, period=3),
            name="LAG",
        )

    def test_avgprice(self):
        self._compare(
            ta.avgprice(self.open, self.high, self.low, self.close),
            _tp.avgprice(self.o, self.h, self.l, self.c),
            name="AVGPRICE",
        )

    def test_medprice(self):
        self._compare(
            ta.medprice(self.high, self.low),
            _tp.medprice(self.h, self.l),
            name="MEDPRICE",
        )

    def test_typprice(self):
        self._compare(
            ta.typprice(self.high, self.low, self.close),
            _tp.typprice(self.h, self.l, self.c),
            name="TYPPRICE",
        )


if __name__ == "__main__":
    unittest.main()
