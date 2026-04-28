"""
Priority 1 — Golden fixture tests.

Each test asserts two things for every column of every tracked indicator:
  1. The last non-NaN value matches the stored golden value within a relative
     tolerance of 1e-4 (0.01 %).
  2. The number of non-NaN rows matches the stored count exactly.

To regenerate fixtures after an intentional algorithm change run:
    python -m tests.fixtures.generate_fixtures
"""

import json
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Load fixtures & sample data
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "expected_values.json"
with open(_FIXTURE_PATH) as _fh:
    _FIXTURES: dict[str, dict] = json.load(_fh)

_DATA_PATH = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# Helper — recompute every indicator from the same definitions used in
# generate_fixtures.py so the two files stay in sync.
# ---------------------------------------------------------------------------


def _compute_all(df: pd.DataFrame) -> dict[str, object]:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    _ema_fast = ta.ema(c, length=8, talib=False)
    _ema_slow = ta.ema(c, length=21, talib=False)
    _trend_bool = _ema_fast > _ema_slow
    _rsi_14_sig = ta.rsi(c, length=14, talib=False)
    return {
        # ---- Overlap (TA-Lib backed) ----
        "sma_20": ta.sma(c, length=20),
        "ema_20": ta.ema(c, length=20, talib=False),
        "dema_10": ta.dema(c, length=10, talib=False),
        "tema_10": ta.tema(c, length=10, talib=False),
        "wma_10": ta.wma(c, length=10, talib=False),
        "midpoint_14": ta.midpoint(c, 14, talib=False),
        "midprice_14": ta.midprice(h, l, 14, talib=False),
        "t3_10": ta.t3(c, 10, talib=False),
        "wcp": ta.wcp(h, l, c, talib=False),
        "hl2": ta.hl2(h, l),
        "hlc3": ta.hlc3(h, l, c, talib=False),
        "ohlc4": ta.ohlc4(o, h, l, c),
        # ---- Overlap (native) ----
        "hma_10": ta.hma(c, length=10),
        "alma_10": ta.alma(c, length=10),
        "trima_10": ta.trima(c, length=10, talib=False),
        "fwma_10": ta.fwma(c, length=10),
        "hilo": ta.hilo(h, l, c),
        "hwma": ta.hwma(c),
        "jma": ta.jma(c),
        "kama": ta.kama(c),
        "linreg_14": ta.linreg(c, length=14),
        "mama": ta.mama(c),
        "mcgd": ta.mcgd(c),
        "mmar": ta.mmar(c),
        "pwma_10": ta.pwma(c, length=10),
        "rainbow": ta.rainbow(c),
        "rma": ta.rma(c),
        "sinwma_14": ta.sinwma(c, length=14),
        "ssf_10": ta.ssf(c, length=10),
        "supertrend": ta.supertrend(h, l, c),
        "swma": ta.swma(c),
        "tsf_14": ta.tsf(c, length=14),
        "vidya": ta.vidya(c),
        "vwap": ta.vwap(h, l, c, v),
        "vwma_10": ta.vwma(c, v, length=10),
        "zlma_10": ta.zlma(c, length=10),
        # ---- Momentum (TA-Lib backed) ----
        "rsi_14": ta.rsi(c, length=14, talib=False),
        "macd_12_26_9": ta.macd(c, fast=12, slow=26, signal=9, talib=False),
        "stoch": ta.stoch(h, l, c),
        "cci_14": ta.cci(h, l, c, length=14, talib=False),
        "roc_10": ta.roc(c, length=10, talib=False),
        "willr_14": ta.willr(h, l, c, length=14, talib=False),
        "apo": ta.apo(c, 12, 26, talib=False),
        "bop": ta.bop(o, h, l, c, talib=False),
        "mom_10": ta.mom(c, length=10, talib=False),
        "uo": ta.uo(h, l, c, talib=False),
        # ---- Momentum (native) ----
        "ao": ta.ao(h, l),
        "bias_10": ta.bias(c, length=10),
        "brar": ta.brar(o, h, l, c),
        "cfo_10": ta.cfo(c, length=10),
        "cg": ta.cg(c),
        "cmo_14": ta.cmo(c, length=14, talib=False),
        "coppock": ta.coppock(c),
        "cti": ta.cti(c),
        "dm_14": ta.dm(h, l, length=14, talib=False),
        "er": ta.er(c),
        "eri": ta.eri(h, l, c),
        "fisher": ta.fisher(h, l),
        "inertia": ta.inertia(c, h, l),
        "kdj": ta.kdj(h, l, c),
        "kst": ta.kst(c),
        "lrsi": ta.lrsi(c),
        "pgo_14": ta.pgo(h, l, c, length=14),
        "po": ta.po(c),
        "ppo": ta.ppo(c),
        "psl": ta.psl(c, o),
        "pvo": ta.pvo(v),
        "qqe": ta.qqe(c),
        "rsx": ta.rsx(c),
        "rvgi": ta.rvgi(o, h, l, c),
        "slope": ta.slope(c),
        "smi": ta.smi(c),
        "squeeze": ta.squeeze(h, l, c),
        "squeeze_pro": ta.squeeze_pro(h, l, c),
        "stc": ta.stc(c),
        "stochrsi": ta.stochrsi(c),
        "td_seq": ta.td_seq(c),
        "trix": ta.trix(c),
        "trixh": ta.trixh(c),
        "tsi": ta.tsi(c),
        "vwmacd": ta.vwmacd(c, v),
        # ---- Volatility (TA-Lib backed) ----
        "atr_14": ta.atr(h, l, c, length=14, talib=False),
        "bbands_20": ta.bbands(c, length=20, talib=False),
        "natr_14": ta.natr(h, l, c, length=14, talib=False),
        "true_range": ta.true_range(h, l, c, talib=False),
        # ---- Volatility (native) ----
        "donchian_20": ta.donchian(h, l, lower_length=20, upper_length=20),
        "kc_20": ta.kc(h, l, c, length=20),
        "aberration": ta.aberration(h, l, c),
        "accbands": ta.accbands(h, l, c),
        "ce": ta.ce(h, l, c),
        "hwc": ta.hwc(c),
        "massi": ta.massi(h, l),
        "pdist": ta.pdist(o, h, l, c),
        "rvi_vol": ta.rvi(c, h, l),
        "thermo": ta.thermo(h, l),
        "ui": ta.ui(c),
        # ---- Trend (TA-Lib backed) ----
        "adx_14": ta.adx(h, l, c, length=14, talib=False),
        "aroon_14": ta.aroon(h, l, length=14, talib=False),
        # ---- Trend (native) ----
        "psar": ta.psar(h, l, c),
        "decreasing": ta.decreasing(c),
        "increasing": ta.increasing(c),
        "adxr": ta.adxr(h, l, c, length=14),
        "amat": ta.amat(c, fast=8, slow=21),
        "chop": ta.chop(h, l, c),
        "cksp": ta.cksp(h, l, c),
        "cpr": ta.cpr(o, h, l, c),
        "decay": ta.decay(c),
        "dpo_14": ta.dpo(c, length=14),
        "long_run": ta.long_run(_ema_fast, _ema_slow),
        "pmax": ta.pmax(h, l, c),
        "qstick": ta.qstick(o, c),
        "short_run": ta.short_run(_ema_fast, _ema_slow),
        "ttm_trend": ta.ttm_trend(h, l, c),
        "vhf": ta.vhf(c),
        "vortex": ta.vortex(h, l, c),
        # ---- Volume (TA-Lib backed) ----
        "obv": ta.obv(c, v, talib=False),
        "mfi_14": ta.mfi(h, l, c, v, length=14, talib=False),
        "ad": ta.ad(h, l, c, v, talib=False),
        "adosc": ta.adosc(h, l, c, v, talib=False),
        # ---- Volume (native) ----
        "cmf_20": ta.cmf(h, l, c, v, length=20),
        "aobv": ta.aobv(c, v),
        "efi": ta.efi(c, v),
        "eom": ta.eom(h, l, c, v),
        "kvo": ta.kvo(h, l, c, v),
        "nvi": ta.nvi(c, v),
        "pvi": ta.pvi(c, v),
        "pvol": ta.pvol(c, v),
        "pvr": ta.pvr(c, v),
        "pvt": ta.pvt(c, v),
        "vfi": ta.vfi(c, v),
        # ---- Statistics (TA-Lib backed) ----
        "stdev_20": ta.stdev(c, length=20, talib=False),
        "variance_20": ta.variance(c, length=20, talib=False),
        # ---- Statistics (native) ----
        "zscore_20": ta.zscore(c, length=20),
        "kurtosis_20": ta.kurtosis(c, length=20),
        "skew_20": ta.skew(c, length=20),
        "beta": ta.beta(c, o),
        "correl": ta.correl(c, o),
        "entropy_10": ta.entropy(c, length=10),
        "mad_10": ta.mad(c, length=10),
        "median_14": ta.median(c, length=14),
        "quantile_14": ta.quantile(c, length=14),
        "tos_stdevall": ta.tos_stdevall(c, length=30),
        # ---- Cycles (TA-Lib backed) ----
        "ht_dcperiod": ta.ht_dcperiod(c),
        "ht_dcphase": ta.ht_dcphase(c),
        "ht_phasor": ta.ht_phasor(c),
        "ht_sine": ta.ht_sine(c),
        "ht_trendmode": ta.ht_trendmode(c),
        "ht_trendline": ta.ht_trendline(c),
        # ---- Cycles (native) ----
        "dsp": ta.dsp(c),
        "ebsw": ta.ebsw(c),
        # ---- Performance ----
        "log_return": ta.log_return(c),
        "percent_return": ta.percent_return(c),
        "drawdown": ta.drawdown(c),
        # ---- Candles ----
        "ha": ta.ha(o, h, l, c),
        "cdl_doji": ta.cdl_doji(o, h, l, c),
        "cdl_inside": ta.cdl_inside(o, h, l, c),
        "cdl_z": ta.cdl_z(o, h, l, c),
        # ---- Overlap (additional) ----
        "ichimoku": ta.ichimoku(h, l, c)[0],
        "ma_ema_20": ta.ma("ema", c, length=20),
        # ---- Trend (additional) ----
        "tsignals": ta.tsignals(_trend_bool),
        "xsignals": ta.xsignals(_rsi_14_sig, 70, 30),
        # ---- Volume (additional) ----
        "vp": ta.vp(c, v),
        # ---- Candles — CDL patterns (via dispatcher) ----
        "cdl_pattern": ta.cdl_pattern(o, h, l, c),
        "cdl_2crows": ta.cdl_pattern(o, h, l, c, name="2crows"),
        "cdl_3blackcrows": ta.cdl_pattern(o, h, l, c, name="3blackcrows"),
        "cdl_3inside": ta.cdl_pattern(o, h, l, c, name="3inside"),
        "cdl_3linestrike": ta.cdl_pattern(o, h, l, c, name="3linestrike"),
        "cdl_3outside": ta.cdl_pattern(o, h, l, c, name="3outside"),
        "cdl_3starsinsouth": ta.cdl_pattern(o, h, l, c, name="3starsinsouth"),
        "cdl_3whitesoldiers": ta.cdl_pattern(o, h, l, c, name="3whitesoldiers"),
        "cdl_abandonedbaby": ta.cdl_pattern(o, h, l, c, name="abandonedbaby"),
        "cdl_advanceblock": ta.cdl_pattern(o, h, l, c, name="advanceblock"),
        "cdl_belthold": ta.cdl_pattern(o, h, l, c, name="belthold"),
        "cdl_breakaway": ta.cdl_pattern(o, h, l, c, name="breakaway"),
        "cdl_closingmarubozu": ta.cdl_pattern(o, h, l, c, name="closingmarubozu"),
        "cdl_concealbabyswall": ta.cdl_pattern(o, h, l, c, name="concealbabyswall"),
        "cdl_counterattack": ta.cdl_pattern(o, h, l, c, name="counterattack"),
        "cdl_darkcloudcover": ta.cdl_pattern(o, h, l, c, name="darkcloudcover"),
        "cdl_dojistar": ta.cdl_pattern(o, h, l, c, name="dojistar"),
        "cdl_dragonflydoji": ta.cdl_pattern(o, h, l, c, name="dragonflydoji"),
        "cdl_engulfing": ta.cdl_pattern(o, h, l, c, name="engulfing"),
        "cdl_eveningdojistar": ta.cdl_pattern(o, h, l, c, name="eveningdojistar"),
        "cdl_eveningstar": ta.cdl_pattern(o, h, l, c, name="eveningstar"),
        "cdl_gapsidesidewhite": ta.cdl_pattern(o, h, l, c, name="gapsidesidewhite"),
        "cdl_gravestonedoji": ta.cdl_pattern(o, h, l, c, name="gravestonedoji"),
        "cdl_hammer": ta.cdl_pattern(o, h, l, c, name="hammer"),
        "cdl_hangingman": ta.cdl_pattern(o, h, l, c, name="hangingman"),
        "cdl_harami": ta.cdl_pattern(o, h, l, c, name="harami"),
        "cdl_haramicross": ta.cdl_pattern(o, h, l, c, name="haramicross"),
        "cdl_highwave": ta.cdl_pattern(o, h, l, c, name="highwave"),
        "cdl_hikkake": ta.cdl_pattern(o, h, l, c, name="hikkake"),
        "cdl_hikkakemod": ta.cdl_pattern(o, h, l, c, name="hikkakemod"),
        "cdl_homingpigeon": ta.cdl_pattern(o, h, l, c, name="homingpigeon"),
        "cdl_identical3crows": ta.cdl_pattern(o, h, l, c, name="identical3crows"),
        "cdl_inneck": ta.cdl_pattern(o, h, l, c, name="inneck"),
        "cdl_invertedhammer": ta.cdl_pattern(o, h, l, c, name="invertedhammer"),
        "cdl_kicking": ta.cdl_pattern(o, h, l, c, name="kicking"),
        "cdl_kickingbylength": ta.cdl_pattern(o, h, l, c, name="kickingbylength"),
        "cdl_ladderbottom": ta.cdl_pattern(o, h, l, c, name="ladderbottom"),
        "cdl_longleggeddoji": ta.cdl_pattern(o, h, l, c, name="longleggeddoji"),
        "cdl_longline": ta.cdl_pattern(o, h, l, c, name="longline"),
        "cdl_marubozu": ta.cdl_pattern(o, h, l, c, name="marubozu"),
        "cdl_matchinglow": ta.cdl_pattern(o, h, l, c, name="matchinglow"),
        "cdl_mathold": ta.cdl_pattern(o, h, l, c, name="mathold"),
        "cdl_morningdojistar": ta.cdl_pattern(o, h, l, c, name="morningdojistar"),
        "cdl_morningstar": ta.cdl_pattern(o, h, l, c, name="morningstar"),
        "cdl_onneck": ta.cdl_pattern(o, h, l, c, name="onneck"),
        "cdl_piercing": ta.cdl_pattern(o, h, l, c, name="piercing"),
        "cdl_rickshawman": ta.cdl_pattern(o, h, l, c, name="rickshawman"),
        "cdl_risefall3methods": ta.cdl_pattern(o, h, l, c, name="risefall3methods"),
        "cdl_separatinglines": ta.cdl_pattern(o, h, l, c, name="separatinglines"),
        "cdl_shootingstar": ta.cdl_pattern(o, h, l, c, name="shootingstar"),
        "cdl_shortline": ta.cdl_pattern(o, h, l, c, name="shortline"),
        "cdl_spinningtop": ta.cdl_pattern(o, h, l, c, name="spinningtop"),
        "cdl_stalledpattern": ta.cdl_pattern(o, h, l, c, name="stalledpattern"),
        "cdl_sticksandwich": ta.cdl_pattern(o, h, l, c, name="sticksandwich"),
        "cdl_takuri": ta.cdl_pattern(o, h, l, c, name="takuri"),
        "cdl_tasukigap": ta.cdl_pattern(o, h, l, c, name="tasukigap"),
        "cdl_thrusting": ta.cdl_pattern(o, h, l, c, name="thrusting"),
        "cdl_tristar": ta.cdl_pattern(o, h, l, c, name="tristar"),
        "cdl_unique3river": ta.cdl_pattern(o, h, l, c, name="unique3river"),
        "cdl_upsidegap2crows": ta.cdl_pattern(o, h, l, c, name="upsidegap2crows"),
        "cdl_xsidegap3methods": ta.cdl_pattern(o, h, l, c, name="xsidegap3methods"),
    }


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

REL_TOL = 1e-4  # 0.01 % relative tolerance for floating-point comparisons


def _approx_equal(actual: float, expected: float) -> bool:
    """Return True when |actual - expected| / |expected| <= REL_TOL."""
    if expected == 0.0:
        return abs(actual) <= REL_TOL
    return abs(actual - expected) / abs(expected) <= REL_TOL


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestIndicatorValues(TestCase):
    """Assert that every tracked indicator produces the expected golden values."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_data()
        cls.results = _compute_all(cls.df)

    @classmethod
    def tearDownClass(cls):
        del cls.df
        del cls.results

    # ------------------------------------------------------------------
    # Internal helper used by every generated test method
    # ------------------------------------------------------------------

    def _check_fixture(self, fixture_key: str) -> None:
        """Assert last value and non-NaN count for every column of fixture_key."""
        self.assertIn(
            fixture_key, self.results, f"No result computed for {fixture_key!r}"
        )
        result = self.results[fixture_key]
        if result is None:
            self.fail(f"{fixture_key!r} returned None")

        # Normalise to DataFrame so we iterate columns uniformly
        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name)

        expected_fixture = _FIXTURES[fixture_key]

        for col, expected in expected_fixture.items():
            with self.subTest(col=col):
                self.assertIn(
                    col,
                    result.columns,
                    f"Column {col!r} missing from {fixture_key!r} result",
                )

                series = result[col]
                actual_non_nan = int(series.notna().sum())
                self.assertEqual(
                    actual_non_nan,
                    expected["non_nan_count"],
                    f"{fixture_key!r}[{col!r}]: non_nan_count "
                    f"{actual_non_nan} != {expected['non_nan_count']}",
                )

                expected_last = expected["last_value"]
                if expected_last is None:
                    return  # column was all-NaN when fixtures were generated

                clean = series.dropna()
                self.assertFalse(
                    clean.empty,
                    f"{fixture_key!r}[{col!r}]: result is all-NaN but fixture has a value",
                )
                actual_last = float(clean.iloc[-1])
                self.assertFalse(
                    math.isnan(actual_last) or math.isinf(actual_last),
                    f"{fixture_key!r}[{col!r}]: last value is NaN/Inf",
                )
                self.assertTrue(
                    _approx_equal(actual_last, expected_last),
                    f"{fixture_key!r}[{col!r}]: last value {actual_last:.8f} "
                    f"differs from golden {expected_last:.8f}",
                )

    # ------------------------------------------------------------------
    # One test method per fixture key
    # ------------------------------------------------------------------

    # Overlap — TA-Lib backed
    def test_sma_20(self):
        self._check_fixture("sma_20")

    def test_ema_20(self):
        self._check_fixture("ema_20")

    def test_dema_10(self):
        self._check_fixture("dema_10")

    def test_tema_10(self):
        self._check_fixture("tema_10")

    def test_wma_10(self):
        self._check_fixture("wma_10")

    def test_midpoint_14(self):
        self._check_fixture("midpoint_14")

    def test_midprice_14(self):
        self._check_fixture("midprice_14")

    def test_t3_10(self):
        self._check_fixture("t3_10")

    def test_wcp(self):
        self._check_fixture("wcp")

    def test_hl2(self):
        self._check_fixture("hl2")

    def test_hlc3(self):
        self._check_fixture("hlc3")

    def test_ohlc4(self):
        self._check_fixture("ohlc4")

    # Overlap — native
    def test_hma_10(self):
        self._check_fixture("hma_10")

    def test_alma_10(self):
        self._check_fixture("alma_10")

    def test_trima_10(self):
        self._check_fixture("trima_10")

    def test_fwma_10(self):
        self._check_fixture("fwma_10")

    def test_hilo(self):
        self._check_fixture("hilo")

    def test_hwma(self):
        self._check_fixture("hwma")

    def test_jma(self):
        self._check_fixture("jma")

    def test_kama(self):
        self._check_fixture("kama")

    def test_linreg_14(self):
        self._check_fixture("linreg_14")

    def test_mama(self):
        self._check_fixture("mama")

    def test_mcgd(self):
        self._check_fixture("mcgd")

    def test_mmar(self):
        self._check_fixture("mmar")

    def test_pwma_10(self):
        self._check_fixture("pwma_10")

    def test_rainbow(self):
        self._check_fixture("rainbow")

    def test_rma(self):
        self._check_fixture("rma")

    def test_sinwma_14(self):
        self._check_fixture("sinwma_14")

    def test_ssf_10(self):
        self._check_fixture("ssf_10")

    def test_supertrend(self):
        self._check_fixture("supertrend")

    def test_swma(self):
        self._check_fixture("swma")

    def test_tsf_14(self):
        self._check_fixture("tsf_14")

    def test_vidya(self):
        self._check_fixture("vidya")

    def test_vwap(self):
        self._check_fixture("vwap")

    def test_vwma_10(self):
        self._check_fixture("vwma_10")

    def test_zlma_10(self):
        self._check_fixture("zlma_10")

    # Momentum — TA-Lib backed
    def test_rsi_14(self):
        self._check_fixture("rsi_14")

    def test_macd_12_26_9(self):
        self._check_fixture("macd_12_26_9")

    def test_stoch(self):
        self._check_fixture("stoch")

    def test_cci_14(self):
        self._check_fixture("cci_14")

    def test_roc_10(self):
        self._check_fixture("roc_10")

    def test_willr_14(self):
        self._check_fixture("willr_14")

    def test_apo(self):
        self._check_fixture("apo")

    def test_bop(self):
        self._check_fixture("bop")

    def test_mom_10(self):
        self._check_fixture("mom_10")

    def test_uo(self):
        self._check_fixture("uo")

    # Momentum — native
    def test_ao(self):
        self._check_fixture("ao")

    def test_bias_10(self):
        self._check_fixture("bias_10")

    def test_brar(self):
        self._check_fixture("brar")

    def test_cfo_10(self):
        self._check_fixture("cfo_10")

    def test_cg(self):
        self._check_fixture("cg")

    def test_cmo_14(self):
        self._check_fixture("cmo_14")

    def test_coppock(self):
        self._check_fixture("coppock")

    def test_cti(self):
        self._check_fixture("cti")

    def test_dm_14(self):
        self._check_fixture("dm_14")

    def test_er(self):
        self._check_fixture("er")

    def test_eri(self):
        self._check_fixture("eri")

    def test_fisher(self):
        self._check_fixture("fisher")

    def test_inertia(self):
        self._check_fixture("inertia")

    def test_kdj(self):
        self._check_fixture("kdj")

    def test_kst(self):
        self._check_fixture("kst")

    def test_lrsi(self):
        self._check_fixture("lrsi")

    def test_pgo_14(self):
        self._check_fixture("pgo_14")

    def test_po(self):
        self._check_fixture("po")

    def test_ppo(self):
        self._check_fixture("ppo")

    def test_psl(self):
        self._check_fixture("psl")

    def test_pvo(self):
        self._check_fixture("pvo")

    def test_qqe(self):
        self._check_fixture("qqe")

    def test_rsx(self):
        self._check_fixture("rsx")

    def test_rvgi(self):
        self._check_fixture("rvgi")

    def test_slope(self):
        self._check_fixture("slope")

    def test_smi(self):
        self._check_fixture("smi")

    def test_squeeze(self):
        self._check_fixture("squeeze")

    def test_squeeze_pro(self):
        self._check_fixture("squeeze_pro")

    def test_stc(self):
        self._check_fixture("stc")

    def test_stochrsi(self):
        self._check_fixture("stochrsi")

    def test_td_seq(self):
        self._check_fixture("td_seq")

    def test_trix(self):
        self._check_fixture("trix")

    def test_trixh(self):
        self._check_fixture("trixh")

    def test_tsi(self):
        self._check_fixture("tsi")

    def test_vwmacd(self):
        self._check_fixture("vwmacd")

    # Volatility — TA-Lib backed
    def test_atr_14(self):
        self._check_fixture("atr_14")

    def test_bbands_20(self):
        self._check_fixture("bbands_20")

    def test_natr_14(self):
        self._check_fixture("natr_14")

    def test_true_range(self):
        self._check_fixture("true_range")

    # Volatility — native
    def test_donchian_20(self):
        self._check_fixture("donchian_20")

    def test_kc_20(self):
        self._check_fixture("kc_20")

    def test_aberration(self):
        self._check_fixture("aberration")

    def test_accbands(self):
        self._check_fixture("accbands")

    def test_ce(self):
        self._check_fixture("ce")

    def test_hwc(self):
        self._check_fixture("hwc")

    def test_massi(self):
        self._check_fixture("massi")

    def test_pdist(self):
        self._check_fixture("pdist")

    def test_rvi_vol(self):
        self._check_fixture("rvi_vol")

    def test_thermo(self):
        self._check_fixture("thermo")

    def test_ui(self):
        self._check_fixture("ui")

    # Trend — TA-Lib backed
    def test_adx_14(self):
        self._check_fixture("adx_14")

    def test_aroon_14(self):
        self._check_fixture("aroon_14")

    # Trend — native
    def test_psar(self):
        self._check_fixture("psar")

    def test_decreasing(self):
        self._check_fixture("decreasing")

    def test_increasing(self):
        self._check_fixture("increasing")

    def test_adxr(self):
        self._check_fixture("adxr")

    def test_amat(self):
        self._check_fixture("amat")

    def test_chop(self):
        self._check_fixture("chop")

    def test_cksp(self):
        self._check_fixture("cksp")

    def test_cpr(self):
        self._check_fixture("cpr")

    def test_decay(self):
        self._check_fixture("decay")

    def test_dpo_14(self):
        self._check_fixture("dpo_14")

    def test_long_run(self):
        self._check_fixture("long_run")

    def test_pmax(self):
        self._check_fixture("pmax")

    def test_qstick(self):
        self._check_fixture("qstick")

    def test_short_run(self):
        self._check_fixture("short_run")

    def test_ttm_trend(self):
        self._check_fixture("ttm_trend")

    def test_vhf(self):
        self._check_fixture("vhf")

    def test_vortex(self):
        self._check_fixture("vortex")

    # Volume — TA-Lib backed
    def test_obv(self):
        self._check_fixture("obv")

    def test_mfi_14(self):
        self._check_fixture("mfi_14")

    def test_ad(self):
        self._check_fixture("ad")

    def test_adosc(self):
        self._check_fixture("adosc")

    # Volume — native
    def test_cmf_20(self):
        self._check_fixture("cmf_20")

    def test_aobv(self):
        self._check_fixture("aobv")

    def test_efi(self):
        self._check_fixture("efi")

    def test_eom(self):
        self._check_fixture("eom")

    def test_kvo(self):
        self._check_fixture("kvo")

    def test_nvi(self):
        self._check_fixture("nvi")

    def test_pvi(self):
        self._check_fixture("pvi")

    def test_pvol(self):
        self._check_fixture("pvol")

    def test_pvr(self):
        self._check_fixture("pvr")

    def test_pvt(self):
        self._check_fixture("pvt")

    def test_vfi(self):
        self._check_fixture("vfi")

    # Statistics — TA-Lib backed
    def test_stdev_20(self):
        self._check_fixture("stdev_20")

    def test_variance_20(self):
        self._check_fixture("variance_20")

    # Statistics — native
    def test_zscore_20(self):
        self._check_fixture("zscore_20")

    def test_kurtosis_20(self):
        self._check_fixture("kurtosis_20")

    def test_skew_20(self):
        self._check_fixture("skew_20")

    def test_beta(self):
        self._check_fixture("beta")

    def test_correl(self):
        self._check_fixture("correl")

    def test_entropy_10(self):
        self._check_fixture("entropy_10")

    def test_mad_10(self):
        self._check_fixture("mad_10")

    def test_median_14(self):
        self._check_fixture("median_14")

    def test_quantile_14(self):
        self._check_fixture("quantile_14")

    def test_tos_stdevall(self):
        self._check_fixture("tos_stdevall")

    # Cycles — TA-Lib backed
    def test_ht_dcperiod(self):
        self._check_fixture("ht_dcperiod")

    def test_ht_dcphase(self):
        self._check_fixture("ht_dcphase")

    def test_ht_phasor(self):
        self._check_fixture("ht_phasor")

    def test_ht_sine(self):
        self._check_fixture("ht_sine")

    def test_ht_trendmode(self):
        self._check_fixture("ht_trendmode")

    def test_ht_trendline(self):
        self._check_fixture("ht_trendline")

    # Cycles — native
    def test_dsp(self):
        self._check_fixture("dsp")

    def test_ebsw(self):
        self._check_fixture("ebsw")

    # Performance
    def test_log_return(self):
        self._check_fixture("log_return")

    def test_percent_return(self):
        self._check_fixture("percent_return")

    def test_drawdown(self):
        self._check_fixture("drawdown")

    # Candles
    def test_ha(self):
        self._check_fixture("ha")

    def test_cdl_doji(self):
        self._check_fixture("cdl_doji")

    def test_cdl_inside(self):
        self._check_fixture("cdl_inside")

    def test_cdl_z(self):
        self._check_fixture("cdl_z")

    # Overlap (additional)
    def test_ichimoku(self):
        self._check_fixture("ichimoku")

    def test_ma_ema_20(self):
        self._check_fixture("ma_ema_20")

    # Trend (additional)
    def test_tsignals(self):
        self._check_fixture("tsignals")

    def test_xsignals(self):
        self._check_fixture("xsignals")

    # Volume (additional)
    def test_vp(self):
        self._check_fixture("vp")

    # Candles — CDL patterns (via dispatcher)
    def test_cdl_pattern(self):
        self._check_fixture("cdl_pattern")

    def test_cdl_2crows(self):
        self._check_fixture("cdl_2crows")

    def test_cdl_3blackcrows(self):
        self._check_fixture("cdl_3blackcrows")

    def test_cdl_3inside(self):
        self._check_fixture("cdl_3inside")

    def test_cdl_3linestrike(self):
        self._check_fixture("cdl_3linestrike")

    def test_cdl_3outside(self):
        self._check_fixture("cdl_3outside")

    def test_cdl_3starsinsouth(self):
        self._check_fixture("cdl_3starsinsouth")

    def test_cdl_3whitesoldiers(self):
        self._check_fixture("cdl_3whitesoldiers")

    def test_cdl_abandonedbaby(self):
        self._check_fixture("cdl_abandonedbaby")

    def test_cdl_advanceblock(self):
        self._check_fixture("cdl_advanceblock")

    def test_cdl_belthold(self):
        self._check_fixture("cdl_belthold")

    def test_cdl_breakaway(self):
        self._check_fixture("cdl_breakaway")

    def test_cdl_closingmarubozu(self):
        self._check_fixture("cdl_closingmarubozu")

    def test_cdl_concealbabyswall(self):
        self._check_fixture("cdl_concealbabyswall")

    def test_cdl_counterattack(self):
        self._check_fixture("cdl_counterattack")

    def test_cdl_darkcloudcover(self):
        self._check_fixture("cdl_darkcloudcover")

    def test_cdl_dojistar(self):
        self._check_fixture("cdl_dojistar")

    def test_cdl_dragonflydoji(self):
        self._check_fixture("cdl_dragonflydoji")

    def test_cdl_engulfing(self):
        self._check_fixture("cdl_engulfing")

    def test_cdl_eveningdojistar(self):
        self._check_fixture("cdl_eveningdojistar")

    def test_cdl_eveningstar(self):
        self._check_fixture("cdl_eveningstar")

    def test_cdl_gapsidesidewhite(self):
        self._check_fixture("cdl_gapsidesidewhite")

    def test_cdl_gravestonedoji(self):
        self._check_fixture("cdl_gravestonedoji")

    def test_cdl_hammer(self):
        self._check_fixture("cdl_hammer")

    def test_cdl_hangingman(self):
        self._check_fixture("cdl_hangingman")

    def test_cdl_harami(self):
        self._check_fixture("cdl_harami")

    def test_cdl_haramicross(self):
        self._check_fixture("cdl_haramicross")

    def test_cdl_highwave(self):
        self._check_fixture("cdl_highwave")

    def test_cdl_hikkake(self):
        self._check_fixture("cdl_hikkake")

    def test_cdl_hikkakemod(self):
        self._check_fixture("cdl_hikkakemod")

    def test_cdl_homingpigeon(self):
        self._check_fixture("cdl_homingpigeon")

    def test_cdl_identical3crows(self):
        self._check_fixture("cdl_identical3crows")

    def test_cdl_inneck(self):
        self._check_fixture("cdl_inneck")

    def test_cdl_invertedhammer(self):
        self._check_fixture("cdl_invertedhammer")

    def test_cdl_kicking(self):
        self._check_fixture("cdl_kicking")

    def test_cdl_kickingbylength(self):
        self._check_fixture("cdl_kickingbylength")

    def test_cdl_ladderbottom(self):
        self._check_fixture("cdl_ladderbottom")

    def test_cdl_longleggeddoji(self):
        self._check_fixture("cdl_longleggeddoji")

    def test_cdl_longline(self):
        self._check_fixture("cdl_longline")

    def test_cdl_marubozu(self):
        self._check_fixture("cdl_marubozu")

    def test_cdl_matchinglow(self):
        self._check_fixture("cdl_matchinglow")

    def test_cdl_mathold(self):
        self._check_fixture("cdl_mathold")

    def test_cdl_morningdojistar(self):
        self._check_fixture("cdl_morningdojistar")

    def test_cdl_morningstar(self):
        self._check_fixture("cdl_morningstar")

    def test_cdl_onneck(self):
        self._check_fixture("cdl_onneck")

    def test_cdl_piercing(self):
        self._check_fixture("cdl_piercing")

    def test_cdl_rickshawman(self):
        self._check_fixture("cdl_rickshawman")

    def test_cdl_risefall3methods(self):
        self._check_fixture("cdl_risefall3methods")

    def test_cdl_separatinglines(self):
        self._check_fixture("cdl_separatinglines")

    def test_cdl_shootingstar(self):
        self._check_fixture("cdl_shootingstar")

    def test_cdl_shortline(self):
        self._check_fixture("cdl_shortline")

    def test_cdl_spinningtop(self):
        self._check_fixture("cdl_spinningtop")

    def test_cdl_stalledpattern(self):
        self._check_fixture("cdl_stalledpattern")

    def test_cdl_sticksandwich(self):
        self._check_fixture("cdl_sticksandwich")

    def test_cdl_takuri(self):
        self._check_fixture("cdl_takuri")

    def test_cdl_tasukigap(self):
        self._check_fixture("cdl_tasukigap")

    def test_cdl_thrusting(self):
        self._check_fixture("cdl_thrusting")

    def test_cdl_tristar(self):
        self._check_fixture("cdl_tristar")

    def test_cdl_unique3river(self):
        self._check_fixture("cdl_unique3river")

    def test_cdl_upsidegap2crows(self):
        self._check_fixture("cdl_upsidegap2crows")

    def test_cdl_xsidegap3methods(self):
        self._check_fixture("cdl_xsidegap3methods")
