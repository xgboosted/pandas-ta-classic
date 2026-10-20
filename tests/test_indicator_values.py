"""
Priority 1 — Golden fixture tests.

Each test asserts two things for every column of every tracked indicator:
  1. The last non-NaN value matches the stored golden value within
     GOLDEN_ATOL + GOLDEN_RTOL * |golden| (see tests/assertions.py).
  2. The number of non-NaN rows matches the stored count exactly.

To regenerate fixtures after an intentional algorithm change run:
    python -m tests.fixtures.generate_fixtures
"""

import json
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

from tests.assertions import golden_value_close as _approx_equal
from tests.fixtures.tracked_indicators import compute_all as _compute_all, load_data as _load_data

# ---------------------------------------------------------------------------
# Load fixtures & sample data
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "expected_values.json"
with open(_FIXTURE_PATH) as _fh:
    _FIXTURES: dict[str, dict] = json.load(_fh)

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

# Comparison lives in tests/assertions.py — see there for how GOLDEN_ATOL and
# GOLDEN_RTOL were derived, and why both terms are needed.


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
        import unittest

        if fixture_key not in _FIXTURES:
            raise unittest.SkipTest(
                f"Fixture {fixture_key!r} not present in expected_values.json " f"(may require an optional dependency such as tulipy)"
            )
        self.assertIn(fixture_key, self.results, f"No result computed for {fixture_key!r}")
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
                    f"{fixture_key!r}[{col!r}]: non_nan_count " f"{actual_non_nan} != {expected['non_nan_count']}",
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
                    f"{fixture_key!r}[{col!r}]: last value {actual_last:.8f} " f"differs from golden {expected_last:.8f}",
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
