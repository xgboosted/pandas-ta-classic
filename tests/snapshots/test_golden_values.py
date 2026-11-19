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

import pandas as pd
import pytest

from tests.assertions import golden_value_close as _approx_equal
from tests.config import get_sample_data
from tests.fixtures.tracked_indicators import compute_all as _compute_all

# ---------------------------------------------------------------------------
# Load fixtures & sample data
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "expected_values.json"
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


class TestIndicatorValues:
    """Assert that every tracked indicator produces the expected golden values."""

    @classmethod
    def setup_class(cls):
        cls.df = get_sample_data()
        cls.results = _compute_all(cls.df)

    @classmethod
    def teardown_class(cls):
        del cls.df
        del cls.results

    # ------------------------------------------------------------------
    # Internal helper used by every generated test method
    # ------------------------------------------------------------------

    def _check_fixture(self, fixture_key: str, subtests) -> None:
        """Assert last value and non-NaN count for every column of fixture_key."""

        if fixture_key not in _FIXTURES:
            pytest.skip(
                f"Fixture {fixture_key!r} not present in expected_values.json " f"(may require an optional dependency such as tulipy)"
            )
        assert fixture_key in self.results, f"No result computed for {fixture_key!r}"
        result = self.results[fixture_key]
        if result is None:
            pytest.fail(f"{fixture_key!r} returned None")

        # Normalise to DataFrame so we iterate columns uniformly
        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name)

        expected_fixture = _FIXTURES[fixture_key]

        for col, expected in expected_fixture.items():
            with subtests.test(col=col):
                assert col in result.columns, f"Column {col!r} missing from {fixture_key!r} result"

                series = result[col]
                actual_non_nan = int(series.notna().sum())
                assert actual_non_nan == expected["non_nan_count"], f"{fixture_key!r}[{col!r}]: non_nan_count " f"{actual_non_nan} != {expected['non_nan_count']}"

                expected_last = expected["last_value"]
                if expected_last is None:
                    continue  # column was all-NaN when fixtures were generated; the next column is still checked

                clean = series.dropna()
                assert not clean.empty, f"{fixture_key!r}[{col!r}]: result is all-NaN but fixture has a value"
                actual_last = float(clean.iloc[-1])
                assert not (math.isnan(actual_last) or math.isinf(actual_last)), f"{fixture_key!r}[{col!r}]: last value is NaN/Inf"
                assert _approx_equal(actual_last, expected_last), f"{fixture_key!r}[{col!r}]: last value {actual_last:.8f} " f"differs from golden {expected_last:.8f}"

    # ------------------------------------------------------------------
    # One test method per fixture key
    # ------------------------------------------------------------------

    # Overlap — TA-Lib backed
    def test_sma_20(self, subtests):
        self._check_fixture("sma_20", subtests=subtests)

    def test_ema_20(self, subtests):
        self._check_fixture("ema_20", subtests=subtests)

    def test_dema_10(self, subtests):
        self._check_fixture("dema_10", subtests=subtests)

    def test_tema_10(self, subtests):
        self._check_fixture("tema_10", subtests=subtests)

    def test_wma_10(self, subtests):
        self._check_fixture("wma_10", subtests=subtests)

    def test_midpoint_14(self, subtests):
        self._check_fixture("midpoint_14", subtests=subtests)

    def test_midprice_14(self, subtests):
        self._check_fixture("midprice_14", subtests=subtests)

    def test_t3_10(self, subtests):
        self._check_fixture("t3_10", subtests=subtests)

    def test_wcp(self, subtests):
        self._check_fixture("wcp", subtests=subtests)

    def test_hl2(self, subtests):
        self._check_fixture("hl2", subtests=subtests)

    def test_hlc3(self, subtests):
        self._check_fixture("hlc3", subtests=subtests)

    def test_ohlc4(self, subtests):
        self._check_fixture("ohlc4", subtests=subtests)

    # Overlap — native
    def test_hma_10(self, subtests):
        self._check_fixture("hma_10", subtests=subtests)

    def test_alma_10(self, subtests):
        self._check_fixture("alma_10", subtests=subtests)

    def test_trima_10(self, subtests):
        self._check_fixture("trima_10", subtests=subtests)

    def test_fwma_10(self, subtests):
        self._check_fixture("fwma_10", subtests=subtests)

    def test_hilo(self, subtests):
        self._check_fixture("hilo", subtests=subtests)

    def test_hwma(self, subtests):
        self._check_fixture("hwma", subtests=subtests)

    def test_jma(self, subtests):
        self._check_fixture("jma", subtests=subtests)

    def test_kama(self, subtests):
        self._check_fixture("kama", subtests=subtests)

    def test_linreg_14(self, subtests):
        self._check_fixture("linreg_14", subtests=subtests)

    def test_mama(self, subtests):
        self._check_fixture("mama", subtests=subtests)

    def test_mcgd(self, subtests):
        self._check_fixture("mcgd", subtests=subtests)

    def test_mmar(self, subtests):
        self._check_fixture("mmar", subtests=subtests)

    def test_pwma_10(self, subtests):
        self._check_fixture("pwma_10", subtests=subtests)

    def test_rainbow(self, subtests):
        self._check_fixture("rainbow", subtests=subtests)

    def test_rma(self, subtests):
        self._check_fixture("rma", subtests=subtests)

    def test_sinwma_14(self, subtests):
        self._check_fixture("sinwma_14", subtests=subtests)

    def test_ssf_10(self, subtests):
        self._check_fixture("ssf_10", subtests=subtests)

    def test_supertrend(self, subtests):
        self._check_fixture("supertrend", subtests=subtests)

    def test_swma(self, subtests):
        self._check_fixture("swma", subtests=subtests)

    def test_tsf_14(self, subtests):
        self._check_fixture("tsf_14", subtests=subtests)

    def test_vidya(self, subtests):
        self._check_fixture("vidya", subtests=subtests)

    def test_vwap(self, subtests):
        self._check_fixture("vwap", subtests=subtests)

    def test_vwma_10(self, subtests):
        self._check_fixture("vwma_10", subtests=subtests)

    def test_zlma_10(self, subtests):
        self._check_fixture("zlma_10", subtests=subtests)

    # Momentum — TA-Lib backed
    def test_rsi_14(self, subtests):
        self._check_fixture("rsi_14", subtests=subtests)

    def test_macd_12_26_9(self, subtests):
        self._check_fixture("macd_12_26_9", subtests=subtests)

    def test_stoch(self, subtests):
        self._check_fixture("stoch", subtests=subtests)

    def test_cci_14(self, subtests):
        self._check_fixture("cci_14", subtests=subtests)

    def test_roc_10(self, subtests):
        self._check_fixture("roc_10", subtests=subtests)

    def test_willr_14(self, subtests):
        self._check_fixture("willr_14", subtests=subtests)

    def test_apo(self, subtests):
        self._check_fixture("apo", subtests=subtests)

    def test_bop(self, subtests):
        self._check_fixture("bop", subtests=subtests)

    def test_mom_10(self, subtests):
        self._check_fixture("mom_10", subtests=subtests)

    def test_uo(self, subtests):
        self._check_fixture("uo", subtests=subtests)

    # Momentum — native
    def test_ao(self, subtests):
        self._check_fixture("ao", subtests=subtests)

    def test_bias_10(self, subtests):
        self._check_fixture("bias_10", subtests=subtests)

    def test_brar(self, subtests):
        self._check_fixture("brar", subtests=subtests)

    def test_cfo_10(self, subtests):
        self._check_fixture("cfo_10", subtests=subtests)

    def test_cg(self, subtests):
        self._check_fixture("cg", subtests=subtests)

    def test_cmo_14(self, subtests):
        self._check_fixture("cmo_14", subtests=subtests)

    def test_coppock(self, subtests):
        self._check_fixture("coppock", subtests=subtests)

    def test_cti(self, subtests):
        self._check_fixture("cti", subtests=subtests)

    def test_dm_14(self, subtests):
        self._check_fixture("dm_14", subtests=subtests)

    def test_er(self, subtests):
        self._check_fixture("er", subtests=subtests)

    def test_eri(self, subtests):
        self._check_fixture("eri", subtests=subtests)

    def test_fisher(self, subtests):
        self._check_fixture("fisher", subtests=subtests)

    def test_inertia(self, subtests):
        self._check_fixture("inertia", subtests=subtests)

    def test_kdj(self, subtests):
        self._check_fixture("kdj", subtests=subtests)

    def test_kst(self, subtests):
        self._check_fixture("kst", subtests=subtests)

    def test_lrsi(self, subtests):
        self._check_fixture("lrsi", subtests=subtests)

    def test_pgo_14(self, subtests):
        self._check_fixture("pgo_14", subtests=subtests)

    def test_po(self, subtests):
        self._check_fixture("po", subtests=subtests)

    def test_ppo(self, subtests):
        self._check_fixture("ppo", subtests=subtests)

    def test_psl(self, subtests):
        self._check_fixture("psl", subtests=subtests)

    def test_pvo(self, subtests):
        self._check_fixture("pvo", subtests=subtests)

    def test_qqe(self, subtests):
        self._check_fixture("qqe", subtests=subtests)

    def test_rsx(self, subtests):
        self._check_fixture("rsx", subtests=subtests)

    def test_rvgi(self, subtests):
        self._check_fixture("rvgi", subtests=subtests)

    def test_slope(self, subtests):
        self._check_fixture("slope", subtests=subtests)

    def test_smi(self, subtests):
        self._check_fixture("smi", subtests=subtests)

    def test_squeeze(self, subtests):
        self._check_fixture("squeeze", subtests=subtests)

    def test_squeeze_pro(self, subtests):
        self._check_fixture("squeeze_pro", subtests=subtests)

    def test_stc(self, subtests):
        self._check_fixture("stc", subtests=subtests)

    def test_stochrsi(self, subtests):
        self._check_fixture("stochrsi", subtests=subtests)

    def test_td_seq(self, subtests):
        self._check_fixture("td_seq", subtests=subtests)

    def test_trix(self, subtests):
        self._check_fixture("trix", subtests=subtests)

    def test_trixh(self, subtests):
        self._check_fixture("trixh", subtests=subtests)

    def test_tsi(self, subtests):
        self._check_fixture("tsi", subtests=subtests)

    def test_vwmacd(self, subtests):
        self._check_fixture("vwmacd", subtests=subtests)

    # Volatility — TA-Lib backed
    def test_atr_14(self, subtests):
        self._check_fixture("atr_14", subtests=subtests)

    def test_bbands_20(self, subtests):
        self._check_fixture("bbands_20", subtests=subtests)

    def test_natr_14(self, subtests):
        self._check_fixture("natr_14", subtests=subtests)

    def test_true_range(self, subtests):
        self._check_fixture("true_range", subtests=subtests)

    # Volatility — native
    def test_donchian_20(self, subtests):
        self._check_fixture("donchian_20", subtests=subtests)

    def test_kc_20(self, subtests):
        self._check_fixture("kc_20", subtests=subtests)

    def test_aberration(self, subtests):
        self._check_fixture("aberration", subtests=subtests)

    def test_accbands(self, subtests):
        self._check_fixture("accbands", subtests=subtests)

    def test_ce(self, subtests):
        self._check_fixture("ce", subtests=subtests)

    def test_hwc(self, subtests):
        self._check_fixture("hwc", subtests=subtests)

    def test_massi(self, subtests):
        self._check_fixture("massi", subtests=subtests)

    def test_pdist(self, subtests):
        self._check_fixture("pdist", subtests=subtests)

    def test_rvi_vol(self, subtests):
        self._check_fixture("rvi_vol", subtests=subtests)

    def test_thermo(self, subtests):
        self._check_fixture("thermo", subtests=subtests)

    def test_ui(self, subtests):
        self._check_fixture("ui", subtests=subtests)

    # Trend — TA-Lib backed
    def test_adx_14(self, subtests):
        self._check_fixture("adx_14", subtests=subtests)

    def test_aroon_14(self, subtests):
        self._check_fixture("aroon_14", subtests=subtests)

    # Trend — native
    def test_psar(self, subtests):
        self._check_fixture("psar", subtests=subtests)

    def test_decreasing(self, subtests):
        self._check_fixture("decreasing", subtests=subtests)

    def test_increasing(self, subtests):
        self._check_fixture("increasing", subtests=subtests)

    def test_adxr(self, subtests):
        self._check_fixture("adxr", subtests=subtests)

    def test_amat(self, subtests):
        self._check_fixture("amat", subtests=subtests)

    def test_chop(self, subtests):
        self._check_fixture("chop", subtests=subtests)

    def test_cksp(self, subtests):
        self._check_fixture("cksp", subtests=subtests)

    def test_cpr(self, subtests):
        self._check_fixture("cpr", subtests=subtests)

    def test_decay(self, subtests):
        self._check_fixture("decay", subtests=subtests)

    def test_dpo_14(self, subtests):
        self._check_fixture("dpo_14", subtests=subtests)

    def test_long_run(self, subtests):
        self._check_fixture("long_run", subtests=subtests)

    def test_pmax(self, subtests):
        self._check_fixture("pmax", subtests=subtests)

    def test_qstick(self, subtests):
        self._check_fixture("qstick", subtests=subtests)

    def test_short_run(self, subtests):
        self._check_fixture("short_run", subtests=subtests)

    def test_ttm_trend(self, subtests):
        self._check_fixture("ttm_trend", subtests=subtests)

    def test_vhf(self, subtests):
        self._check_fixture("vhf", subtests=subtests)

    def test_vortex(self, subtests):
        self._check_fixture("vortex", subtests=subtests)

    # Volume — TA-Lib backed
    def test_obv(self, subtests):
        self._check_fixture("obv", subtests=subtests)

    def test_mfi_14(self, subtests):
        self._check_fixture("mfi_14", subtests=subtests)

    def test_ad(self, subtests):
        self._check_fixture("ad", subtests=subtests)

    def test_adosc(self, subtests):
        self._check_fixture("adosc", subtests=subtests)

    # Volume — native
    def test_cmf_20(self, subtests):
        self._check_fixture("cmf_20", subtests=subtests)

    def test_aobv(self, subtests):
        self._check_fixture("aobv", subtests=subtests)

    def test_efi(self, subtests):
        self._check_fixture("efi", subtests=subtests)

    def test_eom(self, subtests):
        self._check_fixture("eom", subtests=subtests)

    def test_kvo(self, subtests):
        self._check_fixture("kvo", subtests=subtests)

    def test_nvi(self, subtests):
        self._check_fixture("nvi", subtests=subtests)

    def test_pvi(self, subtests):
        self._check_fixture("pvi", subtests=subtests)

    def test_pvol(self, subtests):
        self._check_fixture("pvol", subtests=subtests)

    def test_pvr(self, subtests):
        self._check_fixture("pvr", subtests=subtests)

    def test_pvt(self, subtests):
        self._check_fixture("pvt", subtests=subtests)

    def test_vfi(self, subtests):
        self._check_fixture("vfi", subtests=subtests)

    # Statistics — TA-Lib backed
    def test_stdev_20(self, subtests):
        self._check_fixture("stdev_20", subtests=subtests)

    def test_variance_20(self, subtests):
        self._check_fixture("variance_20", subtests=subtests)

    # Statistics — native
    def test_zscore_20(self, subtests):
        self._check_fixture("zscore_20", subtests=subtests)

    def test_kurtosis_20(self, subtests):
        self._check_fixture("kurtosis_20", subtests=subtests)

    def test_skew_20(self, subtests):
        self._check_fixture("skew_20", subtests=subtests)

    def test_beta(self, subtests):
        self._check_fixture("beta", subtests=subtests)

    def test_correl(self, subtests):
        self._check_fixture("correl", subtests=subtests)

    def test_entropy_10(self, subtests):
        self._check_fixture("entropy_10", subtests=subtests)

    def test_mad_10(self, subtests):
        self._check_fixture("mad_10", subtests=subtests)

    def test_median_14(self, subtests):
        self._check_fixture("median_14", subtests=subtests)

    def test_quantile_14(self, subtests):
        self._check_fixture("quantile_14", subtests=subtests)

    def test_tos_stdevall(self, subtests):
        self._check_fixture("tos_stdevall", subtests=subtests)

    # Cycles — TA-Lib backed
    def test_ht_dcperiod(self, subtests):
        self._check_fixture("ht_dcperiod", subtests=subtests)

    def test_ht_dcphase(self, subtests):
        self._check_fixture("ht_dcphase", subtests=subtests)

    def test_ht_phasor(self, subtests):
        self._check_fixture("ht_phasor", subtests=subtests)

    def test_ht_sine(self, subtests):
        self._check_fixture("ht_sine", subtests=subtests)

    def test_ht_trendmode(self, subtests):
        self._check_fixture("ht_trendmode", subtests=subtests)

    def test_ht_trendline(self, subtests):
        self._check_fixture("ht_trendline", subtests=subtests)

    # Cycles — native
    def test_dsp(self, subtests):
        self._check_fixture("dsp", subtests=subtests)

    def test_ebsw(self, subtests):
        self._check_fixture("ebsw", subtests=subtests)

    # Performance
    def test_log_return(self, subtests):
        self._check_fixture("log_return", subtests=subtests)

    def test_percent_return(self, subtests):
        self._check_fixture("percent_return", subtests=subtests)

    def test_drawdown(self, subtests):
        self._check_fixture("drawdown", subtests=subtests)

    # Candles
    def test_ha(self, subtests):
        self._check_fixture("ha", subtests=subtests)

    def test_cdl_doji(self, subtests):
        self._check_fixture("cdl_doji", subtests=subtests)

    def test_cdl_inside(self, subtests):
        self._check_fixture("cdl_inside", subtests=subtests)

    def test_cdl_z(self, subtests):
        self._check_fixture("cdl_z", subtests=subtests)

    # Overlap (additional)
    def test_ichimoku(self, subtests):
        self._check_fixture("ichimoku", subtests=subtests)

    def test_ma_ema_20(self, subtests):
        self._check_fixture("ma_ema_20", subtests=subtests)

    # Trend (additional)
    def test_tsignals(self, subtests):
        self._check_fixture("tsignals", subtests=subtests)

    def test_xsignals(self, subtests):
        self._check_fixture("xsignals", subtests=subtests)

    # Volume (additional)
    def test_vp(self, subtests):
        self._check_fixture("vp", subtests=subtests)

    # Candles — CDL patterns (via dispatcher)
    def test_cdl_pattern(self, subtests):
        self._check_fixture("cdl_pattern", subtests=subtests)

    def test_cdl_2crows(self, subtests):
        self._check_fixture("cdl_2crows", subtests=subtests)

    def test_cdl_3blackcrows(self, subtests):
        self._check_fixture("cdl_3blackcrows", subtests=subtests)

    def test_cdl_3inside(self, subtests):
        self._check_fixture("cdl_3inside", subtests=subtests)

    def test_cdl_3linestrike(self, subtests):
        self._check_fixture("cdl_3linestrike", subtests=subtests)

    def test_cdl_3outside(self, subtests):
        self._check_fixture("cdl_3outside", subtests=subtests)

    def test_cdl_3starsinsouth(self, subtests):
        self._check_fixture("cdl_3starsinsouth", subtests=subtests)

    def test_cdl_3whitesoldiers(self, subtests):
        self._check_fixture("cdl_3whitesoldiers", subtests=subtests)

    def test_cdl_abandonedbaby(self, subtests):
        self._check_fixture("cdl_abandonedbaby", subtests=subtests)

    def test_cdl_advanceblock(self, subtests):
        self._check_fixture("cdl_advanceblock", subtests=subtests)

    def test_cdl_belthold(self, subtests):
        self._check_fixture("cdl_belthold", subtests=subtests)

    def test_cdl_breakaway(self, subtests):
        self._check_fixture("cdl_breakaway", subtests=subtests)

    def test_cdl_closingmarubozu(self, subtests):
        self._check_fixture("cdl_closingmarubozu", subtests=subtests)

    def test_cdl_concealbabyswall(self, subtests):
        self._check_fixture("cdl_concealbabyswall", subtests=subtests)

    def test_cdl_counterattack(self, subtests):
        self._check_fixture("cdl_counterattack", subtests=subtests)

    def test_cdl_darkcloudcover(self, subtests):
        self._check_fixture("cdl_darkcloudcover", subtests=subtests)

    def test_cdl_dojistar(self, subtests):
        self._check_fixture("cdl_dojistar", subtests=subtests)

    def test_cdl_dragonflydoji(self, subtests):
        self._check_fixture("cdl_dragonflydoji", subtests=subtests)

    def test_cdl_engulfing(self, subtests):
        self._check_fixture("cdl_engulfing", subtests=subtests)

    def test_cdl_eveningdojistar(self, subtests):
        self._check_fixture("cdl_eveningdojistar", subtests=subtests)

    def test_cdl_eveningstar(self, subtests):
        self._check_fixture("cdl_eveningstar", subtests=subtests)

    def test_cdl_gapsidesidewhite(self, subtests):
        self._check_fixture("cdl_gapsidesidewhite", subtests=subtests)

    def test_cdl_gravestonedoji(self, subtests):
        self._check_fixture("cdl_gravestonedoji", subtests=subtests)

    def test_cdl_hammer(self, subtests):
        self._check_fixture("cdl_hammer", subtests=subtests)

    def test_cdl_hangingman(self, subtests):
        self._check_fixture("cdl_hangingman", subtests=subtests)

    def test_cdl_harami(self, subtests):
        self._check_fixture("cdl_harami", subtests=subtests)

    def test_cdl_haramicross(self, subtests):
        self._check_fixture("cdl_haramicross", subtests=subtests)

    def test_cdl_highwave(self, subtests):
        self._check_fixture("cdl_highwave", subtests=subtests)

    def test_cdl_hikkake(self, subtests):
        self._check_fixture("cdl_hikkake", subtests=subtests)

    def test_cdl_hikkakemod(self, subtests):
        self._check_fixture("cdl_hikkakemod", subtests=subtests)

    def test_cdl_homingpigeon(self, subtests):
        self._check_fixture("cdl_homingpigeon", subtests=subtests)

    def test_cdl_identical3crows(self, subtests):
        self._check_fixture("cdl_identical3crows", subtests=subtests)

    def test_cdl_inneck(self, subtests):
        self._check_fixture("cdl_inneck", subtests=subtests)

    def test_cdl_invertedhammer(self, subtests):
        self._check_fixture("cdl_invertedhammer", subtests=subtests)

    def test_cdl_kicking(self, subtests):
        self._check_fixture("cdl_kicking", subtests=subtests)

    def test_cdl_kickingbylength(self, subtests):
        self._check_fixture("cdl_kickingbylength", subtests=subtests)

    def test_cdl_ladderbottom(self, subtests):
        self._check_fixture("cdl_ladderbottom", subtests=subtests)

    def test_cdl_longleggeddoji(self, subtests):
        self._check_fixture("cdl_longleggeddoji", subtests=subtests)

    def test_cdl_longline(self, subtests):
        self._check_fixture("cdl_longline", subtests=subtests)

    def test_cdl_marubozu(self, subtests):
        self._check_fixture("cdl_marubozu", subtests=subtests)

    def test_cdl_matchinglow(self, subtests):
        self._check_fixture("cdl_matchinglow", subtests=subtests)

    def test_cdl_mathold(self, subtests):
        self._check_fixture("cdl_mathold", subtests=subtests)

    def test_cdl_morningdojistar(self, subtests):
        self._check_fixture("cdl_morningdojistar", subtests=subtests)

    def test_cdl_morningstar(self, subtests):
        self._check_fixture("cdl_morningstar", subtests=subtests)

    def test_cdl_onneck(self, subtests):
        self._check_fixture("cdl_onneck", subtests=subtests)

    def test_cdl_piercing(self, subtests):
        self._check_fixture("cdl_piercing", subtests=subtests)

    def test_cdl_rickshawman(self, subtests):
        self._check_fixture("cdl_rickshawman", subtests=subtests)

    def test_cdl_risefall3methods(self, subtests):
        self._check_fixture("cdl_risefall3methods", subtests=subtests)

    def test_cdl_separatinglines(self, subtests):
        self._check_fixture("cdl_separatinglines", subtests=subtests)

    def test_cdl_shootingstar(self, subtests):
        self._check_fixture("cdl_shootingstar", subtests=subtests)

    def test_cdl_shortline(self, subtests):
        self._check_fixture("cdl_shortline", subtests=subtests)

    def test_cdl_spinningtop(self, subtests):
        self._check_fixture("cdl_spinningtop", subtests=subtests)

    def test_cdl_stalledpattern(self, subtests):
        self._check_fixture("cdl_stalledpattern", subtests=subtests)

    def test_cdl_sticksandwich(self, subtests):
        self._check_fixture("cdl_sticksandwich", subtests=subtests)

    def test_cdl_takuri(self, subtests):
        self._check_fixture("cdl_takuri", subtests=subtests)

    def test_cdl_tasukigap(self, subtests):
        self._check_fixture("cdl_tasukigap", subtests=subtests)

    def test_cdl_thrusting(self, subtests):
        self._check_fixture("cdl_thrusting", subtests=subtests)

    def test_cdl_tristar(self, subtests):
        self._check_fixture("cdl_tristar", subtests=subtests)

    def test_cdl_unique3river(self, subtests):
        self._check_fixture("cdl_unique3river", subtests=subtests)

    def test_cdl_upsidegap2crows(self, subtests):
        self._check_fixture("cdl_upsidegap2crows", subtests=subtests)

    def test_cdl_xsidegap3methods(self, subtests):
        self._check_fixture("cdl_xsidegap3methods", subtests=subtests)
