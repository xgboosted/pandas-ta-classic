# -*- coding: utf-8 -*-
"""Lookback consistency tests: talib=True vs talib=False.

Ensures that switching between native and TA-Lib paths does not change:
  1. Output length (must equal input length — no truncated Series)
  2. first_valid_index (NaN lookback must match)

Known differences are marked with pytest.mark.xfail so they are tracked
but do not block CI.  When a difference is fixed, the xfail will start
passing ("xpass") and pytest will flag it for removal.
"""
import unittest

import numpy as np
import pandas as pd
import pytest

from tests.config import HAS_TALIB

pytestmark = pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 500
_close = pd.Series(np.cumsum(np.random.randn(N)) + 100, name="close").clip(lower=1)
_high = _close + np.abs(np.random.randn(N)) * 2
_low = _close - np.abs(np.random.randn(N)) * 2
_low = _low.clip(lower=0.5)
_open = _close.shift(1).fillna(_close.iloc[0])
_volume = pd.Series(np.random.randint(1000, 100000, N), dtype=float, name="volume")
_benchmark = pd.Series(np.cumsum(np.random.randn(N)) + 100).clip(lower=1)


def _fv(result):
    """first_valid_index of result (Series or first column of DataFrame)."""
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0].first_valid_index()
    return result.first_valid_index()


def _assert_consistency(native, talib_result, name, *, check_fv=True):
    """Assert length and optionally first_valid_index match."""
    # Length: both must equal input length
    n_nat = len(native) if isinstance(native, (pd.Series, pd.DataFrame)) else 0
    n_tal = (
        len(talib_result) if isinstance(talib_result, (pd.Series, pd.DataFrame)) else 0
    )
    assert n_nat == N, f"{name}: native length {n_nat} != input length {N}"
    assert n_tal == N, f"{name}: talib length {n_tal} != input length {N}"

    if check_fv:
        fv_nat = _fv(native)
        fv_tal = _fv(talib_result)
        assert (
            fv_nat == fv_tal
        ), f"{name}: first_valid mismatch: native={fv_nat} talib={fv_tal}"

    # For DataFrames, check all columns
    if isinstance(native, pd.DataFrame) and isinstance(talib_result, pd.DataFrame):
        for i in range(min(native.shape[1], talib_result.shape[1])):
            col_nat = native.iloc[:, i]
            col_tal = talib_result.iloc[:, i]
            assert (
                len(col_nat) == N
            ), f"{name}[{i}]: native col length {len(col_nat)} != {N}"
            assert (
                len(col_tal) == N
            ), f"{name}[{i}]: talib col length {len(col_tal)} != {N}"
            if check_fv:
                fvn = col_nat.first_valid_index()
                fvt = col_tal.first_valid_index()
                assert (
                    fvn == fvt
                ), f"{name} col[{i}]: first_valid mismatch: native={fvn} talib={fvt}"


# ---------------------------------------------------------------------------
# Helper to import all indicator functions
# ---------------------------------------------------------------------------
import pandas_ta_classic as ta


# ========================== OVERLAP ========================================


class TestLookbackOverlap(unittest.TestCase):

    def test_ema(self):
        _assert_consistency(
            ta.ema(_close, length=30, talib=False),
            ta.ema(_close, length=30, talib=True),
            "EMA",
        )

    def test_sma(self):
        _assert_consistency(
            ta.sma(_close, length=30, talib=False),
            ta.sma(_close, length=30, talib=True),
            "SMA",
        )

    def test_dema(self):
        _assert_consistency(
            ta.dema(_close, length=30, talib=False),
            ta.dema(_close, length=30, talib=True),
            "DEMA",
        )

    def test_tema(self):
        _assert_consistency(
            ta.tema(_close, length=30, talib=False),
            ta.tema(_close, length=30, talib=True),
            "TEMA",
        )

    def test_kama(self):
        _assert_consistency(
            ta.kama(_close, length=30, talib=False),
            ta.kama(_close, length=30, talib=True),
            "KAMA",
        )

    def test_wma(self):
        _assert_consistency(
            ta.wma(_close, length=30, talib=False),
            ta.wma(_close, length=30, talib=True),
            "WMA",
        )

    def test_trima(self):
        _assert_consistency(
            ta.trima(_close, length=30, talib=False),
            ta.trima(_close, length=30, talib=True),
            "TRIMA",
        )

    def test_t3(self):
        _assert_consistency(
            ta.t3(_close, length=5, talib=False),
            ta.t3(_close, length=5, talib=True),
            "T3",
        )

    def test_bbands(self):
        _assert_consistency(
            ta.bbands(_close, length=20, talib=False),
            ta.bbands(_close, length=20, talib=True),
            "BBANDS",
        )

    def test_midpoint(self):
        _assert_consistency(
            ta.midpoint(_close, length=14, talib=False),
            ta.midpoint(_close, length=14, talib=True),
            "MIDPOINT",
        )

    def test_midprice(self):
        _assert_consistency(
            ta.midprice(_high, _low, length=14, talib=False),
            ta.midprice(_high, _low, length=14, talib=True),
            "MIDPRICE",
        )

    def test_mama(self):
        _assert_consistency(
            ta.mama(_close, talib=False),
            ta.mama(_close, talib=True),
            "MAMA",
        )

    @pytest.mark.xfail(reason="HT_TRENDLINE native fv=37, talib fv=63 (Hilbert warmup)")
    def test_ht_trendline(self):
        _assert_consistency(
            ta.ht_trendline(_close, talib=False),
            ta.ht_trendline(_close, talib=True),
            "HT_TRENDLINE",
        )

    def test_tsf(self):
        _assert_consistency(
            ta.tsf(_close, length=14, talib=False),
            ta.tsf(_close, length=14, talib=True),
            "TSF",
        )

    def test_linreg(self):
        _assert_consistency(
            ta.linreg(_close, length=14, talib=False),
            ta.linreg(_close, length=14, talib=True),
            "LINEARREG",
        )

    def test_linreg_slope(self):
        _assert_consistency(
            ta.linreg(_close, length=14, slope=True, talib=False),
            ta.linreg(_close, length=14, slope=True, talib=True),
            "LINEARREG_SLOPE",
        )

    def test_linreg_intercept(self):
        _assert_consistency(
            ta.linreg(_close, length=14, intercept=True, talib=False),
            ta.linreg(_close, length=14, intercept=True, talib=True),
            "LINEARREG_INTERCEPT",
        )


# ========================== MOMENTUM =======================================


class TestLookbackMomentum(unittest.TestCase):

    @pytest.mark.xfail(reason="RSI native fv=13, talib fv=14 (RMA off-by-one)")
    def test_rsi(self):
        _assert_consistency(
            ta.rsi(_close, length=14, talib=False),
            ta.rsi(_close, length=14, talib=True),
            "RSI",
        )

    @pytest.mark.xfail(reason="MACD native fv=25, talib fv=33 (EMA alignment)")
    def test_macd(self):
        _assert_consistency(
            ta.macd(_close, fast=12, slow=26, signal=9, talib=False),
            ta.macd(_close, fast=12, slow=26, signal=9, talib=True),
            "MACD",
        )

    def test_stoch(self):
        _assert_consistency(
            ta.stoch(_high, _low, _close, k=14, d=3, smooth_k=3, talib=False),
            ta.stoch(_high, _low, _close, k=14, d=3, smooth_k=3, talib=True),
            "STOCH",
        )

    def test_cmo(self):
        _assert_consistency(
            ta.cmo(_close, length=14, talib=False),
            ta.cmo(_close, length=14, talib=True),
            "CMO",
            check_fv=False,  # Different algorithm (simple sum vs Wilder)
        )

    def test_mom(self):
        _assert_consistency(
            ta.mom(_close, length=10, talib=False),
            ta.mom(_close, length=10, talib=True),
            "MOM",
        )

    def test_roc(self):
        _assert_consistency(
            ta.roc(_close, length=10, talib=False),
            ta.roc(_close, length=10, talib=True),
            "ROC",
        )

    def test_willr(self):
        _assert_consistency(
            ta.willr(_high, _low, _close, length=14, talib=False),
            ta.willr(_high, _low, _close, length=14, talib=True),
            "WILLR",
        )

    def test_cci(self):
        _assert_consistency(
            ta.cci(_high, _low, _close, length=14, talib=False),
            ta.cci(_high, _low, _close, length=14, talib=True),
            "CCI",
        )

    @pytest.mark.xfail(reason="MFI native fv=13, talib fv=14 (RMA off-by-one)")
    def test_mfi(self):
        _assert_consistency(
            ta.mfi(_high, _low, _close, _volume, length=14, talib=False),
            ta.mfi(_high, _low, _close, _volume, length=14, talib=True),
            "MFI",
        )

    def test_adx(self):
        _assert_consistency(
            ta.adx(_high, _low, _close, length=14, talib=False),
            ta.adx(_high, _low, _close, length=14, talib=True),
            "ADX",
        )

    def test_apo(self):
        _assert_consistency(
            ta.apo(_close, fast=12, slow=26, talib=False),
            ta.apo(_close, fast=12, slow=26, talib=True),
            "APO",
        )

    def test_ppo(self):
        _assert_consistency(
            ta.ppo(_close, fast=12, slow=26, talib=False),
            ta.ppo(_close, fast=12, slow=26, talib=True),
            "PPO",
        )

    def test_bop(self):
        _assert_consistency(
            ta.bop(_open, _high, _low, _close, talib=False),
            ta.bop(_open, _high, _low, _close, talib=True),
            "BOP",
        )

    def test_aroon(self):
        _assert_consistency(
            ta.aroon(_high, _low, length=25, talib=False),
            ta.aroon(_high, _low, length=25, talib=True),
            "AROON",
        )

    @pytest.mark.xfail(reason="TRIX native fv=14, talib fv=40 (EMA chain stacking)")
    def test_trix(self):
        _assert_consistency(
            ta.trix(_close, length=14, talib=False),
            ta.trix(_close, length=14, talib=True),
            "TRIX",
        )

    @pytest.mark.xfail(reason="UO native fv=27, talib fv=28 (off-by-one)")
    def test_uo(self):
        _assert_consistency(
            ta.uo(_high, _low, _close, talib=False),
            ta.uo(_high, _low, _close, talib=True),
            "UO",
        )

    def test_stochrsi(self):
        _assert_consistency(
            ta.stochrsi(_close, length=14, talib=False),
            ta.stochrsi(_close, length=14, talib=True),
            "STOCHRSI",
        )


# ========================== VOLATILITY =====================================


class TestLookbackVolatility(unittest.TestCase):

    @pytest.mark.xfail(reason="ATR native fv=13, talib fv=14 (RMA off-by-one)")
    def test_atr(self):
        _assert_consistency(
            ta.atr(_high, _low, _close, length=14, talib=False),
            ta.atr(_high, _low, _close, length=14, talib=True),
            "ATR",
        )

    def test_natr(self):
        _assert_consistency(
            ta.natr(_high, _low, _close, length=14, talib=False),
            ta.natr(_high, _low, _close, length=14, talib=True),
            "NATR",
        )


# ========================== VOLUME =========================================


class TestLookbackVolume(unittest.TestCase):

    def test_ad(self):
        _assert_consistency(
            ta.ad(_high, _low, _close, _volume, talib=False),
            ta.ad(_high, _low, _close, _volume, talib=True),
            "AD",
        )

    def test_adosc(self):
        _assert_consistency(
            ta.adosc(_high, _low, _close, _volume, talib=False),
            ta.adosc(_high, _low, _close, _volume, talib=True),
            "ADOSC",
        )

    def test_obv(self):
        _assert_consistency(
            ta.obv(_close, _volume, talib=False),
            ta.obv(_close, _volume, talib=True),
            "OBV",
        )


# ========================== STATISTICS =====================================


class TestLookbackStatistics(unittest.TestCase):

    def test_beta(self):
        _assert_consistency(
            ta.beta(_close, _benchmark, length=5, talib=False),
            ta.beta(_close, _benchmark, length=5, talib=True),
            "BETA",
        )

    def test_correl(self):
        _assert_consistency(
            ta.correl(_close, _benchmark, length=30, talib=False),
            ta.correl(_close, _benchmark, length=30, talib=True),
            "CORREL",
        )

    def test_stdev(self):
        _assert_consistency(
            ta.stdev(_close, length=5, talib=False),
            ta.stdev(_close, length=5, talib=True),
            "STDDEV",
        )

    def test_variance(self):
        _assert_consistency(
            ta.variance(_close, length=5, talib=False),
            ta.variance(_close, length=5, talib=True),
            "VAR",
        )


# ========================== CYCLES =========================================


class TestLookbackCycles(unittest.TestCase):

    @pytest.mark.xfail(reason="HT_DCPERIOD native fv=0, talib fv=32 (Hilbert warmup)")
    def test_ht_dcperiod(self):
        _assert_consistency(
            ta.ht_dcperiod(_close, talib=False),
            ta.ht_dcperiod(_close, talib=True),
            "HT_DCPERIOD",
        )

    @pytest.mark.xfail(reason="HT_DCPHASE native fv=37, talib fv=63 (Hilbert warmup)")
    def test_ht_dcphase(self):
        _assert_consistency(
            ta.ht_dcphase(_close, talib=False),
            ta.ht_dcphase(_close, talib=True),
            "HT_DCPHASE",
        )

    @pytest.mark.xfail(reason="HT_PHASOR native fv=12, talib fv=32 (Hilbert warmup)")
    def test_ht_phasor(self):
        _assert_consistency(
            ta.ht_phasor(_close, talib=False),
            ta.ht_phasor(_close, talib=True),
            "HT_PHASOR",
        )

    @pytest.mark.xfail(reason="HT_SINE native fv=37, talib fv=63 (Hilbert warmup)")
    def test_ht_sine(self):
        _assert_consistency(
            ta.ht_sine(_close, talib=False),
            ta.ht_sine(_close, talib=True),
            "HT_SINE",
        )

    def test_ht_trendmode(self):
        _assert_consistency(
            ta.ht_trendmode(_close, talib=False),
            ta.ht_trendmode(_close, talib=True),
            "HT_TRENDMODE",
        )


# ========================== TREND ==========================================


class TestLookbackTrend(unittest.TestCase):

    @pytest.mark.xfail(reason="ADXR native fv=27, talib fv=40 (ADX seed difference)")
    def test_adxr(self):
        _assert_consistency(
            ta.adxr(_high, _low, _close, length=14, talib=False),
            ta.adxr(_high, _low, _close, length=14, talib=True),
            "ADXR",
        )

    def test_sarext(self):
        _assert_consistency(
            ta.sarext(_high, _low, talib=False),
            ta.sarext(_high, _low, talib=True),
            "SAREXT",
        )
