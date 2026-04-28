"""
Generate golden fixture values for test_indicator_values.py.

Run this script manually whenever indicator algorithms are intentionally changed:

    python -m tests.fixtures.generate_fixtures

The script writes tests/fixtures/expected_values.json. Commit the updated file
alongside the algorithm change so CI continues to pass.

Reference strategy
------------------
Fixtures are split into two categories:

REFERENCE (121 indicators)
    Expected values are computed from the TA-Lib C library (``import talib``)
    or from direct arithmetic — both are entirely independent of the
    pandas-ta-classic code under test.  A test failure here means the native
    implementation has diverged from the mathematically expected result.

    52 CDL patterns use TA-Lib as oracle (exact element-wise match confirmed).
    8 CDL patterns never fire on SPY data so TA-Lib still provides a valid
    independent oracle (both TA-Lib and pandas-ta return all-zero on this dataset).
    Statistics (zscore, kurtosis, skew, median, quantile, mad, entropy) use
    pure pandas/numpy rolling formulas.  correl uses talib.CORREL; slope uses
    talib.LINEARREG_SLOPE.

REGRESSION (102 indicators)
    No independent reference implementation exists for these purely-native
    indicators.  Their expected values ARE computed from pandas-ta-classic
    itself.  These tests detect *unintended changes* between releases, but
    they do NOT verify mathematical correctness against an external standard.
    Regenerate after any intentional algorithm change.

    Notes on excluded TA-Lib candidates:
      TRIMA   — TA-Lib and pandas-ta use different (both valid) weighting
                formulas (~0.19% divergence).
      CMO     — native formula differs from TA-Lib (~2.8% divergence).
      DM      — native normalisation produces very different values from TA-Lib.
      PSARaf  — no TA-Lib equivalent; PSARr has off-by-one vs TA-Lib.
      beta    — algorithm differs from TA-Lib BETA (18.4% divergence).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import talib

import pandas_ta_classic as ta

# ---------------------------------------------------------------------------
# Load sample data
# ---------------------------------------------------------------------------
_DATA_PATH = Path(__file__).parent.parent.parent / "examples" / "data" / "SPY_D.csv"


def _load() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------------


def _indicators(df: pd.DataFrame) -> list[tuple[str, object]]:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    idx = df.index
    cv, hv, lv, ov, vv = c.values, h.values, l.values, o.values, v.values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _s(arr, name: str) -> pd.Series:
        """Wrap a NumPy array in a labelled Series sharing df's DatetimeIndex."""
        return pd.Series(arr, index=idx, name=name)

    def _df(**cols) -> pd.DataFrame:
        return pd.DataFrame({k: pd.Series(v, index=idx) for k, v in cols.items()})

    # Pre-compute multi-output TA-Lib calls so we can reference them below.
    _macd_line, _macd_sig, _macd_hist = talib.MACD(
        cv, fastperiod=12, slowperiod=26, signalperiod=9
    )
    _stoch_k, _stoch_d = talib.STOCH(
        hv,
        lv,
        cv,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    _bb_upper, _bb_mid, _bb_lower = talib.BBANDS(
        cv, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    _bb_bw = (_bb_upper - _bb_lower) / _bb_mid * 100  # bandwidth
    _bb_pct = (cv - _bb_lower) / (_bb_upper - _bb_lower)  # percent
    _adx_arr = talib.ADX(hv, lv, cv, timeperiod=14)
    _dmp_arr = talib.PLUS_DI(hv, lv, cv, timeperiod=14)
    _dmn_arr = talib.MINUS_DI(hv, lv, cv, timeperiod=14)
    _aroon_dn, _aroon_up = talib.AROON(hv, lv, timeperiod=14)
    _aroon_osc = _aroon_up - _aroon_dn
    _ht_i, _ht_q = talib.HT_PHASOR(cv)
    _ht_s, _ht_l = talib.HT_SINE(cv)
    # Additional TA-Lib pre-computes for newly-moved reference indicators
    _stochrsi_k, _stochrsi_d = talib.STOCHRSI(cv, 14, 3, 3)
    _ppo_line = talib.PPO(cv, 12, 26, 0)
    _ppo_sig  = talib.EMA(_ppo_line, 9)
    _ppo_hist = _ppo_line - _ppo_sig
    _trix_line = talib.TRIX(cv, 30)
    _trix_sig  = talib.SMA(_trix_line, 9)
    _sar       = talib.SAR(hv, lv, 0.02, 0.2)
    _sar_long  = _sar < cv                                  # long when SAR < price
    _adxr_arr  = talib.ADXR(hv, lv, cv, 14)
    # Pure-pandas rolling oracles for statistics
    _zscore_ref    = (c - c.rolling(20).mean()) / c.rolling(20).std()
    _kurt_ref      = c.rolling(20).kurt()
    _skew_ref      = c.rolling(20).skew()
    _median_ref    = c.rolling(14).median()
    _quantile_ref  = c.rolling(14).quantile(0.5)
    _mad_ref       = c.rolling(10).apply(
        lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True
    )
    def _rolling_entropy(s, n):
        def _ent(x):
            x = np.abs(x); p = x / x.sum(); p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))
        return s.rolling(n).apply(_ent, raw=True)
    _entropy_ref = _rolling_entropy(c, 10)

    # Pre-compute pandas-ta results used as count_ref for reference indicators.
    _ema_fast = ta.ema(c, length=8, talib=False)
    _ema_slow = ta.ema(c, length=21, talib=False)
    _trend_bool = _ema_fast > _ema_slow  # bool trend for tsignals
    _rsi_14_sig = ta.rsi(c, length=14, talib=False)  # signal for xsignals

    # Each tuple: (fixture_key, value_ref, count_source)
    #   value_ref    — TA-Lib / pure-math Series or DataFrame → used for last_value
    #   count_source — pandas-ta-classic result → used for non_nan_count
    #                  (None means use value_ref for both; for regression entries
    #                   both come from pandas-ta so count_source is also None)

    # ==================================================================
    # REFERENCE — independent of pandas-ta-classic
    # TA-Lib C library or plain arithmetic is the oracle here.
    # ==================================================================
    ref_indicators: list[tuple] = [
        # ---- Overlap ----
        ("sma_20", _s(talib.SMA(cv, 20), "SMA_20"), ta.sma(c, length=20)),
        ("ema_20", _s(talib.EMA(cv, 20), "EMA_20"), ta.ema(c, length=20, talib=False)),
        (
            "dema_10",
            _s(talib.DEMA(cv, 10), "DEMA_10"),
            ta.dema(c, length=10, talib=False),
        ),
        (
            "tema_10",
            _s(talib.TEMA(cv, 10), "TEMA_10"),
            ta.tema(c, length=10, talib=False),
        ),
        ("wma_10", _s(talib.WMA(cv, 10), "WMA_10"), ta.wma(c, length=10, talib=False)),
        (
            "midpoint_14",
            _s(talib.MIDPOINT(cv, 14), "MIDPOINT_14"),
            ta.midpoint(c, 14, talib=False),
        ),
        (
            "midprice_14",
            _s(talib.MIDPRICE(hv, lv, 14), "MIDPRICE_14"),
            ta.midprice(h, l, 14, talib=False),
        ),
        ("t3_10", _s(talib.T3(cv, 10, 0.7), "T3_10_0.7"), ta.t3(c, 10, talib=False)),
        ("wcp", _s(talib.WCLPRICE(hv, lv, cv), "WCP"), ta.wcp(h, l, c, talib=False)),
        # Overlap — pure arithmetic (zero library dependency)
        ("hl2", _s((hv + lv) / 2, "HL2"), ta.hl2(h, l)),
        ("hlc3", _s((hv + lv + cv) / 3, "HLC3"), ta.hlc3(h, l, c, talib=False)),
        ("ohlc4", _s((ov + hv + lv + cv) / 4, "OHLC4"), ta.ohlc4(o, h, l, c)),
        # ---- Momentum ----
        ("rsi_14", _s(talib.RSI(cv, 14), "RSI_14"), ta.rsi(c, length=14, talib=False)),
        (
            "macd_12_26_9",
            _df(
                MACD_12_26_9=_macd_line,
                MACDh_12_26_9=_macd_hist,
                MACDs_12_26_9=_macd_sig,
            ),
            ta.macd(c, fast=12, slow=26, signal=9, talib=False),
        ),
        (
            "stoch",
            _df(STOCHk_14_3_3=_stoch_k, STOCHd_14_3_3=_stoch_d),
            ta.stoch(h, l, c),
        ),
        (
            "cci_14",
            _s(talib.CCI(hv, lv, cv, 14), "CCI_14_0.015"),
            ta.cci(h, l, c, length=14, talib=False),
        ),
        ("roc_10", _s(talib.ROC(cv, 10), "ROC_10"), ta.roc(c, length=10, talib=False)),
        (
            "willr_14",
            _s(talib.WILLR(hv, lv, cv, 14), "WILLR_14"),
            ta.willr(h, l, c, length=14, talib=False),
        ),
        (
            "apo",
            _s(talib.APO(cv, 12, 26, 0), "APO_12_26"),
            ta.apo(c, 12, 26, talib=False),
        ),
        ("bop", _s(talib.BOP(ov, hv, lv, cv), "BOP"), ta.bop(o, h, l, c, talib=False)),
        ("mom_10", _s(talib.MOM(cv, 10), "MOM_10"), ta.mom(c, length=10, talib=False)),
        (
            "uo",
            _s(talib.ULTOSC(hv, lv, cv, 7, 14, 28), "UO_7_14_28"),
            ta.uo(h, l, c, talib=False),
        ),
        # ---- Volatility ----
        (
            "atr_14",
            _s(talib.ATR(hv, lv, cv, 14), "ATRr_14"),
            ta.atr(h, l, c, length=14, talib=False),
        ),
        (
            "bbands_20",
            _df(
                **{
                    "BBL_20_2.0": _bb_lower,
                    "BBM_20_2.0": _bb_mid,
                    "BBU_20_2.0": _bb_upper,
                    "BBB_20_2.0": _bb_bw,
                    "BBP_20_2.0": _bb_pct,
                }
            ),
            ta.bbands(c, length=20, talib=False),
        ),
        (
            "natr_14",
            _s(talib.NATR(hv, lv, cv, 14), "NATR_14"),
            ta.natr(h, l, c, length=14, talib=False),
        ),
        (
            "true_range",
            _s(talib.TRANGE(hv, lv, cv), "TRUERANGE_1"),
            ta.true_range(h, l, c, talib=False),
        ),
        # ---- Trend ----
        (
            "adx_14",
            _df(ADX_14=_adx_arr, DMP_14=_dmp_arr, DMN_14=_dmn_arr),
            ta.adx(h, l, c, length=14, talib=False),
        ),
        (
            "aroon_14",
            _df(AROOND_14=_aroon_dn, AROONU_14=_aroon_up, AROONOSC_14=_aroon_osc),
            ta.aroon(h, l, length=14, talib=False),
        ),
        # ---- Volume ----
        ("obv", _s(talib.OBV(cv, vv), "OBV"), ta.obv(c, v, talib=False)),
        (
            "mfi_14",
            _s(talib.MFI(hv, lv, cv, vv, 14), "MFI_14"),
            ta.mfi(h, l, c, v, length=14, talib=False),
        ),
        ("ad", _s(talib.AD(hv, lv, cv, vv), "AD"), ta.ad(h, l, c, v, talib=False)),
        (
            "adosc",
            _s(talib.ADOSC(hv, lv, cv, vv, 3, 10), "ADOSC_3_10"),
            ta.adosc(h, l, c, v, talib=False),
        ),
        # ---- Statistics ----
        (
            "stdev_20",
            _s(talib.STDDEV(cv, 20), "STDEV_20"),
            ta.stdev(c, length=20, talib=False),
        ),
        (
            "variance_20",
            _s(talib.VAR(cv, 20, 1), "VAR_20"),
            ta.variance(c, length=20, talib=False),
        ),
        # ---- Cycles — all wrap TA-Lib Hilbert Transform ----
        ("ht_dcperiod", _s(talib.HT_DCPERIOD(cv), "HT_DCPERIOD"), ta.ht_dcperiod(c)),
        ("ht_dcphase", _s(talib.HT_DCPHASE(cv), "HT_DCPHASE"), ta.ht_dcphase(c)),
        (
            "ht_phasor",
            _df(HT_PHASOR_INPHASE=_ht_i, HT_PHASOR_QUAD=_ht_q),
            ta.ht_phasor(c),
        ),
        ("ht_sine", _df(HT_SINE=_ht_s, HT_LEADSINE=_ht_l), ta.ht_sine(c)),
        (
            "ht_trendmode",
            _s(talib.HT_TRENDMODE(cv).astype(float), "HT_TRENDMODE"),
            ta.ht_trendmode(c),
        ),
        (
            "ht_trendline",
            _s(talib.HT_TRENDLINE(cv), "HT_TRENDLINE"),
            ta.ht_trendline(c),
        ),
        # ---- Performance — pure arithmetic ----
        (
            "log_return",
            _s(np.concatenate([[np.nan], np.log(cv[1:] / cv[:-1])]), "LOGRET_1"),
            ta.log_return(c),
        ),
        (
            "percent_return",
            _s(np.concatenate([[np.nan], (cv[1:] - cv[:-1]) / cv[:-1]]), "PCTRET_1"),
            ta.percent_return(c),
        ),
        # ---- Overlap (additional) ----
        ("kama",      _s(talib.KAMA(cv, 10), "KAMA_10_2_30"),           ta.kama(c)),
        ("mama",      _df(**{"MAMA_0.5_0.05": talib.MAMA(cv, 0.5, 0.05)[0],
                             "FAMA_0.5_0.05": talib.MAMA(cv, 0.5, 0.05)[1]}),
                                                                        ta.mama(c)),
        ("linreg_14", _s(talib.LINEARREG(cv, 14), "LR_14"),             ta.linreg(c, 14)),
        ("tsf_14",    _s(talib.TSF(cv, 14), "TSF_14"),                  ta.tsf(c, 14)),
        # ---- Momentum (additional) ----
        ("stochrsi",  _df(**{"STOCHRSIk_14_14_3_3": _stochrsi_k,
                             "STOCHRSId_14_14_3_3": _stochrsi_d}),      ta.stochrsi(c)),
        ("ppo",       _df(**{"PPO_12_26_9":  _ppo_line,
                             "PPOs_12_26_9": _ppo_sig,
                             "PPOh_12_26_9": _ppo_hist}),               ta.ppo(c)),
        ("trix",      _df(**{"TRIX_30_9":  _trix_line,
                             "TRIXs_30_9": _trix_sig}),                 ta.trix(c)),
        # ---- Trend (additional) ----
        ("adxr",      _df(**{"ADXR_14": _adxr_arr,
                             "DMP_14":   _dmp_arr,
                             "DMN_14":   _dmn_arr}),                    ta.adxr(h, l, c, 14)),
        ("psar",      _df(**{"PSARl_0.02_0.2": np.where(_sar_long, _sar, np.nan).astype(float),
                             "PSARs_0.02_0.2": np.where(~_sar_long, _sar, np.nan).astype(float)}),
                                                                        ta.psar(h, l, c)),
        # ---- Statistics (additional) — pure pandas/numpy rolling formulas ----
        ("zscore_20",   _s(_zscore_ref,   "ZS_20"),     ta.zscore(c, length=20)),
        ("kurtosis_20", _s(_kurt_ref,     "KURT_20"),   ta.kurtosis(c, length=20)),
        ("skew_20",     _s(_skew_ref,     "SKEW_20"),   ta.skew(c, length=20)),
        ("median_14",   _s(_median_ref,   "MEDIAN_14"), ta.median(c, length=14)),
        ("quantile_14", _s(_quantile_ref, "QTL_14_0.5"),ta.quantile(c, length=14)),
        ("mad_10",      _s(_mad_ref,      "MAD_10"),    ta.mad(c, length=10)),
        ("entropy_10",  _s(_entropy_ref,  "ENTP_10"),   ta.entropy(c, length=10)),
        ("correl",      _s(talib.CORREL(cv, ov, 30), "CORREL_30"),      ta.correl(c, o)),
        ("slope",       _s(talib.LINEARREG_SLOPE(cv, 2), "SLOPE_1"),    ta.slope(c)),
        # ---- CDL patterns — 52 exact TA-Lib matches ----
        ("cdl_2crows",           _s(talib.CDL2CROWS(ov, hv, lv, cv).astype(float),           "CDL_2CROWS"),           ta.cdl_pattern(o, h, l, c, name="2crows")),
        ("cdl_3blackcrows",      _s(talib.CDL3BLACKCROWS(ov, hv, lv, cv).astype(float),      "CDL_3BLACKCROWS"),      ta.cdl_pattern(o, h, l, c, name="3blackcrows")),
        ("cdl_3inside",          _s(talib.CDL3INSIDE(ov, hv, lv, cv).astype(float),          "CDL_3INSIDE"),          ta.cdl_pattern(o, h, l, c, name="3inside")),
        ("cdl_3linestrike",      _s(talib.CDL3LINESTRIKE(ov, hv, lv, cv).astype(float),      "CDL_3LINESTRIKE"),      ta.cdl_pattern(o, h, l, c, name="3linestrike")),
        ("cdl_3outside",         _s(talib.CDL3OUTSIDE(ov, hv, lv, cv).astype(float),         "CDL_3OUTSIDE"),         ta.cdl_pattern(o, h, l, c, name="3outside")),
        ("cdl_3whitesoldiers",   _s(talib.CDL3WHITESOLDIERS(ov, hv, lv, cv).astype(float),   "CDL_3WHITESOLDIERS"),   ta.cdl_pattern(o, h, l, c, name="3whitesoldiers")),
        ("cdl_advanceblock",     _s(talib.CDLADVANCEBLOCK(ov, hv, lv, cv).astype(float),     "CDL_ADVANCEBLOCK"),     ta.cdl_pattern(o, h, l, c, name="advanceblock")),
        ("cdl_belthold",         _s(talib.CDLBELTHOLD(ov, hv, lv, cv).astype(float),         "CDL_BELTHOLD"),         ta.cdl_pattern(o, h, l, c, name="belthold")),
        ("cdl_closingmarubozu",  _s(talib.CDLCLOSINGMARUBOZU(ov, hv, lv, cv).astype(float),  "CDL_CLOSINGMARUBOZU"),  ta.cdl_pattern(o, h, l, c, name="closingmarubozu")),
        ("cdl_counterattack",    _s(talib.CDLCOUNTERATTACK(ov, hv, lv, cv).astype(float),    "CDL_COUNTERATTACK"),    ta.cdl_pattern(o, h, l, c, name="counterattack")),
        ("cdl_darkcloudcover",   _s(talib.CDLDARKCLOUDCOVER(ov, hv, lv, cv).astype(float),   "CDL_DARKCLOUDCOVER"),   ta.cdl_pattern(o, h, l, c, name="darkcloudcover")),
        ("cdl_dojistar",         _s(talib.CDLDOJISTAR(ov, hv, lv, cv).astype(float),         "CDL_DOJISTAR"),         ta.cdl_pattern(o, h, l, c, name="dojistar")),
        ("cdl_dragonflydoji",    _s(talib.CDLDRAGONFLYDOJI(ov, hv, lv, cv).astype(float),    "CDL_DRAGONFLYDOJI"),    ta.cdl_pattern(o, h, l, c, name="dragonflydoji")),
        ("cdl_engulfing",        _s(talib.CDLENGULFING(ov, hv, lv, cv).astype(float),        "CDL_ENGULFING"),        ta.cdl_pattern(o, h, l, c, name="engulfing")),
        ("cdl_eveningdojistar",  _s(talib.CDLEVENINGDOJISTAR(ov, hv, lv, cv).astype(float),  "CDL_EVENINGDOJISTAR"),  ta.cdl_pattern(o, h, l, c, name="eveningdojistar")),
        ("cdl_eveningstar",      _s(talib.CDLEVENINGSTAR(ov, hv, lv, cv).astype(float),      "CDL_EVENINGSTAR"),      ta.cdl_pattern(o, h, l, c, name="eveningstar")),
        ("cdl_gapsidesidewhite", _s(talib.CDLGAPSIDESIDEWHITE(ov, hv, lv, cv).astype(float), "CDL_GAPSIDESIDEWHITE"), ta.cdl_pattern(o, h, l, c, name="gapsidesidewhite")),
        ("cdl_gravestonedoji",   _s(talib.CDLGRAVESTONEDOJI(ov, hv, lv, cv).astype(float),   "CDL_GRAVESTONEDOJI"),   ta.cdl_pattern(o, h, l, c, name="gravestonedoji")),
        ("cdl_hammer",           _s(talib.CDLHAMMER(ov, hv, lv, cv).astype(float),           "CDL_HAMMER"),           ta.cdl_pattern(o, h, l, c, name="hammer")),
        ("cdl_hangingman",       _s(talib.CDLHANGINGMAN(ov, hv, lv, cv).astype(float),       "CDL_HANGINGMAN"),       ta.cdl_pattern(o, h, l, c, name="hangingman")),
        ("cdl_harami",           _s(talib.CDLHARAMI(ov, hv, lv, cv).astype(float),           "CDL_HARAMI"),           ta.cdl_pattern(o, h, l, c, name="harami")),
        ("cdl_haramicross",      _s(talib.CDLHARAMICROSS(ov, hv, lv, cv).astype(float),      "CDL_HARAMICROSS"),      ta.cdl_pattern(o, h, l, c, name="haramicross")),
        ("cdl_highwave",         _s(talib.CDLHIGHWAVE(ov, hv, lv, cv).astype(float),         "CDL_HIGHWAVE"),         ta.cdl_pattern(o, h, l, c, name="highwave")),
        ("cdl_hikkake",          _s(talib.CDLHIKKAKE(ov, hv, lv, cv).astype(float),          "CDL_HIKKAKE"),          ta.cdl_pattern(o, h, l, c, name="hikkake")),
        ("cdl_hikkakemod",       _s(talib.CDLHIKKAKEMOD(ov, hv, lv, cv).astype(float),       "CDL_HIKKAKEMOD"),       ta.cdl_pattern(o, h, l, c, name="hikkakemod")),
        ("cdl_homingpigeon",     _s(talib.CDLHOMINGPIGEON(ov, hv, lv, cv).astype(float),     "CDL_HOMINGPIGEON"),     ta.cdl_pattern(o, h, l, c, name="homingpigeon")),
        ("cdl_identical3crows",  _s(talib.CDLIDENTICAL3CROWS(ov, hv, lv, cv).astype(float),  "CDL_IDENTICAL3CROWS"),  ta.cdl_pattern(o, h, l, c, name="identical3crows")),
        ("cdl_inneck",           _s(talib.CDLINNECK(ov, hv, lv, cv).astype(float),           "CDL_INNECK"),           ta.cdl_pattern(o, h, l, c, name="inneck")),
        ("cdl_invertedhammer",   _s(talib.CDLINVERTEDHAMMER(ov, hv, lv, cv).astype(float),   "CDL_INVERTEDHAMMER"),   ta.cdl_pattern(o, h, l, c, name="invertedhammer")),
        ("cdl_ladderbottom",     _s(talib.CDLLADDERBOTTOM(ov, hv, lv, cv).astype(float),     "CDL_LADDERBOTTOM"),     ta.cdl_pattern(o, h, l, c, name="ladderbottom")),
        ("cdl_longleggeddoji",   _s(talib.CDLLONGLEGGEDDOJI(ov, hv, lv, cv).astype(float),   "CDL_LONGLEGGEDDOJI"),   ta.cdl_pattern(o, h, l, c, name="longleggeddoji")),
        ("cdl_longline",         _s(talib.CDLLONGLINE(ov, hv, lv, cv).astype(float),         "CDL_LONGLINE"),         ta.cdl_pattern(o, h, l, c, name="longline")),
        ("cdl_marubozu",         _s(talib.CDLMARUBOZU(ov, hv, lv, cv).astype(float),         "CDL_MARUBOZU"),         ta.cdl_pattern(o, h, l, c, name="marubozu")),
        ("cdl_matchinglow",      _s(talib.CDLMATCHINGLOW(ov, hv, lv, cv).astype(float),      "CDL_MATCHINGLOW"),      ta.cdl_pattern(o, h, l, c, name="matchinglow")),
        ("cdl_morningdojistar",  _s(talib.CDLMORNINGDOJISTAR(ov, hv, lv, cv).astype(float),  "CDL_MORNINGDOJISTAR"),  ta.cdl_pattern(o, h, l, c, name="morningdojistar")),
        ("cdl_morningstar",      _s(talib.CDLMORNINGSTAR(ov, hv, lv, cv).astype(float),      "CDL_MORNINGSTAR"),      ta.cdl_pattern(o, h, l, c, name="morningstar")),
        ("cdl_onneck",           _s(talib.CDLONNECK(ov, hv, lv, cv).astype(float),           "CDL_ONNECK"),           ta.cdl_pattern(o, h, l, c, name="onneck")),
        ("cdl_piercing",         _s(talib.CDLPIERCING(ov, hv, lv, cv).astype(float),         "CDL_PIERCING"),         ta.cdl_pattern(o, h, l, c, name="piercing")),
        ("cdl_rickshawman",      _s(talib.CDLRICKSHAWMAN(ov, hv, lv, cv).astype(float),      "CDL_RICKSHAWMAN"),      ta.cdl_pattern(o, h, l, c, name="rickshawman")),
        ("cdl_separatinglines",  _s(talib.CDLSEPARATINGLINES(ov, hv, lv, cv).astype(float),  "CDL_SEPARATINGLINES"),  ta.cdl_pattern(o, h, l, c, name="separatinglines")),
        ("cdl_shootingstar",     _s(talib.CDLSHOOTINGSTAR(ov, hv, lv, cv).astype(float),     "CDL_SHOOTINGSTAR"),     ta.cdl_pattern(o, h, l, c, name="shootingstar")),
        ("cdl_shortline",        _s(talib.CDLSHORTLINE(ov, hv, lv, cv).astype(float),        "CDL_SHORTLINE"),        ta.cdl_pattern(o, h, l, c, name="shortline")),
        ("cdl_spinningtop",      _s(talib.CDLSPINNINGTOP(ov, hv, lv, cv).astype(float),      "CDL_SPINNINGTOP"),      ta.cdl_pattern(o, h, l, c, name="spinningtop")),
        ("cdl_stalledpattern",   _s(talib.CDLSTALLEDPATTERN(ov, hv, lv, cv).astype(float),   "CDL_STALLEDPATTERN"),   ta.cdl_pattern(o, h, l, c, name="stalledpattern")),
        ("cdl_sticksandwich",    _s(talib.CDLSTICKSANDWICH(ov, hv, lv, cv).astype(float),    "CDL_STICKSANDWICH"),    ta.cdl_pattern(o, h, l, c, name="sticksandwich")),
        ("cdl_takuri",           _s(talib.CDLTAKURI(ov, hv, lv, cv).astype(float),           "CDL_TAKURI"),           ta.cdl_pattern(o, h, l, c, name="takuri")),
        ("cdl_tasukigap",        _s(talib.CDLTASUKIGAP(ov, hv, lv, cv).astype(float),        "CDL_TASUKIGAP"),        ta.cdl_pattern(o, h, l, c, name="tasukigap")),
        ("cdl_thrusting",        _s(talib.CDLTHRUSTING(ov, hv, lv, cv).astype(float),        "CDL_THRUSTING"),        ta.cdl_pattern(o, h, l, c, name="thrusting")),
        ("cdl_tristar",          _s(talib.CDLTRISTAR(ov, hv, lv, cv).astype(float),          "CDL_TRISTAR"),          ta.cdl_pattern(o, h, l, c, name="tristar")),
        ("cdl_unique3river",     _s(talib.CDLUNIQUE3RIVER(ov, hv, lv, cv).astype(float),     "CDL_UNIQUE3RIVER"),     ta.cdl_pattern(o, h, l, c, name="unique3river")),
        ("cdl_upsidegap2crows",  _s(talib.CDLUPSIDEGAP2CROWS(ov, hv, lv, cv).astype(float),  "CDL_UPSIDEGAP2CROWS"),  ta.cdl_pattern(o, h, l, c, name="upsidegap2crows")),
        ("cdl_xsidegap3methods", _s(talib.CDLXSIDEGAP3METHODS(ov, hv, lv, cv).astype(float), "CDL_XSIDEGAP3METHODS"), ta.cdl_pattern(o, h, l, c, name="xsidegap3methods")),
        # ---- CDL patterns — 8 no-signal patterns (TA-Lib also returns all-zero) ----
        ("cdl_3starsinsouth",    _s(talib.CDL3STARSINSOUTH(ov, hv, lv, cv).astype(float),    "CDL_3STARSINSOUTH"),    ta.cdl_pattern(o, h, l, c, name="3starsinsouth")),
        ("cdl_abandonedbaby",    _s(talib.CDLABANDONEDBABY(ov, hv, lv, cv).astype(float),    "CDL_ABANDONEDBABY"),    ta.cdl_pattern(o, h, l, c, name="abandonedbaby")),
        ("cdl_breakaway",        _s(talib.CDLBREAKAWAY(ov, hv, lv, cv).astype(float),        "CDL_BREAKAWAY"),        ta.cdl_pattern(o, h, l, c, name="breakaway")),
        ("cdl_concealbabyswall", _s(talib.CDLCONCEALBABYSWALL(ov, hv, lv, cv).astype(float), "CDL_CONCEALBABYSWALL"), ta.cdl_pattern(o, h, l, c, name="concealbabyswall")),
        ("cdl_kicking",          _s(talib.CDLKICKING(ov, hv, lv, cv).astype(float),          "CDL_KICKING"),          ta.cdl_pattern(o, h, l, c, name="kicking")),
        ("cdl_kickingbylength",  _s(talib.CDLKICKINGBYLENGTH(ov, hv, lv, cv).astype(float),  "CDL_KICKINGBYLENGTH"),  ta.cdl_pattern(o, h, l, c, name="kickingbylength")),
        ("cdl_mathold",          _s(talib.CDLMATHOLD(ov, hv, lv, cv).astype(float),          "CDL_MATHOLD"),          ta.cdl_pattern(o, h, l, c, name="mathold")),
        ("cdl_risefall3methods", _s(talib.CDLRISEFALL3METHODS(ov, hv, lv, cv).astype(float), "CDL_RISEFALL3METHODS"), ta.cdl_pattern(o, h, l, c, name="risefall3methods")),
    ]

    # ==================================================================
    # REGRESSION — no independent reference exists for these indicators.
    # These guard against unintended changes between releases but do NOT
    # verify mathematical correctness against an external standard.
    # ==================================================================
    reg_indicators: list[tuple] = [
        # ---- Overlap ----
        ("hma_10",      ta.hma(c, length=10),                              None),
        ("alma_10",     ta.alma(c, length=10),                             None),
        ("trima_10",    ta.trima(c, length=10, talib=False),               None),
        ("fwma_10",     ta.fwma(c, length=10),                             None),
        ("hilo",        ta.hilo(h, l, c),                                  None),
        ("hwma",        ta.hwma(c),                                        None),
        ("jma",         ta.jma(c),                                         None),
        ("mcgd",        ta.mcgd(c),                                        None),
        ("mmar",        ta.mmar(c),                                        None),
        ("pwma_10",     ta.pwma(c, length=10),                             None),
        ("rainbow",     ta.rainbow(c),                                     None),
        ("rma",         ta.rma(c),                                         None),
        ("sinwma_14",   ta.sinwma(c, length=14),                           None),
        ("ssf_10",      ta.ssf(c, length=10),                              None),
        ("supertrend",  ta.supertrend(h, l, c),                            None),
        ("swma",        ta.swma(c),                                        None),
        ("vidya",       ta.vidya(c),                                       None),
        ("vwap",        ta.vwap(h, l, c, v),                               None),
        ("vwma_10",     ta.vwma(c, v, length=10),                          None),
        ("zlma_10",     ta.zlma(c, length=10),                             None),
        # ---- Momentum ----
        ("ao",          ta.ao(h, l),                                       None),
        ("bias_10",     ta.bias(c, length=10),                             None),
        ("brar",        ta.brar(o, h, l, c),                               None),
        ("cfo_10",      ta.cfo(c, length=10),                              None),
        ("cg",          ta.cg(c),                                          None),
        ("cmo_14",      ta.cmo(c, length=14, talib=False),                 None),
        ("coppock",     ta.coppock(c),                                     None),
        ("cti",         ta.cti(c),                                         None),
        ("dm_14",       ta.dm(h, l, length=14, talib=False),               None),
        ("er",          ta.er(c),                                          None),
        ("eri",         ta.eri(h, l, c),                                   None),
        ("fisher",      ta.fisher(h, l),                                   None),
        ("inertia",     ta.inertia(c, h, l),                               None),
        ("kdj",         ta.kdj(h, l, c),                                   None),
        ("kst",         ta.kst(c),                                         None),
        ("lrsi",        ta.lrsi(c),                                        None),
        ("pgo_14",      ta.pgo(h, l, c, length=14),                        None),
        ("po",          ta.po(c),                                          None),
        ("psl",         ta.psl(c, o),                                      None),
        ("pvo",         ta.pvo(v),                                         None),
        ("qqe",         ta.qqe(c),                                         None),
        ("rsx",         ta.rsx(c),                                         None),
        ("rvgi",        ta.rvgi(o, h, l, c),                               None),
        ("smi",         ta.smi(c),                                         None),
        ("squeeze",     ta.squeeze(h, l, c),                               None),
        ("squeeze_pro", ta.squeeze_pro(h, l, c),                           None),
        ("stc",         ta.stc(c),                                         None),
        ("td_seq",      ta.td_seq(c),                                      None),
        ("trixh",       ta.trixh(c),                                       None),
        ("tsi",         ta.tsi(c),                                         None),
        ("vwmacd",      ta.vwmacd(c, v),                                   None),
        # ---- Volatility ----
        ("donchian_20", ta.donchian(h, l, lower_length=20, upper_length=20), None),
        ("kc_20",       ta.kc(h, l, c, length=20),                        None),
        ("aberration",  ta.aberration(h, l, c),                            None),
        ("accbands",    ta.accbands(h, l, c),                              None),
        ("ce",          ta.ce(h, l, c),                                    None),
        ("hwc",         ta.hwc(c),                                         None),
        ("massi",       ta.massi(h, l),                                    None),
        ("pdist",       ta.pdist(o, h, l, c),                              None),
        ("rvi_vol",     ta.rvi(c, h, l),                                   None),
        ("thermo",      ta.thermo(h, l),                                   None),
        ("ui",          ta.ui(c),                                          None),
        # ---- Trend ----
        ("decreasing",  ta.decreasing(c),                                  None),
        ("increasing",  ta.increasing(c),                                  None),
        ("amat",        ta.amat(c, fast=8, slow=21),                       None),
        ("chop",        ta.chop(h, l, c),                                  None),
        ("cksp",        ta.cksp(h, l, c),                                  None),
        ("cpr",         ta.cpr(o, h, l, c),                                None),
        ("decay",       ta.decay(c),                                       None),
        ("dpo_14",      ta.dpo(c, length=14),                              None),
        ("long_run",    ta.long_run(_ema_fast, _ema_slow),                 None),
        ("pmax",        ta.pmax(h, l, c),                                  None),
        ("qstick",      ta.qstick(o, c),                                   None),
        ("short_run",   ta.short_run(_ema_fast, _ema_slow),                None),
        ("ttm_trend",   ta.ttm_trend(h, l, c),                             None),
        ("vhf",         ta.vhf(c),                                         None),
        ("vortex",      ta.vortex(h, l, c),                                None),
        # ---- Volume ----
        ("cmf_20",      ta.cmf(h, l, c, v, length=20),                    None),
        ("aobv",        ta.aobv(c, v),                                     None),
        ("efi",         ta.efi(c, v),                                      None),
        ("eom",         ta.eom(h, l, c, v),                                None),
        ("kvo",         ta.kvo(h, l, c, v),                                None),
        ("nvi",         ta.nvi(c, v),                                      None),
        ("pvi",         ta.pvi(c, v),                                      None),
        ("pvol",        ta.pvol(c, v),                                     None),
        ("pvr",         ta.pvr(c, v),                                      None),
        ("pvt",         ta.pvt(c, v),                                      None),
        ("vfi",         ta.vfi(c, v),                                      None),
        # ---- Statistics ----
        ("beta",        ta.beta(c, o),                                     None),
        ("tos_stdevall",ta.tos_stdevall(c, length=30),                     None),
        # ---- Cycles ----
        ("dsp",         ta.dsp(c),                                         None),
        ("ebsw",        ta.ebsw(c),                                        None),
        # ---- Performance ----
        ("drawdown",    ta.drawdown(c),                                    None),
        # ---- Candles ----
        ("ha",          ta.ha(o, h, l, c),                                 None),
        ("cdl_doji",    ta.cdl_doji(o, h, l, c),                           None),
        # cdl_doji uses custom bodyPercent/lookback params that differ from TA-Lib
        ("cdl_inside",  ta.cdl_inside(o, h, l, c),                         None),
        ("cdl_z",       ta.cdl_z(o, h, l, c),                             None),
        # cdl_inside, cdl_z: no TA-Lib equivalent
        # cdl_pattern (dispatcher): all-pattern aggregate, no TA-Lib equivalent
        ("cdl_pattern", ta.cdl_pattern(o, h, l, c),                        None),
        # ---- Overlap (additional) ----
        ("ichimoku",    ta.ichimoku(h, l, c)[0],                            None),
        ("ma_ema_20",   ta.ma("ema", c, length=20),                         None),
        # ---- Trend (additional) ----
        ("tsignals",    ta.tsignals(_trend_bool),                           None),
        ("xsignals",    ta.xsignals(_rsi_14_sig, 70, 30),                  None),
        # ---- Volume (additional) ----
        ("vp",          ta.vp(c, v),                                        None),
    ]

    return ref_indicators + reg_indicators


# ---------------------------------------------------------------------------
# Serialise
# ---------------------------------------------------------------------------


def _serialise(value_ref, count_ref=None) -> dict:
    """Return {col: {last_value, non_nan_count}} for a Series or DataFrame.

    Parameters
    ----------
    value_ref:
        The REFERENCE result used for ``last_value`` (TA-Lib or pure math for
        reference indicators; the native result for regression indicators).
    count_ref:
        The pandas-ta-classic result used for ``non_nan_count``.  Allows the
        fixture to track the library's own warmup behaviour independently of
        the value reference.  When *None*, ``value_ref`` is used for both.
    """
    if count_ref is None:
        count_ref = value_ref

    if isinstance(value_ref, pd.Series):
        value_ref = value_ref.to_frame(name=value_ref.name)
    if isinstance(count_ref, pd.Series):
        count_ref = count_ref.to_frame(name=count_ref.name)

    out = {}
    for col in value_ref.columns:
        # Skip non-numeric columns (e.g. categorical labels returned by cpr)
        if not pd.api.types.is_numeric_dtype(value_ref[col]):
            continue

        # last_value from the independent reference
        last_clean = value_ref[col].dropna()
        if last_clean.empty:
            last_val = None
        else:
            v = float(last_clean.iloc[-1])
            last_val = None if math.isnan(v) or math.isinf(v) else round(v, 8)

        # non_nan_count from the native pandas-ta-classic run
        if col in count_ref.columns:
            non_nan = int(count_ref[col].notna().sum())
        else:
            non_nan = int(value_ref[col].notna().sum())

        out[col] = {"last_value": last_val, "non_nan_count": non_nan}
    return out


def generate() -> None:
    df = _load()
    fixtures: dict[str, dict] = {}

    for key, value_ref, count_ref in _indicators(df):
        if value_ref is None:
            print(f"  SKIP  {key!r:30s}  (returned None)")
            continue
        fixtures[key] = _serialise(value_ref, count_ref)
        col_summary = list(fixtures[key].keys())
        print(f"  OK    {key!r:30s}  cols={col_summary}")

    out_path = Path(__file__).parent / "expected_values.json"
    with open(out_path, "w") as fh:
        json.dump(fixtures, fh, indent=2)
    print(f"\nWrote {len(fixtures)} fixtures → {out_path}")


if __name__ == "__main__":
    generate()
