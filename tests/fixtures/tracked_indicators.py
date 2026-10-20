"""The tracked indicator set: every result ``test_indicator_values.py`` checks
against ``expected_values.json`` and ``test_regression.py`` against
``regression_snapshots.json``, computed on ``SPY_D``.
"""

from pathlib import Path

import pandas as pd

import pandas_ta_classic as ta

_DATA_PATH = Path(__file__).parent.parent.parent / "examples" / "data" / "SPY_D.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# Helper — recompute every indicator from the same definitions used in
# generate_fixtures.py so the two files stay in sync.
# ---------------------------------------------------------------------------


def compute_all(df: pd.DataFrame) -> dict[str, object]:
    o, h, low, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
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
        "midprice_14": ta.midprice(h, low, 14, talib=False),
        "t3_10": ta.t3(c, 10, talib=False),
        "wcp": ta.wcp(h, low, c, talib=False),
        "hl2": ta.hl2(h, low),
        "hlc3": ta.hlc3(h, low, c, talib=False),
        "ohlc4": ta.ohlc4(o, h, low, c),
        # ---- Overlap (native) ----
        "hma_10": ta.hma(c, length=10),
        "alma_10": ta.alma(c, length=10),
        "trima_10": ta.trima(c, length=10, talib=False),
        "fwma_10": ta.fwma(c, length=10),
        "hilo": ta.hilo(h, low, c),
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
        "supertrend": ta.supertrend(h, low, c),
        "swma": ta.swma(c),
        "tsf_14": ta.tsf(c, length=14),
        "vidya": ta.vidya(c),
        "vwap": ta.vwap(h, low, c, v),
        "vwma_10": ta.vwma(c, v, length=10),
        "zlma_10": ta.zlma(c, length=10),
        # ---- Momentum (TA-Lib backed) ----
        "rsi_14": ta.rsi(c, length=14, talib=False),
        "macd_12_26_9": ta.macd(c, fast=12, slow=26, signal=9, talib=False),
        "stoch": ta.stoch(h, low, c, talib=False),
        "cci_14": ta.cci(h, low, c, length=14, talib=False),
        "roc_10": ta.roc(c, length=10, talib=False),
        "willr_14": ta.willr(h, low, c, length=14, talib=False),
        "apo": ta.apo(c, 12, 26, talib=False),
        "bop": ta.bop(o, h, low, c, talib=False),
        "mom_10": ta.mom(c, length=10, talib=False),
        "uo": ta.uo(h, low, c, talib=False),
        # ---- Momentum (native) ----
        "ao": ta.ao(h, low),
        "bias_10": ta.bias(c, length=10),
        "brar": ta.brar(o, h, low, c),
        "cfo_10": ta.cfo(c, length=10),
        "cg": ta.cg(c),
        "cmo_14": ta.cmo(c, length=14, talib=False),
        "coppock": ta.coppock(c),
        "cti": ta.cti(c),
        "dm_14": ta.dm(h, low, length=14, talib=False),
        "er": ta.er(c),
        "eri": ta.eri(h, low, c),
        "fisher": ta.fisher(h, low),
        "inertia": ta.inertia(c, h, low),
        "kdj": ta.kdj(h, low, c),
        "kst": ta.kst(c),
        "lrsi": ta.lrsi(c),
        "pgo_14": ta.pgo(h, low, c, length=14),
        "po": ta.po(c),
        "ppo": ta.ppo(c),
        "psl": ta.psl(c, o),
        "pvo": ta.pvo(v),
        "qqe": ta.qqe(c),
        "rsx": ta.rsx(c),
        "rvgi": ta.rvgi(o, h, low, c),
        "slope": ta.slope(c),
        "smi": ta.smi(c),
        "squeeze": ta.squeeze(h, low, c),
        "squeeze_pro": ta.squeeze_pro(h, low, c),
        "stc": ta.stc(c),
        "stochrsi": ta.stochrsi(c),
        "td_seq": ta.td_seq(c),
        "trix": ta.trix(c),
        "trixh": ta.trixh(c),
        "tsi": ta.tsi(c),
        "vwmacd": ta.vwmacd(c, v),
        # ---- Volatility (TA-Lib backed) ----
        "atr_14": ta.atr(h, low, c, length=14, talib=False),
        "bbands_20": ta.bbands(c, length=20, talib=False),
        "natr_14": ta.natr(h, low, c, length=14, talib=False),
        "true_range": ta.true_range(h, low, c, talib=False),
        # ---- Volatility (native) ----
        "donchian_20": ta.donchian(h, low, lower_length=20, upper_length=20),
        "kc_20": ta.kc(h, low, c, length=20),
        "aberration": ta.aberration(h, low, c),
        "accbands": ta.accbands(h, low, c),
        "ce": ta.ce(h, low, c),
        "hwc": ta.hwc(c),
        "massi": ta.massi(h, low),
        "pdist": ta.pdist(o, h, low, c),
        "rvi_vol": ta.rvi(c, h, low),
        "thermo": ta.thermo(h, low),
        "ui": ta.ui(c),
        # ---- Trend (TA-Lib backed) ----
        "adx_14": ta.adx(h, low, c, length=14, talib=False),
        "aroon_14": ta.aroon(h, low, length=14, talib=False),
        # ---- Trend (native) ----
        "psar": ta.psar(h, low, c),
        "decreasing": ta.decreasing(c),
        "increasing": ta.increasing(c),
        "adxr": ta.adxr(h, low, c, length=14, talib=False),
        "amat": ta.amat(c, fast=8, slow=21),
        "chop": ta.chop(h, low, c),
        "cksp": ta.cksp(h, low, c),
        "cpr": ta.cpr(o, h, low, c),
        "decay": ta.decay(c),
        "dpo_14": ta.dpo(c, length=14),
        "long_run": ta.long_run(_ema_fast, _ema_slow),
        "pmax": ta.pmax(h, low, c),
        "qstick": ta.qstick(o, c),
        "short_run": ta.short_run(_ema_fast, _ema_slow),
        "ttm_trend": ta.ttm_trend(h, low, c),
        "vhf": ta.vhf(c),
        "vortex": ta.vortex(h, low, c),
        # ---- Volume (TA-Lib backed) ----
        "obv": ta.obv(c, v, talib=False),
        "mfi_14": ta.mfi(h, low, c, v, length=14, talib=False),
        "ad": ta.ad(h, low, c, v, talib=False),
        "adosc": ta.adosc(h, low, c, v, talib=False),
        # ---- Volume (native) ----
        "cmf_20": ta.cmf(h, low, c, v, length=20),
        "aobv": ta.aobv(c, v),
        "efi": ta.efi(c, v),
        "eom": ta.eom(h, low, c, v),
        "kvo": ta.kvo(h, low, c, v),
        "nvi": ta.nvi(c, v),
        "pvi": ta.pvi(c, v),
        "pvol": ta.pvol(c, v),
        "pvr": ta.pvr(c, v),
        "pvt": ta.pvt(c, v),
        "vfi": ta.vfi(h, low, c, v),
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
        "ha": ta.ha(o, h, low, c),
        "cdl_doji": ta.cdl_doji(o, h, low, c),
        "cdl_inside": ta.cdl_inside(o, h, low, c),
        "cdl_z": ta.cdl_z(o, h, low, c),
        # ---- Overlap (additional) ----
        "ichimoku": ta.ichimoku(h, low, c),
        "ma_ema_20": ta.ma("ema", c, length=20),
        # ---- Trend (additional) ----
        "tsignals": ta.tsignals(_trend_bool),
        "xsignals": ta.xsignals(_rsi_14_sig, 70, 30),
        # ---- Volume (additional) ----
        "vp": ta.vp(c, v),
        # ---- Candles — CDL patterns (via dispatcher) ----
        "cdl_pattern": ta.cdl_pattern(o, h, low, c),
        "cdl_2crows": ta.cdl_pattern(o, h, low, c, name="2crows"),
        "cdl_3blackcrows": ta.cdl_pattern(o, h, low, c, name="3blackcrows"),
        "cdl_3inside": ta.cdl_pattern(o, h, low, c, name="3inside"),
        "cdl_3linestrike": ta.cdl_pattern(o, h, low, c, name="3linestrike"),
        "cdl_3outside": ta.cdl_pattern(o, h, low, c, name="3outside"),
        "cdl_3starsinsouth": ta.cdl_pattern(o, h, low, c, name="3starsinsouth"),
        "cdl_3whitesoldiers": ta.cdl_pattern(o, h, low, c, name="3whitesoldiers"),
        "cdl_abandonedbaby": ta.cdl_pattern(o, h, low, c, name="abandonedbaby"),
        "cdl_advanceblock": ta.cdl_pattern(o, h, low, c, name="advanceblock"),
        "cdl_belthold": ta.cdl_pattern(o, h, low, c, name="belthold"),
        "cdl_breakaway": ta.cdl_pattern(o, h, low, c, name="breakaway"),
        "cdl_closingmarubozu": ta.cdl_pattern(o, h, low, c, name="closingmarubozu"),
        "cdl_concealbabyswall": ta.cdl_pattern(o, h, low, c, name="concealbabyswall"),
        "cdl_counterattack": ta.cdl_pattern(o, h, low, c, name="counterattack"),
        "cdl_darkcloudcover": ta.cdl_pattern(o, h, low, c, name="darkcloudcover"),
        "cdl_dojistar": ta.cdl_pattern(o, h, low, c, name="dojistar"),
        "cdl_dragonflydoji": ta.cdl_pattern(o, h, low, c, name="dragonflydoji"),
        "cdl_engulfing": ta.cdl_pattern(o, h, low, c, name="engulfing"),
        "cdl_eveningdojistar": ta.cdl_pattern(o, h, low, c, name="eveningdojistar"),
        "cdl_eveningstar": ta.cdl_pattern(o, h, low, c, name="eveningstar"),
        "cdl_gapsidesidewhite": ta.cdl_pattern(o, h, low, c, name="gapsidesidewhite"),
        "cdl_gravestonedoji": ta.cdl_pattern(o, h, low, c, name="gravestonedoji"),
        "cdl_hammer": ta.cdl_pattern(o, h, low, c, name="hammer"),
        "cdl_hangingman": ta.cdl_pattern(o, h, low, c, name="hangingman"),
        "cdl_harami": ta.cdl_pattern(o, h, low, c, name="harami"),
        "cdl_haramicross": ta.cdl_pattern(o, h, low, c, name="haramicross"),
        "cdl_highwave": ta.cdl_pattern(o, h, low, c, name="highwave"),
        "cdl_hikkake": ta.cdl_pattern(o, h, low, c, name="hikkake"),
        "cdl_hikkakemod": ta.cdl_pattern(o, h, low, c, name="hikkakemod"),
        "cdl_homingpigeon": ta.cdl_pattern(o, h, low, c, name="homingpigeon"),
        "cdl_identical3crows": ta.cdl_pattern(o, h, low, c, name="identical3crows"),
        "cdl_inneck": ta.cdl_pattern(o, h, low, c, name="inneck"),
        "cdl_invertedhammer": ta.cdl_pattern(o, h, low, c, name="invertedhammer"),
        "cdl_kicking": ta.cdl_pattern(o, h, low, c, name="kicking"),
        "cdl_kickingbylength": ta.cdl_pattern(o, h, low, c, name="kickingbylength"),
        "cdl_ladderbottom": ta.cdl_pattern(o, h, low, c, name="ladderbottom"),
        "cdl_longleggeddoji": ta.cdl_pattern(o, h, low, c, name="longleggeddoji"),
        "cdl_longline": ta.cdl_pattern(o, h, low, c, name="longline"),
        "cdl_marubozu": ta.cdl_pattern(o, h, low, c, name="marubozu"),
        "cdl_matchinglow": ta.cdl_pattern(o, h, low, c, name="matchinglow"),
        "cdl_mathold": ta.cdl_pattern(o, h, low, c, name="mathold"),
        "cdl_morningdojistar": ta.cdl_pattern(o, h, low, c, name="morningdojistar"),
        "cdl_morningstar": ta.cdl_pattern(o, h, low, c, name="morningstar"),
        "cdl_onneck": ta.cdl_pattern(o, h, low, c, name="onneck"),
        "cdl_piercing": ta.cdl_pattern(o, h, low, c, name="piercing"),
        "cdl_rickshawman": ta.cdl_pattern(o, h, low, c, name="rickshawman"),
        "cdl_risefall3methods": ta.cdl_pattern(o, h, low, c, name="risefall3methods"),
        "cdl_separatinglines": ta.cdl_pattern(o, h, low, c, name="separatinglines"),
        "cdl_shootingstar": ta.cdl_pattern(o, h, low, c, name="shootingstar"),
        "cdl_shortline": ta.cdl_pattern(o, h, low, c, name="shortline"),
        "cdl_spinningtop": ta.cdl_pattern(o, h, low, c, name="spinningtop"),
        "cdl_stalledpattern": ta.cdl_pattern(o, h, low, c, name="stalledpattern"),
        "cdl_sticksandwich": ta.cdl_pattern(o, h, low, c, name="sticksandwich"),
        "cdl_takuri": ta.cdl_pattern(o, h, low, c, name="takuri"),
        "cdl_tasukigap": ta.cdl_pattern(o, h, low, c, name="tasukigap"),
        "cdl_thrusting": ta.cdl_pattern(o, h, low, c, name="thrusting"),
        "cdl_tristar": ta.cdl_pattern(o, h, low, c, name="tristar"),
        "cdl_unique3river": ta.cdl_pattern(o, h, low, c, name="unique3river"),
        "cdl_upsidegap2crows": ta.cdl_pattern(o, h, low, c, name="upsidegap2crows"),
        "cdl_xsidegap3methods": ta.cdl_pattern(o, h, low, c, name="xsidegap3methods"),
    }
