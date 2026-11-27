"""Every indicator with a TA-Lib counterpart agrees with it on every known data problem.

It replaced ``test_oracle_talib.py``, which compared 60 hand-picked calls on
``SPY_D`` only and intersected both sides *after* ``dropna()``, and the
``assert_talib`` calls of the category tests. Two classes of defect were
invisible to that comparison by construction:

* **Data it never sees.** ``SPY_D`` has no flat bar, no bodiless bar on a flat
  run, no zero-volume session and no malformed OHLC. ``kama`` was 0.085 off
  TA-Lib on a flat block and ``cdl_doji`` read 0 where ``CDLDOJI`` reads 100;
  neither showed up there.
* **NaN where TA-Lib has a value.** A ``dropna()`` intersection drops exactly
  the bars on which native gave up, so ``willr``'s all-NaN flat series passed.

This module runs one table, ``ORACLE``, over a set of frames that each plant
one data problem met in the field, and compares values *and* the NaN mask from
the first bar on which both sides are valid. ``test_every_talib_indicator_has_an_oracle``
keeps the table complete against ``docs/indicator_support_matrix.rst``: an
indicator the matrix lists with a TA-Lib counterpart must have a case here or a
reason in ``NO_ORACLE``. ``VARIANTS`` adds the other paths of those
indicators that TA-Lib expresses too (a moving-average choice, a smoothing,
sarext's options), and ``test_warmup_matches_talib`` requires every case to
publish its first value on TA-Lib's bar, except TA-Lib's own conventions in
``WARMUP_DIFFERENCES``.

Every candle pattern is compared against its ``CDL*`` function the same way,
and ``cdl_doji`` also with TA-Lib's candle settings changed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.assertions import TALIB_ATOL, TALIB_RTOL
from tests.contracts import params, spy

talib = pytest.importorskip("talib")

_ROOT = Path(__file__).parents[2]
_N = 400


# ---------------------------------------------------------------------------
# Frames: one known data problem each.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bars:
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series


def _bars(o, h, l, c, v, index) -> Bars:
    as_series = lambda x: pd.Series(np.asarray(x, dtype=float), index=index)
    return Bars(as_series(o), as_series(h), as_series(l), as_series(c), as_series(v))


def _spy() -> pd.DataFrame:
    return spy(_N).astype(float)


def _frame_spy() -> Bars:
    df = _spy()
    return _bars(df.open, df.high, df.low, df.close, df.volume, df.index)


def _frame_flat() -> Bars:
    """Every bar identical: no range, no body, no movement, no variance."""
    index = pd.date_range("2020-01-01", periods=_N, freq="D")
    x = np.full(_N, 0.5)
    return _bars(x, x, x, x, np.full(_N, 1000.0), index)


def _frame_flat_block() -> Bars:
    """Real data with a 60-bar block of one price in the middle."""
    df = _spy()
    o, h, l, c = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close"))
    block = slice(150, 210)
    price = c[block.start]
    for a in (o, h, l, c):
        a[block] = price
    return _bars(o, h, l, c, df.volume, df.index)


def _frame_no_range() -> Bars:
    """A moving close with ``high == low == open == close`` on every bar."""
    df = _spy()
    c = df.close.to_numpy()
    return _bars(c, c, c, c, df.volume, df.index)


def _frame_planted() -> Bars:
    """Real data with scattered flat bars, bodiless bars and short flat runs."""
    df = _spy()
    o, h, l, c = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close"))
    rng = np.random.default_rng(7)
    bodiless = rng.choice(np.arange(20, _N), 30, replace=False)
    o[bodiless] = c[bodiless]
    for i in rng.choice(np.arange(20, _N), 30, replace=False):
        o[i] = h[i] = l[i] = c[i]
    for start in (100, 260):
        o[start : start + 20] = h[start : start + 20] = l[start : start + 20] = c[start : start + 20] = c[start]
    return _bars(o, h, l, c, df.volume, df.index)


def _frame_malformed() -> Bars:
    """``high == low`` on a block while open and close sit off it.

    No valid feed produces this, nothing in the library rejects it, and it is
    the only frame on which a zero range meets a live numerator.
    """
    df = _spy()
    o, h, l, c = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close"))
    block = slice(150, 190)
    h[block] = l[block] = c[block.start]
    return _bars(o, h, l, c, df.volume, df.index)


def _frame_zero_volume() -> Bars:
    """Halted sessions: a block with no volume, some at one price."""
    df = _spy()
    o, h, l, c = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close"))
    v = df.volume.to_numpy(copy=True)
    v[150:190] = 0.0
    v[300:400:7] = 0.0  # isolated zero-volume bars too
    halted = slice(170, 180)
    o[halted] = h[halted] = l[halted] = c[halted] = c[halted.start]
    return _bars(o, h, l, c, v, df.index)


def _scaled(k: float) -> Callable[[], Bars]:
    """Real data in another price unit: a crypto pair (1e-6) or an index (1e6)."""

    def build() -> Bars:
        df = _spy()
        return _bars(df.open * k, df.high * k, df.low * k, df.close * k, df.volume, df.index)

    return build


def _frame_spike() -> Bars:
    """A flash spike to 10x on 1000x volume, and a flash crash to 0.1x."""
    df = _spy()
    o, h, l, c = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close"))
    v = df.volume.to_numpy(copy=True)
    for a in (o, h, l, c):
        a[200] *= 10
        a[260] *= 0.1
    v[200] *= 1000
    return _bars(o, h, l, c, v, df.index)


def _frame_huge_volume() -> Bars:
    """Volume in the 1e15-1e17 range, where running sums lose integer precision."""
    df = _spy()
    return _bars(df.open, df.high, df.low, df.close, df.volume * 1e9, df.index)


def _frame_nan_prefix() -> Bars:
    """37 missing bars at the start, as the output of an indicator fed to another one.

    37 is where the Hilbert transform starts its recursion (``ht_start``)."""
    df = _spy()
    o, h, l, c, v = (df[k].to_numpy(copy=True) for k in ("open", "high", "low", "close", "volume"))
    for a in (o, h, l, c, v):
        a[:37] = np.nan
    return _bars(o, h, l, c, v, df.index)


def _frame_spy_full() -> Bars:
    """All of SPY_D (5241 bars): a recursion that drifts away from TA-Lib does so over the long run."""
    df = spy().astype(float)
    return _bars(df.open, df.high, df.low, df.close, df.volume, df.index)


FRAMES: dict[str, Callable[[], Bars]] = {
    "spy": _frame_spy,
    "flat": _frame_flat,
    "flat_block": _frame_flat_block,
    "no_range": _frame_no_range,
    "planted": _frame_planted,
    "malformed": _frame_malformed,
    "zero_volume": _frame_zero_volume,
    "x1e-6": _scaled(1e-6),
    "x1e6": _scaled(1e6),
    "spike": _frame_spike,
    "huge_volume": _frame_huge_volume,
    "nan_prefix": _frame_nan_prefix,
    "spy_full": _frame_spy_full,
}


@pytest.fixture(scope="module")
def frames() -> dict[str, Bars]:
    return {name: build() for name, build in FRAMES.items()}


# ---------------------------------------------------------------------------
# The oracle table.
# ---------------------------------------------------------------------------

Columns = list[pd.Series | np.ndarray]


@dataclass(frozen=True)
class Case:
    native: Callable[[Bars], Columns]
    oracle: Callable[[Bars], Columns]
    # Bars before which the two seed differently and only converge. The
    # largest is 250 (the MACD family, which reads only close); every problem
    # frame still reaches it after bar 250: the spike at 200, the planted run
    # at 260, and the halted closes at 170-180, which the recursion carries on.
    settle: int = 0


def _cols(result, *prefixes: str) -> Columns:
    """Columns of a native DataFrame picked by name prefix, in the given order."""
    picked = []
    for prefix in prefixes:
        match = [c for c in result.columns if c == prefix] or [c for c in result.columns if c.startswith(prefix)]
        assert len(match) == 1, f"{prefix!r} matches {match} in {list(result.columns)}"
        picked.append(result[match[0]])
    return picked


def _psar_stop(b) -> Columns:
    """psar's long and short stops combined: the single SAR line TA-Lib returns."""
    long_, short = _cols(ta.psar(b.high, b.low, b.close), "PSARl_", "PSARs_")
    return [long_.combine_first(short)]


_KIJUN = 26


def _ichimoku_unshifted(b) -> Columns:
    """ichimoku's lines with both spans moved back to the bar they are computed on.

    Native publishes the spans ``kijun`` bars forward, where a chart draws them;
    TA-Lib returns them on the bar of their inputs.
    """
    tenkan, kijun, span_a, span_b = _cols(ta.ichimoku(b.high, b.low, b.close, 9, _KIJUN, 52), "ITS_", "IKS_", "ISA_", "ISB_")
    return [tenkan, kijun, span_a.shift(-_KIJUN), span_b.shift(-_KIJUN)]


def _ichimoku_talib(b) -> Columns:
    """TA-Lib's ICHIMOKU without the spans of the last ``kijun`` bars, which native has not published yet."""
    tenkan, kijun, span_a, span_b = T.ICHIMOKU(b.high, b.low, 9, _KIJUN, 52)
    for span in (span_a, span_b):
        span[-_KIJUN:] = np.nan
    return [tenkan, kijun, span_a, span_b]


T = talib
ORACLE: dict[str, Case] = {
    # overlap
    "sma": Case(lambda b: [ta.sma(b.close, 10)], lambda b: [T.SMA(b.close, 10)]),
    "ema": Case(lambda b: [ta.ema(b.close, 10)], lambda b: [T.EMA(b.close, 10)]),
    "wma": Case(lambda b: [ta.wma(b.close, 10)], lambda b: [T.WMA(b.close, 10)]),
    "dema": Case(lambda b: [ta.dema(b.close, 10)], lambda b: [T.DEMA(b.close, 10)]),
    "tema": Case(lambda b: [ta.tema(b.close, 10)], lambda b: [T.TEMA(b.close, 10)]),
    "trima": Case(lambda b: [ta.trima(b.close, 10)], lambda b: [T.TRIMA(b.close, 10)]),
    "kama": Case(lambda b: [ta.kama(b.close, 10)], lambda b: [T.KAMA(b.close, 10)]),
    "t3": Case(lambda b: [ta.t3(b.close, 5, a=0.7)], lambda b: [T.T3(b.close, 5, 0.7)]),
    "mama": Case(lambda b: _cols(ta.mama(b.close, fastlimit=0.5, slowlimit=0.05), "MAMA_", "FAMA_"), lambda b: list(T.MAMA(b.close, 0.5, 0.05))),
    "midpoint": Case(lambda b: [ta.midpoint(b.close, 14)], lambda b: [T.MIDPOINT(b.close, 14)]),
    "midprice": Case(lambda b: [ta.midprice(b.high, b.low, 14)], lambda b: [T.MIDPRICE(b.high, b.low, 14)]),
    "ht_trendline": Case(lambda b: [ta.ht_trendline(b.close)], lambda b: [T.HT_TRENDLINE(b.close)]),
    "linreg": Case(lambda b: [ta.linreg(b.close, 14)], lambda b: [T.LINEARREG(b.close, 14)]),
    "tsf": Case(lambda b: [ta.tsf(b.close, 14)], lambda b: [T.TSF(b.close, 14)]),
    "linregangle": Case(lambda b: [ta.linregangle(b.close, 14)], lambda b: [T.LINEARREG_ANGLE(b.close, 14)]),
    "linregintercept": Case(lambda b: [ta.linregintercept(b.close, 14)], lambda b: [T.LINEARREG_INTERCEPT(b.close, 14)]),
    "linregslope": Case(lambda b: [ta.linregslope(b.close, 14)], lambda b: [T.LINEARREG_SLOPE(b.close, 14)]),
    "avgprice": Case(lambda b: [ta.avgprice(b.open, b.high, b.low, b.close)], lambda b: [T.AVGPRICE(b.open, b.high, b.low, b.close)]),
    "ohlc4": Case(lambda b: [ta.ohlc4(b.open, b.high, b.low, b.close)], lambda b: [T.AVGPRICE(b.open, b.high, b.low, b.close)]),
    "medprice": Case(lambda b: [ta.medprice(b.high, b.low)], lambda b: [T.MEDPRICE(b.high, b.low)]),
    "hl2": Case(lambda b: [ta.hl2(b.high, b.low)], lambda b: [T.MEDPRICE(b.high, b.low)]),
    "typprice": Case(lambda b: [ta.typprice(b.high, b.low, b.close)], lambda b: [T.TYPPRICE(b.high, b.low, b.close)]),
    "hlc3": Case(lambda b: [ta.hlc3(b.high, b.low, b.close)], lambda b: [T.TYPPRICE(b.high, b.low, b.close)]),
    "wcp": Case(lambda b: [ta.wcp(b.high, b.low, b.close)], lambda b: [T.WCLPRICE(b.high, b.low, b.close)]),
    "mavp": Case(
        lambda b: [ta.mavp(b.close, pd.Series(10.0, index=b.close.index))],
        lambda b: [T.MAVP(b.close, np.full(b.close.size, 10.0), 2, 30, 0)],
    ),
    # momentum
    "apo": Case(lambda b: [ta.apo(b.close, 12, 26, mamode="sma")], lambda b: [T.APO(b.close, 12, 26, 0)]),
    "ppo": Case(lambda b: _cols(ta.ppo(b.close, 12, 26, mamode="sma"), "PPO_"), lambda b: [T.PPO(b.close, 12, 26, 0)]),
    "bop": Case(lambda b: [ta.bop(b.open, b.high, b.low, b.close)], lambda b: [T.BOP(b.open, b.high, b.low, b.close)]),
    "cci": Case(lambda b: [ta.cci(b.high, b.low, b.close, 14)], lambda b: [T.CCI(b.high, b.low, b.close, 14)]),
    # Native CMO defaults to Chande's sums; TA-Lib smooths with Wilder. talib=True
    # selects the Wilder path, and scalar=50 keeps it native (rule 6: TA-Lib
    # cannot express a scalar), so this compares our rma path, not TA-Lib to itself.
    "cmo": Case(lambda b: [ta.cmo(b.close, 14, scalar=50, talib=True)], lambda b: [T.CMO(b.close, 14) / 2]),
    # The MACD family seeds its EMAs differently from TA-Lib; the difference has
    # decayed to about 2e-12 relative by bar 250 (still 2.5e-7 at bar 154), so
    # the comparison starts there.
    # macdfix is MACD(12, 26) natively, not TA-Lib's fixed-alpha MACDFIX.
    "macd": Case(lambda b: _cols(ta.macd(b.close, 12, 26, 9), "MACD_", "MACDs_", "MACDh_"), lambda b: list(T.MACD(b.close, 12, 26, 9)), settle=250),
    "macdext": Case(
        lambda b: _cols(ta.macdext(b.close), "MACDEXT_", "MACDEXTs_", "MACDEXTh_"), lambda b: list(T.MACDEXT(b.close, 12, 1, 26, 1, 9, 1)), settle=250
    ),
    "macdfix": Case(
        lambda b: _cols(ta.macdfix(b.close, signal=9), "MACDFIX_", "MACDFIXs_", "MACDFIXh_"), lambda b: list(T.MACD(b.close, 12, 26, 9)), settle=250
    ),
    "mom": Case(lambda b: [ta.mom(b.close, 10)], lambda b: [T.MOM(b.close, 10)]),
    # The previous bar's TSF is the forecast for this bar, as in TA-Lib and Tulip (#344).
    "fosc": Case(lambda b: [ta.fosc(b.close, 14)], lambda b: [T.FOSC(b.close, 14)]),
    "roc": Case(lambda b: [ta.roc(b.close, 10)], lambda b: [T.ROC(b.close, 10)]),
    "rocp": Case(lambda b: [ta.rocp(b.close, 10)], lambda b: [T.ROCP(b.close, 10)]),
    "rocr": Case(lambda b: [ta.rocr(b.close, 10)], lambda b: [T.ROCR(b.close, 10)]),
    "rocr100": Case(lambda b: [ta.rocr100(b.close, 10)], lambda b: [T.ROCR100(b.close, 10)]),
    "rsi": Case(lambda b: [ta.rsi(b.close, 14)], lambda b: [T.RSI(b.close, 14)]),
    "stoch": Case(
        lambda b: _cols(ta.stoch(b.high, b.low, b.close, 14, 3, 3), "STOCHk_", "STOCHd_"),
        lambda b: list(T.STOCH(b.high, b.low, b.close, 14, 3, 0, 3, 0)),
    ),
    "stochf": Case(
        lambda b: _cols(ta.stochf(b.high, b.low, b.close, 14, 3), "STOCHFk_", "STOCHFd_"), lambda b: list(T.STOCHF(b.high, b.low, b.close, 14, 3, 0))
    ),
    "stochrsi": Case(
        lambda b: _cols(ta.stochrsi(b.close, length=14, rsi_length=14, k=1, d=3), "STOCHRSIk_", "STOCHRSId_"),
        lambda b: list(T.STOCHRSI(b.close, 14, 14, 3, 0)),
    ),
    "trix": Case(lambda b: _cols(ta.trix(b.close, 30), "TRIX_"), lambda b: [T.TRIX(b.close, 30)]),
    "willr": Case(lambda b: [ta.willr(b.high, b.low, b.close, 14)], lambda b: [T.WILLR(b.high, b.low, b.close, 14)]),
    "uo": Case(lambda b: [ta.uo(b.high, b.low, b.close, 7, 14, 28)], lambda b: [T.ULTOSC(b.high, b.low, b.close, 7, 14, 28)]),
    # trend
    "aroon": Case(
        lambda b: _cols(ta.aroon(b.high, b.low, 14), "AROOND_", "AROONU_", "AROONOSC_"),
        lambda b: [*T.AROON(b.high, b.low, 14), T.AROONOSC(b.high, b.low, 14)],
    ),
    "adx": Case(
        lambda b: _cols(ta.adx(b.high, b.low, b.close, 14), "ADX_", "DMP_", "DMN_"),
        lambda b: [T.ADX(b.high, b.low, b.close, 14), T.PLUS_DI(b.high, b.low, b.close, 14), T.MINUS_DI(b.high, b.low, b.close, 14)],
    ),
    "adxr": Case(lambda b: _cols(ta.adxr(b.high, b.low, b.close, 14), "ADXR_"), lambda b: [T.ADXR(b.high, b.low, b.close, 14)]),
    "dx": Case(lambda b: [ta.dx(b.high, b.low, b.close, 14)], lambda b: [T.DX(b.high, b.low, b.close, 14)]),
    "dm": Case(
        lambda b: _cols(ta.dm(b.high, b.low, 14), "PLUS_DM_", "MINUS_DM_"), lambda b: [T.PLUS_DM(b.high, b.low, 14), T.MINUS_DM(b.high, b.low, 14)]
    ),
    "plus_dm": Case(lambda b: [ta.plus_dm(b.high, b.low, 14)], lambda b: [T.PLUS_DM(b.high, b.low, 14)]),
    "minus_dm": Case(lambda b: [ta.minus_dm(b.high, b.low, 14)], lambda b: [T.MINUS_DM(b.high, b.low, 14)]),
    "sarext": Case(lambda b: [ta.sarext(b.high, b.low)], lambda b: [T.SAREXT(b.high, b.low)]),
    # DPO is TA-Lib's causal form: the price length // 2 + 1 bars ago less the SMA (#344).
    "dpo": Case(lambda b: [ta.dpo(b.close, 20, centered=False)], lambda b: [T.DPO(b.close, 20)]),
    "psar": Case(_psar_stop, lambda b: [T.SAR(b.high, b.low)]),
    # volatility
    "atr": Case(lambda b: [ta.atr(b.high, b.low, b.close, 14)], lambda b: [T.ATR(b.high, b.low, b.close, 14)]),
    "natr": Case(lambda b: [ta.natr(b.high, b.low, b.close, 14)], lambda b: [T.NATR(b.high, b.low, b.close, 14)]),
    "true_range": Case(lambda b: [ta.true_range(b.high, b.low, b.close)], lambda b: [T.TRANGE(b.high, b.low, b.close)]),
    "bbands": Case(lambda b: _cols(ta.bbands(b.close, 20, 2), "BBL_", "BBM_", "BBU_"), lambda b: list(T.BBANDS(b.close, 20, 2, 2, 0))[::-1]),
    # statistics
    "stdev": Case(lambda b: [ta.stdev(b.close, 20)], lambda b: [T.STDDEV(b.close, 20, 1)]),
    "variance": Case(lambda b: [ta.variance(b.close, 20)], lambda b: [T.VAR(b.close, 20, 1)]),
    "beta": Case(lambda b: [ta.beta(b.close, b.open, 5)], lambda b: [T.BETA(b.open, b.close, 5)]),
    "correl": Case(lambda b: [ta.correl(b.close, b.open, 30)], lambda b: [T.CORREL(b.close, b.open, 30)]),
    # volume
    "ad": Case(lambda b: [ta.ad(b.high, b.low, b.close, b.volume)], lambda b: [T.AD(b.high, b.low, b.close, b.volume)]),
    "adosc": Case(
        lambda b: [ta.adosc(b.high, b.low, b.close, b.volume, fast=3, slow=10)], lambda b: [T.ADOSC(b.high, b.low, b.close, b.volume, 3, 10)]
    ),
    "mfi": Case(lambda b: [ta.mfi(b.high, b.low, b.close, b.volume, 14)], lambda b: [T.MFI(b.high, b.low, b.close, b.volume, 14)]),
    "obv": Case(lambda b: [ta.obv(b.close, b.volume)], lambda b: [T.OBV(b.close, b.volume)]),
    "pvt": Case(lambda b: [ta.pvt(b.close, b.volume)], lambda b: [T.PVT(b.close, b.volume)]),
    "nvi": Case(lambda b: [ta.nvi(b.close, b.volume)], lambda b: [T.NVI(b.close, b.volume)]),
    "pvi": Case(lambda b: [ta.pvi(b.close, b.volume)], lambda b: [T.PVI(b.close, b.volume)]),
    # cycles
    "ht_dcperiod": Case(lambda b: [ta.ht_dcperiod(b.close)], lambda b: [T.HT_DCPERIOD(b.close)]),
    "ht_dcphase": Case(lambda b: [ta.ht_dcphase(b.close)], lambda b: [T.HT_DCPHASE(b.close)]),
    "ht_phasor": Case(lambda b: _cols(ta.ht_phasor(b.close), "HT_PHASOR_INPHASE", "HT_PHASOR_QUAD"), lambda b: list(T.HT_PHASOR(b.close))),
    "ht_sine": Case(lambda b: _cols(ta.ht_sine(b.close), "HT_SINE", "HT_LEADSINE"), lambda b: list(T.HT_SINE(b.close))),
    "ht_trendmode": Case(lambda b: [ta.ht_trendmode(b.close)], lambda b: [T.HT_TRENDMODE(b.close)]),
    # math
    "add": Case(lambda b: [ta.add(b.close, b.open)], lambda b: [T.ADD(b.close, b.open)]),
    "sub": Case(lambda b: [ta.sub(b.close, b.open)], lambda b: [T.SUB(b.close, b.open)]),
    "mult": Case(lambda b: [ta.mult(b.close, b.open)], lambda b: [T.MULT(b.close, b.open)]),
    "div": Case(lambda b: [ta.div(b.close, b.open)], lambda b: [T.DIV(b.close, b.open)]),
    "rolling_max": Case(lambda b: [ta.rolling_max(b.close, 10)], lambda b: [T.MAX(b.close, 10)]),
    "rolling_min": Case(lambda b: [ta.rolling_min(b.close, 10)], lambda b: [T.MIN(b.close, 10)]),
    "rolling_sum": Case(lambda b: [ta.rolling_sum(b.close, 10)], lambda b: [T.SUM(b.close, 10)]),
    "minmax": Case(lambda b: _cols(ta.minmax(b.close, 30), "MIN_", "MAX_"), lambda b: list(T.MINMAX(b.close, 30))),
    # Counterparts TA-Lib added in 0.8 (the floor is ta-lib>=0.8.3).
    # overlap
    "hma": Case(lambda b: [ta.hma(b.close, 20)], lambda b: [T.HMA(b.close, 20)]),
    "rma": Case(lambda b: [ta.rma(b.close, 30)], lambda b: [T.RMA(b.close, 30)]),
    "vwma": Case(lambda b: [ta.vwma(b.close, b.volume, 30)], lambda b: [T.VWMA(b.close, b.volume, 30)]),
    "zlma": Case(lambda b: [ta.zlma(b.close, 30)], lambda b: [T.ZLEMA(b.close, 30)]),
    "mcgd": Case(lambda b: [ta.mcgd(b.close, 14)], lambda b: [T.MCGD(b.close, 14)]),
    "ichimoku": Case(_ichimoku_unshifted, _ichimoku_talib),
    # momentum
    "ao": Case(lambda b: [ta.ao(b.high, b.low, 5, 34)], lambda b: [T.AO(b.high, b.low, 5, 34)]),
    "cg": Case(lambda b: [ta.cg(b.close, 10)], lambda b: [T.CG(b.close, 10)]),
    "coppock": Case(lambda b: [ta.coppock(b.close, 10, 11, 14)], lambda b: [T.COPPOCK(b.close, 10, 11, 14)]),
    "cti": Case(lambda b: [ta.cti(b.close, 20)], lambda b: [T.CTI(b.close, 20)]),
    "cvi": Case(lambda b: [ta.cvi(b.high, b.low, 10)], lambda b: [T.CVI(b.high, b.low, 10, 10)]),
    "er": Case(lambda b: [ta.er(b.close, 10)], lambda b: [T.ER(b.close, 10)]),
    "eri": Case(lambda b: _cols(ta.eri(b.high, b.low, b.close, 13), "BULLP_", "BEARP_"), lambda b: list(T.ERI(b.high, b.low, b.close, 13))),
    # The two seed the first value differently; they agree from bar 74 on every frame
    # but nan_prefix, whose 37 NaN bars come first (3.7e-11 apart at bar 110, equal
    # from 111); settle 150 leaves margin.
    "fisher": Case(
        lambda b: _cols(ta.fisher(b.high, b.low, 10, 1), "FISHERT_", "FISHERTs_"), lambda b: list(T.FISHER(b.high, b.low, 10)), settle=150
    ),
    # KDJ's matype 13 is the 1/n smoothing of the original KDJ.
    "kdj": Case(
        lambda b: _cols(ta.kdj(b.high, b.low, b.close, 9, 3), "K_", "D_", "J_"), lambda b: list(T.KDJ(b.high, b.low, b.close, 9, 3, 13, 3, 13))
    ),
    "kst": Case(lambda b: _cols(ta.kst(b.close), "KST_", "KSTs_"), lambda b: list(T.KST(b.close))),
    "qstick": Case(lambda b: [ta.qstick(b.open, b.close, 10)], lambda b: [T.QSTICK(b.open, b.close, 10)]),
    # STC's two stochastic stages amplify the seed difference of its MACD; it fades
    # slowest on nan_prefix, whose NaN bars come first (TOLERANCE).
    "stc": Case(lambda b: _cols(ta.stc(b.close, 10, 23, 50), "STC_"), lambda b: [T.STC(b.close, 23, 50, 10)], settle=320),
    "tsi": Case(lambda b: _cols(ta.tsi(b.close, 13, 25), "TSI_"), lambda b: [T.TSI(b.close, 25, 13)]),
    # trend
    "chop": Case(lambda b: [ta.chop(b.high, b.low, b.close, 14)], lambda b: [T.CHOP(b.high, b.low, b.close, 14)]),
    "cksp": Case(
        lambda b: _cols(ta.cksp(b.high, b.low, b.close, 10, 1.0, 9), "CKSPl_", "CKSPs_"), lambda b: list(T.CKSP(b.high, b.low, b.close, 10, 1.0, 9))
    ),
    "supertrend": Case(
        lambda b: _cols(ta.supertrend(b.high, b.low, b.close, 10, 3.0), "SUPERT_", "SUPERTd_"),
        lambda b: list(T.SUPERTREND(b.high, b.low, b.close, 10, 3.0)),
    ),
    "vhf": Case(lambda b: [ta.vhf(b.close, 28)], lambda b: [T.VHF(b.close, 28)]),
    "vortex": Case(lambda b: _cols(ta.vortex(b.high, b.low, b.close, 14), "VTXP_", "VTXM_"), lambda b: list(T.VORTEX(b.high, b.low, b.close, 14))),
    # volatility
    "accbands": Case(
        lambda b: _cols(ta.accbands(b.high, b.low, b.close, 20), "ACCBU_", "ACCBM_", "ACCBL_"), lambda b: list(T.ACCBANDS(b.high, b.low, b.close, 20))
    ),
    "donchian": Case(lambda b: _cols(ta.donchian(b.high, b.low, 20, 20), "DCU_", "DCM_", "DCL_"), lambda b: list(T.DONCHIAN(b.high, b.low, 20))),
    "massi": Case(lambda b: [ta.massi(b.high, b.low, 9, 25)], lambda b: [T.MASSI(b.high, b.low, 9, 25)]),
    # statistics
    "kurtosis": Case(lambda b: [ta.kurtosis(b.close, 30)], lambda b: [T.KURTOSIS(b.close, 30)]),
    "mad": Case(lambda b: [ta.mad(b.close, 14)], lambda b: [T.AVGDEV(b.close, 14)]),
    "median": Case(lambda b: [ta.median(b.close, 30)], lambda b: [T.MEDIAN(b.close, 30)]),
    # volume
    "cmf": Case(lambda b: [ta.cmf(b.high, b.low, b.close, b.volume, None, 20)], lambda b: [T.CMF(b.high, b.low, b.close, b.volume, 20)]),
    "efi": Case(lambda b: [ta.efi(b.close, b.volume, 13)], lambda b: [T.EFI(b.close, b.volume, 13)]),
    # emv is the one-bar ease of movement, EMV before its smoothing; eom smooths it (EMV's own default).
    "emv": Case(lambda b: [ta.emv(b.high, b.low, b.volume)], lambda b: [T.EMV(b.high, b.low, b.volume, 1, 10000.0)]),
    "eom": Case(lambda b: [ta.eom(b.high, b.low, b.close, b.volume, 14, divisor=10000)], lambda b: [T.EMV(b.high, b.low, b.volume, 14, 10000.0)]),
    "marketfi": Case(lambda b: [ta.marketfi(b.high, b.low, b.volume)], lambda b: [T.MARKETFI(b.high, b.low, b.volume)]),
    "pvo": Case(lambda b: _cols(ta.pvo(b.volume, 12, 26), "PVO_"), lambda b: [T.PVO(b.volume, 12, 26, 1)]),
    "wad": Case(lambda b: [ta.wad(b.high, b.low, b.close)], lambda b: [T.WAD(b.high, b.low, b.close)]),
    # candles
    "ha": Case(
        lambda b: _cols(ta.ha(b.open, b.high, b.low, b.close), "HA_open", "HA_high", "HA_low", "HA_close"),
        lambda b: list(T.HA(b.open, b.high, b.low, b.close)),
    ),
}

# Elementwise math transforms map one to one onto a TA-Lib function of the same
# name. They take a ratio in their domain (asin/acos need [-1, 1]).
_TRANSFORMS = {
    "acos": "ACOS", "asin": "ASIN", "atan": "ATAN", "ceil": "CEIL", "cos": "COS", "cosh": "COSH",
    "exp": "EXP", "floor": "FLOOR", "ln": "LN", "log10": "LOG10", "sin": "SIN", "sinh": "SINH",
    "sqrt": "SQRT", "tan": "TAN", "tanh": "TANH",
}  # fmt: skip


def _ratio(b: Bars) -> pd.Series:
    return (b.close / b.close.max()).clip(-1, 1)


for _name, _fn in _TRANSFORMS.items():
    ORACLE[_name] = Case(
        lambda b, n=_name: [getattr(ta, n)(_ratio(b))],
        lambda b, f=_fn: [getattr(T, f)(_ratio(b))],
    )

# Listed with a TA-Lib counterpart in docs/indicator_support_matrix.rst, but not
# the same function, so a value comparison would test the wrong thing.
NO_ORACLE = {
    "ma": "a dispatcher; every MA it reaches has its own case",
    "alma": "OPEN: native's weights are mirrored: its offset 0.85 puts the peak on the oldest bars of the window, TA-Lib's ALMA "
    "on the newest (and floors the centre to a bar); expected_values.json freezes the native values",
    "kc": "KC is EMA(typical price, 20) +- 2 * ATR(10); native uses one length for the basis of close and the band of the true range",
    "kvo": "KVO's volume force uses Klinger's trend and cumulative measurement; native signs the volume by the change of hlc3",
    "rvi": "RVI takes the deviation (10) and its smoothing (14) over two windows; native uses one length for both, "
    "and with equal windows the two still differ (up to 16.8 on SPY_D at 14, 14)",
    "smi": "a different indicator: TA-Lib's SMI is Blau's Stochastic Momentum Index, native smi is the SMI Ergodic of the TSI",
    "vidya": "TALIB: TA-Lib seeds VIDYA one bar later (bar 14, native 13); with the CMO period equal to length the seeds "
    "meet to 1e-8 only after 160 to 410 bars and to 1e-13 after 800, longer than the 400-bar frames; test_vidya_converges_to_talib "
    "compares SPY_D",
    "vwap": "VWAP is cumulative from the first bar; native anchors to a calendar period and needs a DatetimeIndex",
}

# A second path of an indicator in ORACLE that TA-Lib can express too, keyed
# "<indicator>-<option>[-<value>]"; compared on every frame like ORACLE.
VARIANTS: dict[str, Case] = {
    "apo-mamode-ema": Case(lambda b: [ta.apo(b.close, 12, 26, mamode="ema")], lambda b: [T.APO(b.close, 12, 26, 1)]),
    "ppo-mamode-ema": Case(lambda b: _cols(ta.ppo(b.close, 12, 26, mamode="ema"), "PPO_"), lambda b: [T.PPO(b.close, 12, 26, 1)]),
    # stochrsi's default k=3 smooths %K with an SMA of 3: TA-Lib's fast %D.
    "stochrsi-k-3": Case(
        lambda b: _cols(ta.stochrsi(b.close, length=14, rsi_length=14, k=3, d=3), "STOCHRSIk_"), lambda b: [T.STOCHRSI(b.close, 14, 14, 3, 0)[1]]
    ),
}

# sarext's options, each family on its own (offsetonreverse, startvalue and the
# long acceleration factors diverged from TA-Lib); "init-above-max" starts the
# acceleration above its maximum, which both sides clamp.
SAREXT_OPTIONS = {
    "offsetonreverse": {"offsetonreverse": 0.01},
    "startvalue-long": {"startvalue": 100.0},
    "startvalue-short": {"startvalue": -150.0},
    "long-acceleration": {"accelerationinitlong": 0.01, "accelerationlong": 0.03, "accelerationmaxlong": 0.25},
    "short-acceleration": {"accelerationinitshort": 0.04, "accelerationshort": 0.01, "accelerationmaxshort": 0.1},
    "init-above-max": {"accelerationinitlong": 0.5, "accelerationmaxlong": 0.2, "accelerationshort": 0.5, "accelerationmaxshort": 0.3},
}
for _option, _kwargs in SAREXT_OPTIONS.items():
    VARIANTS[f"sarext-{_option}"] = Case(lambda b, k=_kwargs: [ta.sarext(b.high, b.low, **k)], lambda b, k=_kwargs: [T.SAREXT(b.high, b.low, **k)])


# ---------------------------------------------------------------------------
# Comparison.
# ---------------------------------------------------------------------------


def _first_common_valid(native: np.ndarray, oracle: np.ndarray) -> int | None:
    both = np.flatnonzero(np.isfinite(native) & np.isfinite(oracle))
    return None if both.size == 0 else int(both[0])


def _disagreement(native, oracle, settle: int = 0, tol: float = 1.0) -> str | None:
    """Describe the first disagreement, or None when the two agree.

    *tol* multiplies TALIB_ATOL and TALIB_RTOL (TOLERANCE says where and why).

    Compared from the first bar on which both sides hold a finite value, so a
    warm-up that differs by a bar is not a finding, but NaN on one side and a
    value on the other after that is.
    """
    native = np.asarray(native, dtype=float)
    oracle = np.asarray(oracle, dtype=float)
    assert native.shape == oracle.shape, f"shape {native.shape} vs {oracle.shape}"
    start = _first_common_valid(native, oracle)
    if start is not None:
        start = max(start, settle)
    if start is None:
        if np.isfinite(native).any() or np.isfinite(oracle).any():
            return f"no common valid bar: native has {int(np.isfinite(native).sum())} values, TA-Lib {int(np.isfinite(oracle).sum())}"
        return None
    n, o = native[start:], oracle[start:]
    nan_mismatch = np.flatnonzero(np.isnan(n) != np.isnan(o))
    if nan_mismatch.size:
        i = nan_mismatch[0]
        return f"bar {start + i}: native {n[i]!r}, TA-Lib {o[i]!r} ({nan_mismatch.size} NaN-mask mismatches)"
    with np.errstate(invalid="ignore"):
        # TALIB_ATOL + TALIB_RTOL * |expected|, element by element, with the absolute term
        # scaled down to columns smaller than 1: at prices of 1e-4 a fixed 1e-11
        # would pass a 1e-8 relative error, and a variance of 1e-13 any error. A
        # column TA-Lib reads as all zero has no scale and keeps the plain TALIB_ATOL.
        scale = float(np.nanmax(np.abs(o[np.isfinite(o)]))) if np.isfinite(o).any() else 0.0
        scale = scale or 1.0
        off = ~((n == o) | (np.isnan(n) & np.isnan(o)) | (np.abs(n - o) <= tol * (TALIB_ATOL * min(1.0, scale) + TALIB_RTOL * np.abs(o))))
    if off.any():
        i = int(np.flatnonzero(off)[0])
        return f"bar {start + i}: native {n[i]!r}, TA-Lib {o[i]!r}, max abs diff {np.nanmax(np.abs(n - o)):.3e} on {int(off.sum())} bars"
    return None


# (indicator, frame) pairs where native deliberately differs from TA-Lib, with
# the reason. A pair listed here that starts agreeing fails (strict xfail), so
# the list cannot go stale.
_TALIB_CCI_RESIDUE = (
    "TALIB: TA-Lib's running-sum mean leaves 1.1e-13 on a window of one typical price, so its mean "
    "deviation and tp - mean are both that residue and CCI reads +-1/0.015 = +-66.67; native "
    "reads the 0.0 marker"
)
_STOCHRSI_NOISE = (
    "TALIB: RSI is constant on a flat block in exact arithmetic but wobbles by 1e-14; TA-Lib 0.7's STOCHRSI "
    "divides that wobble by itself and reads noise (33, 40, ...), native and TA-Lib 0.8 treat the "
    "residue range as no range and read the 0.0 marker"
)
_HILBERT_ON_A_CONSTANT = (
    "KEPT: the Hilbert transform of a constant input has no dominant cycle; both sides amplify "
    "rounding (equal to 1e-8, then up to 19 apart), so neither value means anything"
)
_ZERO_MARKER = "KEPT: native reads the 0.0 marker on {what} (degenerate-window convention, test_degenerate_input.py); TA-Lib reads {talib}"
_CHOP_DIVISION = (
    "KEPT: where the highs outlive the lows of a flat block, the true range sums to a value over a zero range, a real "
    "division by zero that reads inf (test_degenerate_input.py _REAL_DIVISION_BY_ZERO); TA-Lib reads 100"
)
# TA-Lib changed between 0.7 and 0.8 (measured with 0.7.1 and 0.8.2, the
# version CI installs): 0.8 fixed the CCI residue, CORREL's zero test and
# STOCHRSI's flat-block noise (as native stochrsi does), and
# clamps WILLR to -100..0 where malformed bars put the close outside the range.
# The behaviour is the C library's, read from it: a source build can link a C
# library of another version than the Python wrapper (wrapper 0.8.2 ships C 0.8.1).
_TA_C_VERSION = talib.__ta_version__.decode().split()[0]
_TALIB_08 = tuple(int(part) for part in _TA_C_VERSION.split(".")[:2]) >= (0, 8)

# Cases whose arithmetic needs more than TALIB_ATOL + TALIB_RTOL * |expected|: the
# factor on both terms, and the worst bar measured on Windows and Linux, Python 3.11 to
# 3.15, numba on and off (TA-Lib 0.8.3; every variant reads the same worst bar). Each
# factor is the 1-2-5 step that leaves at least 10x margin; a case cancels digits
# where its value is small against the rest of its column.
# test_every_tolerance_is_needed fails an entry that the case no longer needs.
TOLERANCE: dict[str, tuple[float, str]] = {
    "adosc": (5000, "a difference of two EMAs of AD, which reaches 2.7e8: 2.5e-9 relative at 194 (flat_block, bar 209)"),
    "beta": (2000, "the regression over the window that holds the spike: 5e-9 relative at -0.52 (spike, bar 227)"),
    "linregangle": (200, "the arctan of a slope that cancels to 0.005 degrees: 3.8e-8 relative (nan_prefix, bar 251)"),
    "ad": (50, "a cumulative sum that reaches 3.8e10: 9e-7 absolute at 19627 (spy_full, bar 76)"),
    "fosc": (50, "close less its forecast, 0.24 percent: 1.9e-10 relative (spy_full, bar 2252)"),
    "ht_dcphase": (50, "the Hilbert transform's long recursion: 1.7e-10 at -5.05 degrees (spy_full, bar 3940)"),
    "ht_sine": (20, "the Hilbert transform's long recursion: 1.6e-11 at 0.166 (spy_full, bar 2546)"),
    "variance": (
        20,
        "TA-Lib's running sums of x and x**2 cancel digits in sum(x**2) - sum(x)**2 / n: 6.1e-11 at 3.98 in a column up to 3.7e5 (spike, bar 220)",
    ),
    "cci": (10, "a mean deviation near 0: 6e-11 relative at -0.198 (planted, bar 109)"),
    "ht_phasor": (10, "the Hilbert transform's long recursion: 2.2e-10 at 22.0 (flat_block, bar 214)"),
    "correl": (10, "TA-Lib's running sums: 2.3e-11 relative at 0.58 (spy_full, bar 4452)"),
    "linregslope": (10, "a slope that cancels to -0.038: 1.9e-10 relative (flat_block, bar 360)"),
    "ht_dcperiod": (10, "the Hilbert transform's long recursion: 1.5e-10 at 25.0 (flat_block, bar 217)"),
    "stdev": (10, "TA-Lib's running sums, as variance: 7.7e-12 relative at 1.99 in a column up to 607 (spike, bar 220)"),
    "apo-mamode-ema": (5, "the EMA seeds converging, at prices of 1e6: 2.3e-12 relative (x1e6, bar 71)"),
    "macd": (2, "the histogram, a difference near 0, after the EMA seeds: 1.9e-12 relative at -28096 (x1e6, bar 267)"),
    "macdext": (2, "as macd"),
    "macdfix": (2, "as macd"),
    "linreg": (2, "2e-12 relative at -4.8 in a column up to 1030 (spike, bar 213)"),
    "tsf": (2, "1.2e-12 relative at 8.74 in a column up to 1110 (spike, bar 212)"),
    "cti": (20, "a correlation that cancels to -0.38 over the planted run: 4.6e-11 relative (planted, bar 118)"),
    "eri": (1000, "the bear power, low less an EMA at prices of 1e6: 5.8e-10 relative at 308 (x1e6, bar 376)"),
    "stc": (100, "its MACD's seed difference, amplified by two stochastic stages, still 1.8e-10 relative (nan_prefix, bar 359)"),
}

KNOWN_DIFFERENCES: dict[tuple[str, str], str] = {
    ("ht_dcperiod", "flat"): _HILBERT_ON_A_CONSTANT,
    ("ht_dcphase", "flat"): _HILBERT_ON_A_CONSTANT,
    ("ht_dcphase", "flat_block"): _HILBERT_ON_A_CONSTANT,
    ("ht_sine", "flat"): _HILBERT_ON_A_CONSTANT,
    ("ht_sine", "flat_block"): _HILBERT_ON_A_CONSTANT,
    ("ht_trendmode", "flat"): _HILBERT_ON_A_CONSTANT,
    ("ht_trendmode", "flat_block"): _HILBERT_ON_A_CONSTANT,
    ("mama", "flat_block"): _HILBERT_ON_A_CONSTANT,
    ("mama", "planted"): _HILBERT_ON_A_CONSTANT,
    ("chop", "flat"): _ZERO_MARKER.format(what="a window with no true range and no range", talib="100"),
    ("chop", "flat_block"): _CHOP_DIVISION,
    ("chop", "planted"): _CHOP_DIVISION,
    ("chop", "malformed"): _CHOP_DIVISION,
    ("er", "flat"): _ZERO_MARKER.format(what="a window with no movement", talib="1.0"),
    ("er", "flat_block"): _ZERO_MARKER.format(what="a window with no movement", talib="1.0"),
    ("er", "planted"): _ZERO_MARKER.format(what="a window with no movement", talib="1.0"),
    ("massi", "flat"): _ZERO_MARKER.format(what="a bar with no range (its EMA ratio is 0/0)", talib="25, a ratio of 1 on each of 25 bars"),
    ("massi", "no_range"): _ZERO_MARKER.format(what="a bar with no range (its EMA ratio is 0/0)", talib="25, a ratio of 1 on each of 25 bars"),
    ("kurtosis", "flat"): _ZERO_MARKER.format(what="a window with no variance", talib="NaN"),
    ("kurtosis", "flat_block"): _ZERO_MARKER.format(what="a window with no variance", talib="NaN"),
    ("cmf", "zero_volume"): (
        "KEPT: a window with no volume has no money flow to weigh, so native reads NaN (test_degenerate_input.py); TA-Lib reads 0.0"
    ),
    ("marketfi", "zero_volume"): (
        "KEPT: a range on no volume is a real division by zero and native reads inf (test_degenerate_input.py); TA-Lib reads 0.0"
    ),
    ("emv", "zero_volume"): (
        "KEPT: a range on no volume is a real division by zero and native reads inf (test_degenerate_input.py); TA-Lib reads 0.0"
    ),
    ("eom", "zero_volume"): (
        "KEPT: a bar on no volume whose midpoint did not move is 0/0 and reads NaN (test_degenerate_input.py), "
        "and the NaN stays in the 14-bar average; TA-Lib reads a value"
    ),
    ("mcgd", "spike"): (
        "TALIB: MCGD divides by (close / previous MCGD) ** 4, which a flash crash to 0.1x drives towards zero: TA-Lib "
        "swings to -1.9e5 and then 1e64, native follows the price"
    ),
}
if _TALIB_08:
    KNOWN_DIFFERENCES[("willr", "malformed")] = (
        "TALIB: TA-Lib 0.8 clamps WILLR to -100..0, so it reads 0.0 where malformed bars put the close above "
        "the window's highest high (12 bars); native and TA-Lib 0.7 compute the formula, which falls "
        "outside -100..0 there and shows the bad bars. Kept unclamped by maintainer decision: TA-Lib 0.8 "
        "does not clamp STOCH or STOCHF, so clamping willr would split it from stochf, and clamping all four "
        "would break STOCH/STOCHF parity"
    )
    KNOWN_DIFFERENCES[("mfi", "spy_full")] = (
        "TALIB: TA-Lib 0.8 (not 0.7) is off by up to 6.7 on 28 bars between 333 and 458 of SPY_D, starting where "
        "two typical prices are equal in exact arithmetic but 1.4e-14 apart in floating point; native counts that "
        "as an up move and matches a plain loop over the definition to 3e-14"
    )
else:
    KNOWN_DIFFERENCES[("stochrsi", "flat_block")] = _STOCHRSI_NOISE
    KNOWN_DIFFERENCES[("stochrsi", "planted")] = _STOCHRSI_NOISE
    KNOWN_DIFFERENCES[("cci", "flat_block")] = _TALIB_CCI_RESIDUE
    KNOWN_DIFFERENCES[("cci", "planted")] = _TALIB_CCI_RESIDUE
    KNOWN_DIFFERENCES[("correl", "x1e-6")] = (
        "TALIB: TA-Lib 0.7's CORREL reads 0.0 at small prices (0.958 natively, unchanged from x1): the product "
        "of the two variances, about 1e-20 there, falls under its absolute zero test; correlation "
        "does not depend on the unit"
    )


_CASES = {**ORACLE, **VARIANTS}


@pytest.mark.parametrize(("name", "frame"), params([(name, frame) for name in sorted(_CASES) for frame in FRAMES], KNOWN_DIFFERENCES))
def test_native_matches_talib(name: str, frame: str, frames) -> None:
    case = _CASES[name]
    bars = frames[frame]
    native, oracle = case.native(bars), case.oracle(bars)
    assert len(native) == len(oracle), f"{len(native)} native columns vs {len(oracle)} TA-Lib outputs"
    problems = [f"column {i}: {d}" for i, (n, o) in enumerate(zip(native, oracle)) if (d := _disagreement(n, o, case.settle, TOLERANCE.get(name, (1.0, ""))[0]))]
    assert not problems, f"{name} on {frame}: " + "; ".join(problems)


# _disagreement() starts at the first bar both sides fill, so a value one bar
# early or late passes it. Every case also publishes its first value on TA-Lib's
# bar, except these, where TA-Lib's lookback is a convention of its own.
_TALIB_ALIGNS_TO_SIGNAL = "TALIB: TA-Lib starts every MACD output where the signal line starts, 8 bars after native starts the MACD line"
_TALIB_ALIGNS_K_TO_D = "TALIB: TA-Lib starts %K where %D starts, two bars later; native publishes %K as soon as it has a value"
WARMUP_DIFFERENCES = {
    "macd": _TALIB_ALIGNS_TO_SIGNAL,
    "macdext": _TALIB_ALIGNS_TO_SIGNAL,
    "macdfix": _TALIB_ALIGNS_TO_SIGNAL,
    "stoch": _TALIB_ALIGNS_K_TO_D,
    "stochf": _TALIB_ALIGNS_K_TO_D,
    "stochrsi": _TALIB_ALIGNS_K_TO_D,
    "mavp": "TALIB: MAVP's lookback is its maximum period (30) whatever periods the bars ask for; native starts at the window a bar asks for",
    "pvt": "TALIB: PVT reads 0 on bar 0; native has no previous close there and reads NaN",
    "wad": "TALIB: WAD reads 0 on bar 0; native has no previous close there and reads NaN",
    "kdj": _TALIB_ALIGNS_K_TO_D,
    "kst": "TALIB: TA-Lib starts KST where its signal line starts, 8 bars after native starts the KST line",
    "ichimoku": "TALIB: TA-Lib starts every ICHIMOKU line where span B starts (bar 51); native publishes tenkan and kijun as soon as each has a window",
    "supertrend": "TALIB: SUPERTREND reads trend 0 during its warm-up; native reads NaN until the first band exists",
    "mcgd": "TALIB: MCGD withholds its first 13 bars; native seeds with the first close and publishes from bar 0, with the same values after",
    "fisher": "OPEN: FISHER starts on the first full window; native's loop starts one bar later (range(length, m)) although "
    "the window's position is known there, and the two agree from bar 74 on",
}
# Integer output with no NaN at all: both sides read 0 through the lookback, so a
# NaN mask cannot show its start. test_native_matches_talib compares those zeros by value.
_WARMUP_BY_VALUE = {"ht_trendmode"}


def _warmup_mismatch(native, oracle, real: np.ndarray) -> str | None:
    """Where the NaN masks differ on the bars with input (*real*), or None. Bars with missing
    input are left out: native reads NaN there, TA-Lib returns 0 for ht_trendmode's integers."""
    native_nan, oracle_nan = np.isnan(np.asarray(native, dtype=float)), np.isnan(np.asarray(oracle, dtype=float))
    differ = np.flatnonzero((native_nan != oracle_nan) & real)
    if not differ.size:
        return None
    first = lambda nan: int(np.flatnonzero(~nan & real)[0]) if (~nan & real).any() else None
    return f"first value at bar {first(native_nan)}, TA-Lib at {first(oracle_nan)} ({differ.size} bars differ)"


@pytest.mark.parametrize("frame", ["spy", "nan_prefix", "spy_full"])
@pytest.mark.parametrize(
    "name", params([(name,) for name in sorted(_CASES) if name not in _WARMUP_BY_VALUE], {(name,): reason for name, reason in WARMUP_DIFFERENCES.items()})
)
def test_warmup_matches_talib(name: str, frame: str, frames) -> None:
    case, bars = _CASES[name], frames[frame]
    real = np.isfinite(bars.close.to_numpy())
    native, oracle = case.native(bars), case.oracle(bars)
    assert len(native) == len(oracle), f"{len(native)} native columns vs {len(oracle)} TA-Lib outputs"
    problems = [f"column {i}: {d}" for i, (n, o) in enumerate(zip(native, oracle)) if (d := _warmup_mismatch(n, o, real))]
    assert not problems, f"{name} on {frame}: " + "; ".join(problems)


def test_warmup_mismatch_rejects_a_shifted_start() -> None:
    oracle = np.r_[np.full(5, np.nan), np.arange(10.0)]
    real = np.ones(15, dtype=bool)
    assert _warmup_mismatch(oracle, oracle, real) is None
    assert _warmup_mismatch(np.r_[np.nan * np.ones(4), np.arange(11.0)], oracle, real) == "first value at bar 4, TA-Lib at 5 (1 bars differ)"
    assert _warmup_mismatch(np.r_[np.nan * np.ones(6), np.arange(9.0)], oracle, real) == "first value at bar 6, TA-Lib at 5 (1 bars differ)"


@pytest.mark.parametrize("name", sorted(TOLERANCE))
def test_every_tolerance_is_needed(name: str, frames) -> None:
    """A TOLERANCE factor is at most 25x what the case needs (10x margin, rounded up to 1-2-5): at a 25th of it, some frame fails."""
    case, factor = _CASES[name], TOLERANCE[name][0]
    tight = [
        frame
        for frame in FRAMES
        if (name, frame) not in KNOWN_DIFFERENCES
        and any(_disagreement(n, o, case.settle, factor / 25) for n, o in zip(case.native(frames[frame]), case.oracle(frames[frame])))
    ]
    assert tight, f"{name} passes every frame at {factor / 25:g}x the default: lower its TOLERANCE factor"


def test_disagreement_rejects_a_wrong_result() -> None:
    """The comparison must fail on wrong results: a helper that accepts everything turns every case into a no-op."""
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    values = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 50))
    values[:5] = np.nan
    expected = pd.Series(values, index=index)
    assert _disagreement(expected * (1 + 1e-12), expected) is None
    assert _disagreement(expected * (1 + 1e-9), expected) is not None
    assert _disagreement(expected * (1 + 1e-6), expected) is not None
    assert _disagreement(expected + 1e-6, expected) is not None
    holed = expected.copy()
    holed.iloc[30] = np.nan
    assert _disagreement(holed, expected) is not None


# ---------------------------------------------------------------------------
# Candle patterns.
# ---------------------------------------------------------------------------

# Native patterns without a CDL* function of the same name.
_NO_CDL_TWIN = {"inside", "z", "ha"}


_CANDLE_ROWS = [(pattern, frame) for pattern in sorted(p for p in ta.ALL_PATTERNS if p not in _NO_CDL_TWIN) for frame in FRAMES]


@pytest.mark.parametrize(
    ("pattern", "frame"),
    params(_CANDLE_ROWS, KNOWN_DIFFERENCES, key=lambda row: (f"cdl_{row[0]}", row[1]), id=lambda row: f"cdl_{row[0]}-{row[1]}"),
)
def test_candle_pattern_matches_talib(pattern: str, frame: str, frames) -> None:
    b = frames[frame]
    native = ta.cdl_pattern(b.open, b.high, b.low, b.close, name=pattern).iloc[:, 0]
    oracle = getattr(T, f"CDL{pattern.upper()}")(b.open, b.high, b.low, b.close)
    problem = _disagreement(native, oracle)
    assert problem is None, f"cdl_{pattern} on {frame}: {problem}"


@pytest.mark.parametrize(("length", "factor"), [(10, 0), (3, 10), (25, 10), (10, 25), (7, 2.5)])
@pytest.mark.parametrize("frame", FRAMES)
def test_cdl_doji_matches_talib_with_its_settings(length: int, factor: float, frame: str, frames) -> None:
    """``cdl_doji(length, factor)`` is CDLDOJI with BodyDoji set to (HighLow, length, factor / 100).

    TA-Lib keeps that setting process-wide, so it is restored afterwards.
    """
    lib = getattr(T, "_ta_lib", None)
    if lib is None or not all(hasattr(lib, name) for name in ("_ta_set_candle_settings", "_ta_restore_candle_default_settings", "CandleSettingType")):
        pytest.skip("this TA-Lib build does not expose the candle-settings API")
    b = frames[frame]
    high_low = 1  # TA_RangeType_HighLow
    try:
        lib._ta_set_candle_settings(lib.CandleSettingType.BodyDoji, high_low, length, factor / 100)
        oracle = T.CDLDOJI(b.open, b.high, b.low, b.close)
    finally:
        lib._ta_restore_candle_default_settings(lib.CandleSettingType.AllCandleSettings)
    problem = _disagreement(ta.cdl_doji(b.open, b.high, b.low, b.close, length=length, factor=factor), oracle)
    assert problem is None, f"cdl_doji({length}, {factor}) on {frame}: {problem}"


# ---------------------------------------------------------------------------
# Comparisons the table cannot express.
# ---------------------------------------------------------------------------


def _random_walk_hl(seed: int, n: int = 5000) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    # From 200, not 100: seed 0's walk fell to -11.85, and sarext reads NaN, with a
    # warning, where high or low is at or below zero (#345). A shift changes no SAR step.
    c = 200 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    return pd.Series(np.maximum(o, c) + rng.random(n)), pd.Series(np.minimum(o, c) - rng.random(n))


@pytest.mark.parametrize("kwargs", [{}, *SAREXT_OPTIONS.values()], ids=["defaults", *SAREXT_OPTIONS])
@pytest.mark.parametrize("seed", [0, 3])
def test_sarext_matches_talib_on_a_long_walk(kwargs: dict, seed: int) -> None:
    """5000 bars of a random walk reverse far more often than the 400-bar frames."""
    high, low = _random_walk_hl(seed)
    expected = T.SAREXT(high.to_numpy(float), low.to_numpy(float), **kwargs)
    np.testing.assert_allclose(ta.sarext(high, low, **kwargs).to_numpy(float), expected, rtol=0, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("frame", FRAMES)
def test_window_index_matches_talib_up_to_ties(frame: str, frames) -> None:
    """TA-Lib returns the absolute index: MAXINDEX = maxindex + i - length + 1, except where a tie picks another bar.

    TA-Lib returns an integer even for a window with missing bars, so only the
    windows of 30 real bars are compared, and native must have a value on each."""
    close = frames[frame].close
    values, bars = close.to_numpy(float), np.arange(close.size)
    start = bars[29:] - 29
    full = (close.rolling(30).count() == 30).to_numpy()[29:]
    assert full.any(), f"no full window on {frame}"
    for func, oracle in ((ta.maxindex, T.MAXINDEX), (ta.minindex, T.MININDEX)):
        got = func(close, length=30).to_numpy(float)[29:]
        expected = np.asarray(oracle(close, 30), float)[29:] - start
        assert np.isfinite(got[full]).all(), f"{func.__name__} reads NaN in a window of 30 real bars on {frame}"
        picked = values[start[full] + got[full].astype(int)]
        np.testing.assert_array_equal(picked, values[start[full] + expected[full].astype(int)], err_msg=f"{func.__name__} on {frame}")


# ---------------------------------------------------------------------------
# Completeness.
# ---------------------------------------------------------------------------


def _matrix_talib_indicators() -> set[str]:
    text = (_ROOT / "docs" / "indicator_support_matrix.rst").read_text(encoding="utf-8")
    rows = re.findall(r"   \* - (.+)\n     - (.+)\n     - (.+)\n     - (.+)\n", text)
    return {name.strip() for name, _category, _native, talib_ in rows[1:] if talib_.strip() == "yes"}


def test_the_cmo_case_stays_native() -> None:
    # The cmo case compares native Wilder smoothing with TA-Lib only while scalar=50
    # keeps talib=True off TA-Lib (rule 6). If TA-Lib ever runs there, this fails
    # before the case turns into TA-Lib compared with itself.
    close = pd.Series(np.linspace(100.0, 120.0, 60) + np.sin(np.arange(60.0)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ta.Imports, "talib", False)
        native = ta.cmo(close, 14, scalar=50, talib=True)
    pd.testing.assert_series_equal(ta.cmo(close, 14, scalar=50, talib=True), native)


def test_every_talib_indicator_has_an_oracle() -> None:
    """An indicator with a TA-Lib counterpart is compared, or says why not."""
    listed = _matrix_talib_indicators()
    missing = sorted(listed - set(ORACLE) - set(NO_ORACLE))
    assert not missing, f"listed with TA-Lib in docs/indicator_support_matrix.rst but not in ORACLE or NO_ORACLE: {missing}"
    unlisted = sorted(set(ORACLE) - listed)
    assert not unlisted, f"compared with TA-Lib here but listed without it in docs/indicator_support_matrix.rst: {unlisted}"
    stale = sorted(set(NO_ORACLE) - listed - {"ma"})
    assert not stale, f"NO_ORACLE gives a reason for an indicator listed without TA-Lib in docs/indicator_support_matrix.rst: {stale}"


def test_vidya_converges_to_talib(frames) -> None:
    """VIDYA's seed sits a bar later on TA-Lib; over SPY_D the two still meet (NO_ORACLE)."""
    b = frames["spy_full"]
    assert _disagreement(ta.vidya(b.close, 14), talib.VIDYA(b.close, 14, 14), settle=800) is None
