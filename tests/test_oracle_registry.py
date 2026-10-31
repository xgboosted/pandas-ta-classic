"""Every indicator with a TA-Lib counterpart agrees with it on every known data problem.

``test_oracle_talib.py`` compares 60 hand-picked calls on ``SPY_D`` only, and
its ``_compare`` intersects both sides *after* ``dropna()``. Two classes of
defect were invisible to it by construction:

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
reason in ``NO_ORACLE``.

Every candle pattern is compared against its ``CDL*`` function the same way.
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

_ROOT = Path(__file__).parent.parent
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
    # largest is 200 (the MACD family, which reads only close); every problem
    # frame still reaches it after bar 200: the spike at 200, the planted run
    # at 260, and the halted closes at 170-180, which the recursion carries on.
    settle: int = 0
    # Relative tolerance, TALIB_RTOL unless TA-Lib's own arithmetic needs more.
    rtol: float = TALIB_RTOL


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
    # decayed below the tolerance by bar 200 (still 2.5e-7 relative at bar 154),
    # so both this suite and test_oracle_talib.py compare from there.
    # macdfix is MACD(12, 26) natively, not TA-Lib's fixed-alpha MACDFIX.
    "macd": Case(lambda b: _cols(ta.macd(b.close, 12, 26, 9), "MACD_", "MACDs_", "MACDh_"), lambda b: list(T.MACD(b.close, 12, 26, 9)), settle=200),
    "macdext": Case(
        lambda b: _cols(ta.macdext(b.close), "MACDEXT_", "MACDEXTs_", "MACDEXTh_"), lambda b: list(T.MACDEXT(b.close, 12, 1, 26, 1, 9, 1)), settle=200
    ),
    "macdfix": Case(
        lambda b: _cols(ta.macdfix(b.close, signal=9), "MACDFIX_", "MACDFIXs_", "MACDFIXh_"), lambda b: list(T.MACD(b.close, 12, 26, 9)), settle=200
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
    # TA-Lib keeps running sums of x and x**2 and cancels digits in
    # sum(x**2) - sum(x)**2 / n: 3e-8 relative off the exact deviation on the
    # planted frame, where native is exact; 1e-7 leaves room for other builds.
    "stdev": Case(lambda b: [ta.stdev(b.close, 20)], lambda b: [T.STDDEV(b.close, 20, 1)], rtol=1e-7),
    "variance": Case(lambda b: [ta.variance(b.close, 20)], lambda b: [T.VAR(b.close, 20, 1)], rtol=1e-7),  # as stdev
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
}


# ---------------------------------------------------------------------------
# Comparison.
# ---------------------------------------------------------------------------


def _first_common_valid(native: np.ndarray, oracle: np.ndarray) -> int | None:
    both = np.flatnonzero(np.isfinite(native) & np.isfinite(oracle))
    return None if both.size == 0 else int(both[0])


def _disagreement(native, oracle, settle: int = 0, rtol: float = TALIB_RTOL) -> str | None:
    """Describe the first disagreement, or None when the two agree.

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
        # assert_talib()'s tolerance, element by element, with the absolute term
        # scaled down to columns smaller than 1: at prices of 1e-4 a fixed 1e-9
        # would pass a 1e-6 relative error, and a variance of 1e-11 any error. A
        # column TA-Lib reads as all zero has no scale and keeps the plain 1e-9.
        scale = float(np.nanmax(np.abs(o[np.isfinite(o)]))) if np.isfinite(o).any() else 0.0
        scale = scale or 1.0
        off = ~((n == o) | (np.isnan(n) & np.isnan(o)) | (np.abs(n - o) <= TALIB_ATOL * min(1.0, scale) + rtol * np.abs(o)))
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
# TA-Lib changed between 0.7 and 0.8 (measured with 0.7.1 and 0.8.2, the
# version CI installs): 0.8 fixed the CCI residue, CORREL's zero test and
# STOCHRSI's flat-block noise (as native stochrsi does), and
# clamps WILLR to -100..0 where malformed bars put the close outside the range.
# The behaviour is the C library's, read from it: a source build can link a C
# library of another version than the Python wrapper (wrapper 0.8.2 ships C 0.8.1).
_TA_C_VERSION = talib.__ta_version__.decode().split()[0]
_TALIB_08 = tuple(int(part) for part in _TA_C_VERSION.split(".")[:2]) >= (0, 8)

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
}
if _TALIB_08:
    KNOWN_DIFFERENCES[("willr", "malformed")] = (
        "TALIB: TA-Lib 0.8 clamps WILLR to -100..0, so it reads 0.0 where malformed bars put the close above "
        "the window's highest high (12 bars); native and TA-Lib 0.7 compute the formula, which falls "
        "outside -100..0 there and shows the bad bars. Kept unclamped by maintainer decision: TA-Lib 0.8 "
        "does not clamp STOCH or STOCHF, so clamping willr would split it from stochf, and clamping all four "
        "would break STOCH/STOCHF parity"
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


@pytest.mark.parametrize(("name", "frame"), params([(name, frame) for name in sorted(ORACLE) for frame in FRAMES], KNOWN_DIFFERENCES))
def test_native_matches_talib(name: str, frame: str, frames) -> None:
    case = ORACLE[name]
    bars = frames[frame]
    native, oracle = case.native(bars), case.oracle(bars)
    assert len(native) == len(oracle), f"{len(native)} native columns vs {len(oracle)} TA-Lib outputs"
    problems = [f"column {i}: {d}" for i, (n, o) in enumerate(zip(native, oracle)) if (d := _disagreement(n, o, case.settle, case.rtol))]
    assert not problems, f"{name} on {frame}: " + "; ".join(problems)


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
