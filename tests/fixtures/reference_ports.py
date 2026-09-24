"""Independent reference ports of indicators that TA-Lib and tulipy do not cover.

Why this module exists
----------------------
Before these ports, the indicators below were "regression-only": their
expected values were snapshots of this package's own output, which guard
against change but cannot show that the numbers are right. Each function here
is a plain-loop transcription of the definition the indicator cites, written
without any pandas_ta_classic code, so ``tests/test_reference_ports.py`` can
compare the package against it.

Pine conventions are followed where a TradingView script is the source:
``ta.rma`` (Wilder) seeds with an SMA and ``ta.rsi`` uses it. EMAs seed with an
SMA of the first ``n`` values, as this package does; the comparison skips the
warm-up, where seed conventions differ.

These are deliberately naive O(n * window) loops, optimised for being
obviously correct.
"""

from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    start = int(np.flatnonzero(np.isfinite(x))[0])
    if start + n > len(x):
        return out
    out[start + n - 1] = np.mean(x[start : start + n])
    for i in range(start + n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    start = int(np.flatnonzero(np.isfinite(x))[0])
    out[start + n - 1] = np.mean(x[start : start + n])
    for i in range(start + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rolling(x: np.ndarray, n: int, f) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1 : i + 1]
        if np.isfinite(w).all():
            out[i] = f(w)
    return out


def roc(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    out[n:] = 100 * (c[n:] - c[:-n]) / c[:-n]
    return out


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[np.nan, np.diff(c)]
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    return 100 - 100 / (1 + rma(up, n) / rma(dn, n))


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.r_[np.nan, np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])]


def kst(c, r=(10, 15, 20, 30), s=(10, 10, 10, 15), sig=9):
    """Know Sure Thing (Pring; TradingView built-in): weighted sum of SMA-smoothed percentage ROCs."""
    k = sum(w * rolling(roc(c, ri), si, np.mean) for w, ri, si in zip((1, 2, 3, 4), r, s))
    return k, rolling(k, sig, np.mean)


def smi(c, fast=5, slow=20, signal=5):
    """SMI Ergodic (TradingView built-in): erg = tsi(close, fast, slow) on a -1..1 scale."""
    m = np.r_[np.nan, np.diff(c)]
    erg = ema(ema(m, slow), fast) / ema(ema(np.abs(m), slow), fast)
    sig = ema(erg, signal)
    return erg, sig, erg - sig


def stc(c, tclen=10, fast=12, slow=26, factor=0.5):
    """Schaff Trend Cycle (ProRealCode "schaff-trend-cycle2"): the stochastic is held while the range is 0."""
    macd = ema(c, fast) - ema(c, slow)
    n = len(c)
    f1, pf, f2, out = (np.full(n, np.nan) for _ in range(4))
    for i in range(tclen - 1, n):
        w = macd[i - tclen + 1 : i + 1]
        if np.isfinite(w).all():
            lo, hi = w.min(), w.max()
            f1[i] = 100 * (macd[i] - lo) / (hi - lo) if hi > lo else (f1[i - 1] if np.isfinite(f1[i - 1]) else 0.0)
            pf[i] = f1[i] if not np.isfinite(pf[i - 1]) else pf[i - 1] + factor * (f1[i] - pf[i - 1])
        w = pf[i - tclen + 1 : i + 1]
        if np.isfinite(w).all():
            lo, hi = w.min(), w.max()
            f2[i] = 100 * (pf[i] - lo) / (hi - lo) if hi > lo else (f2[i - 1] if np.isfinite(f2[i - 1]) else 0.0)
            out[i] = f2[i] if not np.isfinite(out[i - 1]) else out[i - 1] + factor * (f2[i] - out[i - 1])
    return out


def qqe(c, rsi_period=14, sf=5, q=4.236):
    """QQE (TradingView "QQE MT4" by Glaz): trailing line on the smoothed RSI."""
    wilders = rsi_period * 2 - 1
    rsima = ema(rsi(c, rsi_period), sf)
    dar = ema(ema(np.abs(np.r_[np.nan, np.diff(rsima)]), wilders), wilders) * q
    n = len(c)
    longband, shortband, trend = np.zeros(n), np.zeros(n), np.zeros(n)
    line = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isfinite(dar[i]) and np.isfinite(rsima[i - 1])):
            trend[i] = 1
            continue
        r, r1 = rsima[i], rsima[i - 1]
        nl, ns = r - dar[i], r + dar[i]
        longband[i] = max(longband[i - 1], nl) if (r1 > longband[i - 1] and r > longband[i - 1]) else nl
        shortband[i] = min(shortband[i - 1], ns) if (r1 < shortband[i - 1] and r < shortband[i - 1]) else ns
        cross_up = r > shortband[i - 1] and r1 <= (shortband[i - 2] if i > 1 else shortband[i - 1])
        cross_dn = longband[i - 1] > r and (longband[i - 2] if i > 1 else longband[i - 1]) <= r1
        trend[i] = 1 if cross_up else (-1 if cross_dn else (trend[i - 1] or 1))
        line[i] = longband[i] if trend[i] == 1 else shortband[i]
    return line, rsima


def squeeze_pro_flags(h, l, c):
    """Squeeze Pro (Carter): Bollinger(20, 2) inside Keltner(20, SMA of TR) at 2, 1.5 and 1 widths."""
    basis = rolling(c, 20, np.mean)
    dev = rolling(c, 20, lambda w: w.std(ddof=0))
    kr = rolling(true_range(h, l, c), 20, np.mean)
    bl, bu = basis - 2 * dev, basis + 2 * dev
    flags = {k: (bl > basis - w * kr) & (bu < basis + w * kr) for k, w in (("wide", 2.0), ("normal", 1.5), ("narrow", 1.0))}
    flags["off"] = (bl < basis - 2.0 * kr) & (bu > basis + 2.0 * kr)
    return flags


def ttm_trend(h, l, c, length=6):
    """TTM Trend (ProRealCode "ttm-trend-price"): close against the average HL2 of the previous 6 bars."""
    hl2 = (h + l) / 2
    out = np.full(len(c), np.nan)
    for i in range(length, len(c)):
        out[i] = 1 if c[i] > hl2[i - length : i].mean() else -1
    return out


def cksp(h, l, c, p=10, x=1.0, q=9):
    """Chande Kroll Stop (TradingView built-in): stops from highest high / lowest low and a Wilder ATR."""
    a = rma(true_range(h, l, c), p)
    first_high = rolling(h, p, np.max) - x * a
    first_low = rolling(l, p, np.min) + x * a
    return rolling(first_high, q, np.max), rolling(first_low, q, np.min)


def ichimoku_lines(h, l, c, tenkan=9, kijun=26, senkou=52):
    """Ichimoku (Hosoda): midpoints of the high/low ranges; spans shifted forward and Chikou back by `kijun` bars."""
    mid = lambda n: (rolling(h, n, np.max) + rolling(l, n, np.min)) / 2
    t, k, sb = mid(tenkan), mid(kijun), mid(senkou)
    shift = lambda x, n: np.r_[np.full(n, np.nan), x[:-n]] if n > 0 else np.r_[x[-n:], np.full(-n, np.nan)]
    return {"ITS": t, "IKS": k, "ISA": shift((t + k) / 2, kijun), "ISB": shift(sb, kijun), "ICS": shift(c, -kijun)}


def squeeze_flags(h, l, c):
    """TTM Squeeze (Carter): Bollinger(20, 2) inside Keltner(20, 1.5 x SMA of TR)."""
    basis = rolling(c, 20, np.mean)
    dev = rolling(c, 20, lambda w: w.std(ddof=0))
    kr = rolling(true_range(h, l, c), 20, np.mean)
    bl, bu = basis - 2 * dev, basis + 2 * dev
    return (bl > basis - 1.5 * kr) & (bu < basis + 1.5 * kr), (bl < basis - 1.5 * kr) & (bu > basis + 1.5 * kr)


def supertrend_direction(h, l, c, n=7, f=3.0):
    """Supertrend direction (TradingView ta.supertrend), +1 for an uptrend; ATR as TA-Lib (Wilder, seeded after bar 0)."""
    tr = true_range(h, l, c)
    atr = rma(tr, n)
    src = (h + l) / 2
    ub, lb = src + f * atr, src - f * atr
    st, d = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for i in range(len(c)):
        if i > 0 and np.isfinite(lb[i - 1]):
            lb[i] = lb[i] if (lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
            ub[i] = ub[i] if (ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        if i == 0 or not np.isfinite(atr[i - 1]):
            d[i] = 1
        elif st[i - 1] == ub[i - 1]:
            d[i] = 1 if c[i] > ub[i] else -1
        else:
            d[i] = -1 if c[i] < lb[i] else 1
        st[i] = lb[i] if d[i] == 1 else ub[i]
    return d


def cpr_daily(h, l, c):
    """Classic daily CPR from the previous bar: pivot, BC, TC and the first two R/S levels."""
    ph, pl, pc = (np.r_[np.nan, x[:-1]] for x in (h, l, c))
    p = (ph + pl + pc) / 3
    bc = (ph + pl) / 2
    return {"PIVOT": p, "BC": bc, "TC": 2 * p - bc, "R1": 2 * p - pl, "S1": 2 * p - ph, "R2": p + (ph - pl), "S2": p - (ph - pl)}


def vwap(high, low, close, volume, keys):
    """VWAP: cumulative typical-price x volume over cumulative volume, reset whenever the anchor key changes."""
    tp = (high + low + close) / 3
    out = np.full(len(close), np.nan)
    num = den = 0.0
    prev = None
    for i, key in enumerate(keys):
        if key != prev:
            num = den = 0.0
            prev = key
        num += tp[i] * volume[i]
        den += volume[i]
        out[i] = num / den if den else np.nan
    return out


def vidya(c, length=14):
    """VIDYA (Chande) with CMO over `length` bars, seeded with the SMA of the first `length` closes."""
    d = np.r_[np.nan, np.diff(c)]
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    su, sd = rolling(up, length, np.sum), rolling(dn, length, np.sum)
    k = np.abs((su - sd) / (su + sd))
    a = 2 / (length + 1)
    out = np.full(len(c), np.nan)
    out[length - 1] = np.mean(c[:length])
    for i in range(length, len(c)):
        out[i] = a * k[i] * c[i] + (1 - a * k[i]) * out[i - 1]
    return out
