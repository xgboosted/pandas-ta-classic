"""Exact rational-arithmetic reference implementations of the rolling statistics.

Why this module exists
----------------------
``expected_values.json`` needs a reference that is independent of *both* the
code under test and the installed dependency versions.  ``pandas`` rolling
satisfies neither requirement for the statistics group:

``roll_kurt`` / ``roll_skew`` / ``roll_var`` maintain running power sums and
add/remove one observation per step.  On SPY closes (mean ~330, tiny window
variance) that cancels catastrophically, and the error grows with the length
of the *series* rather than the window — 5222 rows give ~1.5e-8 absolute
error in ``rolling(20).kurt()`` on pandas 3.0 versus ~6.9e-10 on pandas 2.3.
The consequence is a golden value that changes on a pandas upgrade while the
package under test computed the same thing it always did.

The functions here take the exact decimal value of each CSV field
(``Fraction(str(x))``, not the float64 approximation), evaluate the textbook
formula in exact rational arithmetic, and only convert to float at the very
end.  The result is the mathematically correct value for the input data, is
bit-identical on every platform and dependency version, and is derived
independently of ``pandas_ta_classic`` — so it can still catch a development
error.

Irrational results (``sqrt``, ``log2``) cannot be represented exactly; those
are the single closing float operation applied to an exact rational argument,
which is well-conditioned here (no cancellation, all terms same sign).

These are deliberately naive O(n * length) loops.  They run once, in the
manual ``make fixtures`` step, and are optimised for being obviously correct.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Optional, Sequence

import pandas as pd

__all__ = [
    "rolling_beta",
    "rolling_entropy",
    "rolling_kurtosis",
    "rolling_mad",
    "rolling_median",
    "rolling_quantile",
    "rolling_skew",
    "rolling_ui",
    "rolling_zscore",
]


def _exact(series: pd.Series) -> list[Fraction]:
    """Return the exact decimal value of every element.

    ``Fraction(str(328.79))`` is 32879/100 — the number written in the CSV.
    ``Fraction(328.79)`` would instead be the float64 approximation, which is
    what we are trying to stay independent of.
    """
    return [Fraction(str(v)) for v in series.tolist()]


def _rolling(
    series: pd.Series,
    length: int,
    window_fn: Callable[[Sequence[Fraction]], float],
) -> pd.Series:
    """Apply *window_fn* to every full window of *length*, NaN during warmup."""
    exact = _exact(series)
    out: list[Optional[float]] = [None] * len(exact)
    for end in range(length, len(exact) + 1):
        out[end - 1] = window_fn(exact[end - length : end])
    return pd.Series(out, index=series.index, dtype="float64")


# ---------------------------------------------------------------------------
# Central moments
# ---------------------------------------------------------------------------


def _mean(window: Sequence[Fraction]) -> Fraction:
    """Exact arithmetic mean of the window."""
    return sum(window, Fraction(0)) / len(window)


def _central_moment(window: Sequence[Fraction], order: int) -> Fraction:
    """sum((x - mean) ** order) over the window, exactly."""
    mean = _mean(window)
    return sum(((x - mean) ** order for x in window), Fraction(0))


def rolling_zscore(close: pd.Series, length: int) -> pd.Series:
    """(close - rolling mean) / rolling sample stdev (ddof=1)."""

    def _z(w: Sequence[Fraction]) -> float:
        var = _central_moment(w, 2) / (len(w) - 1)
        return float(w[-1] - _mean(w)) / math.sqrt(float(var))

    return _rolling(close, length, _z)


def rolling_kurtosis(close: pd.Series, length: int) -> pd.Series:
    """Fisher (excess) kurtosis with the standard small-sample correction."""

    def _k(w: Sequence[Fraction]) -> float:
        n = Fraction(len(w))
        m2 = _central_moment(w, 2)
        m4 = _central_moment(w, 4)
        numer = n * (n + 1) * (n - 1) * m4
        denom = (n - 2) * (n - 3) * m2**2
        adj = Fraction(3) * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return float(numer / denom - adj)

    return _rolling(close, length, _k)


def rolling_skew(close: pd.Series, length: int) -> pd.Series:
    """Sample skewness with the standard small-sample correction."""

    def _sk(w: Sequence[Fraction]) -> float:
        n = Fraction(len(w))
        m2 = _central_moment(w, 2)
        m3 = _central_moment(w, 3)
        # m3 / (m2/(n-1))**1.5 involves a rational power; take the exact ratio
        # apart so only the final sqrt is inexact.
        variance = m2 / (n - 1)
        scale = float(variance) ** 1.5
        return float(n / ((n - 1) * (n - 2))) * float(m3) / scale

    return _rolling(close, length, _sk)


def rolling_mad(close: pd.Series, length: int) -> pd.Series:
    """Mean absolute deviation from the window mean."""

    def _mad(w: Sequence[Fraction]) -> float:
        mean = sum(w) / len(w)
        return float(sum(abs(x - mean) for x in w) / len(w))

    return _rolling(close, length, _mad)


# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------


def rolling_median(close: pd.Series, length: int) -> pd.Series:
    """Median of the window (mean of the two centre values when even)."""

    def _med(w: Sequence[Fraction]) -> float:
        s = sorted(w)
        mid = len(s) // 2
        if len(s) % 2:
            return float(s[mid])
        return float((s[mid - 1] + s[mid]) / 2)

    return _rolling(close, length, _med)


def rolling_quantile(close: pd.Series, length: int, q: float) -> pd.Series:
    """Quantile with linear interpolation — matches numpy's default method."""
    q_exact = Fraction(str(q))

    def _q(w: Sequence[Fraction]) -> float:
        s = sorted(w)
        pos = q_exact * (len(s) - 1)
        lo = int(pos)
        if lo == pos:
            return float(s[lo])
        frac = pos - lo
        return float(s[lo] + (s[lo + 1] - s[lo]) * frac)

    return _rolling(close, length, _q)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def rolling_entropy(close: pd.Series, length: int) -> pd.Series:
    """Base-2 Shannon entropy of the window normalised to a probability vector."""

    def _ent(w: Sequence[Fraction]) -> float:
        total = sum(abs(x) for x in w)
        acc = 0.0
        for x in w:
            p = abs(x) / total
            if p > 0:
                acc -= float(p) * math.log2(float(p))
        return acc

    return _rolling(close, length, _ent)


def rolling_beta(close: pd.Series, open_: pd.Series, length: int) -> pd.Series:
    """Cov(close returns, open returns) / Var(open returns) over *length* bars.

    Returns are simple period-over-period returns, computed exactly from the
    raw decimals rather than from an already-rounded float series.
    """
    c_exact = _exact(close)
    o_exact = _exact(open_)
    n_bars = len(c_exact)

    # ret[j] is the return of bar j + 1; bar 0 has no predecessor.
    c_ret = [c_exact[i] / c_exact[i - 1] - 1 for i in range(1, n_bars)]
    o_ret = [o_exact[i] / o_exact[i - 1] - 1 for i in range(1, n_bars)]

    out: list[Optional[float]] = [None] * n_bars
    # A window of `length` returns ending at bar `bar` spans ret indices
    # bar - length .. bar - 1, so the first defined output is at bar `length`.
    for bar in range(length, n_bars):
        cw = c_ret[bar - length : bar]
        ow = o_ret[bar - length : bar]
        c_mean = _mean(cw)
        o_mean = _mean(ow)
        cov = sum(((a - c_mean) * (b - o_mean) for a, b in zip(cw, ow)), Fraction(0)) / (length - 1)
        var = sum(((b - o_mean) ** 2 for b in ow), Fraction(0)) / (length - 1)
        out[bar] = float(cov / var)
    return pd.Series(out, index=close.index, dtype="float64")


def rolling_ui(close: pd.Series, length: int) -> pd.Series:
    """Ulcer Index: sqrt(mean(drawdown_pct ** 2)) over *length* bars.

    Drawdown is measured against a rolling maximum of the same length, so the
    first defined value needs ``2 * length - 1`` observations.
    """
    exact = _exact(close)
    n_bars = len(exact)

    # dd[j] is the drawdown of bar j + length - 1; earlier bars have no full
    # rolling maximum yet.
    dd_offset = length - 1
    dd = []
    for bar in range(dd_offset, n_bars):
        rmax = max(exact[bar - dd_offset : bar + 1])
        dd.append((exact[bar] - rmax) / rmax * 100)

    out: list[Optional[float]] = [None] * n_bars
    # Averaging `length` drawdowns ending at bar `bar` needs dd indices
    # bar - dd_offset - length + 1 .. bar - dd_offset, so the first defined
    # output is at bar 2 * length - 2.
    for bar in range(2 * length - 2, n_bars):
        hi = bar - dd_offset + 1
        squares = sum((d**2 for d in dd[hi - length : hi]), Fraction(0))
        out[bar] = math.sqrt(float(squares / length))
    return pd.Series(out, index=close.index, dtype="float64")
