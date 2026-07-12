"""Benchmark indicator runtime, and TA-Lib passthrough where available.

A reproducible harness for the performance figures cited in ``CHANGELOG.md`` and
a general-purpose way to find the current slowest indicators. It builds a
fixed-seed synthetic OHLCV frame, warms each indicator once (so numba JIT
compilation is excluded), then reports the median wall-clock time over several
runs — slowest first, so bottlenecks surface on their own without any hard-coded
list of "interesting" indicators.

Numba matters: numba-accelerated indicators need ``pip install numba`` for their
full speed; without it they fall back to plain Python loops. The banner prints
whether numba is active so a run is never ambiguous.

Usage::

    python tools/bench_indicators.py                 # every indicator, slowest first
    python tools/bench_indicators.py --top 20        # only the 20 slowest
    python tools/bench_indicators.py rsi macd jma    # specific names
    python tools/bench_indicators.py --rows 20000    # a larger frame
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
import warnings
from statistics import median

import numpy as np
import pandas as pd

import pandas_ta_classic as ta


def _numba_active() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def all_indicator_names() -> list[str]:
    """Every indicator name discovered across the category registry."""
    return sorted({n for v in ta.Category.values() for n in v})


def make_ohlcv(rows: int, seed: int = 42) -> pd.DataFrame:
    """Fixed-seed random-walk OHLCV frame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(rows))
    high = close + rng.random(rows) * 2
    low = close - rng.random(rows) * 2
    open_ = close + rng.standard_normal(rows)
    volume = rng.integers(1_000, 100_000, rows).astype(float)
    idx = pd.date_range("2000-01-01", periods=rows, freq="min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _time(fn, repeats: int) -> float:
    """Median wall-clock time in milliseconds over *repeats* runs."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples) * 1000.0


def _has_talib_param(name: str) -> bool:
    fn = getattr(ta, name, None)
    if fn is None:
        return False
    try:
        return "talib" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


def bench(names: list[str], rows: int, repeats: int) -> list[dict]:
    df = make_ohlcv(rows)
    talib_available = ta.Imports["talib"]
    results = []
    for name in names:
        acc = getattr(df.ta, name, None)
        if acc is None:
            results.append({"name": name, "status": "no-accessor"})
            continue
        try:
            acc()  # warmup (also triggers numba JIT compilation)
            acc()
            native = _time(acc, repeats)
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "status": f"err:{type(exc).__name__}"})
            continue
        talib_ms = None
        if talib_available and _has_talib_param(name):
            try:
                f = lambda: acc(talib=True)  # noqa: E731
                f()
                talib_ms = _time(f, repeats)
            except Exception:  # noqa: BLE001
                talib_ms = None
        results.append({"name": name, "native_ms": native, "talib_ms": talib_ms, "status": "ok"})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="indicator names to benchmark (default: all)")
    parser.add_argument("--rows", type=int, default=5000, help="number of bars (default: 5000)")
    parser.add_argument("--repeats", type=int, default=7, help="timed runs per indicator (default: 7)")
    parser.add_argument("--top", type=int, default=None, help="print only the N slowest (default: all)")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    names = args.names or all_indicator_names()

    numba = _numba_active()
    print(f"numba: {'ACTIVE - JIT speedups apply' if numba else 'ABSENT - @njit falls back to plain Python'}")
    print(f"rows: {args.rows}   repeats: {args.repeats}   indicators: {len(names)}\n")

    results = bench(names, args.rows, args.repeats)
    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: r["native_ms"], reverse=True)
    shown = ok[: args.top] if args.top else ok

    print(f"{'indicator':22} {'native_ms':>10} {'talib_ms':>10} {'nat/talib':>10}")
    for r in shown:
        tal = f"{r['talib_ms']:.3f}" if r["talib_ms"] else "-"
        sp = f"{r['native_ms'] / r['talib_ms']:.1f}x" if r["talib_ms"] else "-"
        print(f"{r['name']:22} {r['native_ms']:10.3f} {tal:>10} {sp:>10}")

    errs = [r for r in results if r["status"] != "ok"]
    for r in errs:
        print(f"{r['name']:22} {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
