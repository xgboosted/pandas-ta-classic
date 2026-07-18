"""Generate the frozen tulipy oracle golden file.

tulipy (last release 2020) is unmaintained and ships no wheels for CPython
>=3.12, so it cannot be installed in the full CI matrix.  Its output is fully
deterministic, so we snapshot it once here and compare pandas-ta-classic native
output against the frozen arrays in ``test_oracle_tulipy.py`` on every Python
version — no tulipy install required at test time.

Run offline on a Python where tulipy installs (CPython <3.12):

    python tests/fixtures/generate_tulipy_oracle.py

This overwrites ``tests/fixtures/tulipy_oracle.json``.  Each value is a
front-trimmed array exactly as tulipy returns it; ``NaN`` is serialized as
``null``.  The test aligns each array to the tail of the date index, matching
the previous live-oracle behaviour.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tulipy as tp
except ImportError as exc:  # pragma: no cover - generator guard
    raise SystemExit(
        "tulipy is required to regenerate the oracle golden file and is only "
        "installable on CPython <3.12. Run this script there."
    ) from exc

_HERE = Path(__file__).parent
_DATA_PATH = _HERE.parent.parent / "examples" / "data" / "SPY_D.csv"
_OUT_PATH = _HERE / "tulipy_oracle.json"

# The test only ever compares the last <=2000 bars.  Store a slightly larger
# tail so intersection/index-alignment is unaffected while keeping the golden
# file small.  Each array is tail-anchored, so trimming the head is transparent.
_TAIL = 2100


def _to_list(arr):
    """Serialize the tail of a tulipy array to a JSON-safe list (NaN -> None)."""
    a = np.asarray(arr, dtype=float)[-_TAIL:]
    return [None if np.isnan(x) else float(x) for x in a]


def main():
    df = pd.read_csv(_DATA_PATH, index_col="date", parse_dates=True)
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    df.columns = df.columns.str.lower()
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    macd_line, _, _ = tp.macd(c, short_period=12, long_period=26, signal_period=9)
    stoch_k, stoch_d = tp.stoch(h, low, c, 14, 3, 3)
    aroon_down, aroon_up = tp.aroon(h, low, period=14)
    bb_lower, bb_mid, bb_upper = tp.bbands(c, period=20, stddev=2.0)
    fisher_val, fisher_sig = tp.fisher(h, low, period=9)
    msw_sine, msw_lead = tp.msw(c, period=5)

    oracle = {
        # Overlap / moving averages
        "sma": tp.sma(c, period=20),
        "ema": tp.ema(c, period=20),
        "wma": tp.wma(c, period=20),
        "hma": tp.hma(c, period=9),
        "zlema": tp.zlema(c, period=20),
        "wilders": tp.wilders(c, period=14),
        "dema": tp.dema(c, period=20),
        "tema": tp.tema(c, period=20),
        "kama": tp.kama(c, period=10),
        "vwma": tp.vwma(c, v, period=20),
        "wcprice": tp.wcprice(h, low, c),
        "qstick": tp.qstick(o, c, period=10),
        "trima": tp.trima(c, period=20),
        # Oscillators / momentum
        "rsi": tp.rsi(c, period=14),
        "mom": tp.mom(c, period=10),
        "roc": tp.roc(c, period=10),
        "rocr": tp.rocr(c, period=10),
        "willr": tp.willr(h, low, c, period=14),
        "cci": tp.cci(h, low, c, period=14),
        "bop": tp.bop(o, h, low, c),
        "mfi": tp.mfi(h, low, c, v, period=14),
        "cvi": tp.cvi(h, low, period=10),
        "ao": tp.ao(h, low),
        "vhf": tp.vhf(c, period=28),
        "wad": tp.wad(h, low, c),
        "dpo": tp.dpo(c, period=20),
        "cmo": tp.cmo(c, period=14),
        "fosc": tp.fosc(c, period=14),
        "trix": tp.trix(c, period=18),
        "stochrsi": tp.stochrsi(c, period=14),
        "macd": macd_line,
        "obv": tp.obv(c, v),
        # Stochastic
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        # Trend / directional
        "aroon_down": aroon_down,
        "aroon_up": aroon_up,
        "aroonosc": tp.aroonosc(h, low, period=14),
        "dx": tp.dx(h, low, c, period=14),
        # Volatility
        "atr": tp.atr(h, low, c, period=14),
        "natr": tp.natr(h, low, c, period=14),
        "bbands_lower": bb_lower,
        "bbands_mid": bb_mid,
        "bbands_upper": bb_upper,
        # Statistics
        "stddev": tp.stddev(c, period=20),
        "var": tp.var(c, period=20),
        "stderr": tp.stderr(c, period=20),
        "linreg": tp.linreg(c, period=14),
        "tsf": tp.tsf(c, period=14),
        # Fisher / MSW
        "fisher_val": fisher_val,
        "fisher_sig": fisher_sig,
        "msw_sine": msw_sine,
        "msw_lead": msw_lead,
        # Misc
        "edecay": tp.edecay(c, period=5),
        "emv": tp.emv(h, low, v),
        "md": tp.md(c, period=5),
        "lag": tp.lag(c, period=3),
        "avgprice": tp.avgprice(o, h, low, c),
        "medprice": tp.medprice(h, low),
        "typprice": tp.typprice(h, low, c),
    }

    payload = {
        "_meta": {
            "source": "tulipy",
            "tulipy_version": getattr(tp, "__version__", "0.4.0"),
            "data_file": _DATA_PATH.name,
            "rows": int(len(c)),
            "tail": _TAIL,
            "note": "Frozen tulipy output (tail only); regenerate with generate_tulipy_oracle.py on CPython <3.12.",
        },
        "arrays": {k: _to_list(val) for k, val in oracle.items()},
    }

    _OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {_OUT_PATH} ({len(oracle)} arrays, {len(c)} source rows)")


if __name__ == "__main__":
    main()
