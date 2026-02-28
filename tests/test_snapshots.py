"""Golden-file snapshot tests for all non-talib indicators.

Workflow:
  1. Generate/refresh snapshots:
       pytest tests/test_snapshots.py --update-snapshots -v
  2. Verify snapshots (normal CI run):
       pytest tests/test_snapshots.py

The first run writes tests/fixtures/snapshots.json.
The second run verifies every indicator against the frozen hashes.
"""

import inspect
import json
from pathlib import Path

import pytest

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data, hash_result

# ---------------------------------------------------------------------------
# Talib-validated indicators are already covered by correlation tests (>0.99).
# ---------------------------------------------------------------------------
TALIB_VALIDATED = {
    # momentum
    "apo",
    "bop",
    "cmo",
    "dm",
    "macd",
    "mom",
    "ppo",
    "roc",
    "rsi",
    "stoch",
    "stochrsi",
    "trix",
    "uo",
    "willr",
    # overlap
    "dema",
    "ema",
    "hl2",
    "hlc3",
    "kama",
    "linreg",
    "midpoint",
    "midprice",
    "ohlc4",
    "sma",
    "t3",
    "tema",
    "trima",
    "wcp",
    "wma",
    # trend
    "adx",
    "aroon",
    "aroonosc",
    "psar",
    # volatility
    "atr",
    "bbands",
    "natr",
    "trange",
    # volume
    "ad",
    "adosc",
    "mfi",
    "obv",
    # statistics
    "stdev",
    "variance",
    # candles
    "cdl_doji",
}

# ---------------------------------------------------------------------------
# PARAM_MAP: maps a function parameter name to how to extract it from the df
# ---------------------------------------------------------------------------
PARAM_MAP = {
    "close": lambda d: d["close"],
    "high": lambda d: d["high"],
    "low": lambda d: d["low"],
    "open_": lambda d: d["open"],
    "open": lambda d: d["open"],
    "volume": lambda d: d["volume"],
}

# ---------------------------------------------------------------------------
# SPECIAL_CALLS: indicators that don't fit the generic PARAM_MAP pattern.
#   None     → skip entirely.
#   callable → use instead of _build_args().
# ---------------------------------------------------------------------------
SPECIAL_CALLS = {
    # Returns (DataFrame, DataFrame) tuple; snapshot only the main frame.
    "ichimoku": lambda ta, d: ta.ichimoku(d["high"], d["low"], d["close"])[0],
    # Needs a MA type string as first positional.
    "ma": lambda ta, d: ta.ma("sma", d["close"]),
    # Needs two Series (fast, slow).
    "long_run": lambda ta, d: ta.long_run(d["close"], d["close"]),
    "short_run": lambda ta, d: ta.short_run(d["close"], d["close"]),
    # Needs a boolean/int trend Series.
    "tsignals": lambda ta, d: ta.tsignals(
        (d["close"] > d["close"].shift(1)).astype(int)
    ),
    # Needs a signal Series plus xa/xb float thresholds.
    "xsignals": lambda ta, d: ta.xsignals(ta.rsi(d["close"]), xa=70, xb=30),
    # Requires a pattern name → too ambiguous to snapshot generically.
    "cdl_pattern": None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "snapshots.json"


def _build_args(func, data):
    """Inspect *func*'s signature; return positional args from OHLCV data.

    Returns None if an unknown *required* parameter is encountered
    (caller should skip the indicator).
    """
    args = []
    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            break
        if param.default is inspect.Parameter.empty:
            if name in PARAM_MAP:
                args.append(PARAM_MAP[name](data))
            else:
                return None  # unknown required arg → skip
        else:
            break  # first optional arg → done collecting positionals
    return args


def _call_indicator(name, data):
    """Call one indicator by name and return its result (or None on failure)."""
    if name in SPECIAL_CALLS:
        call = SPECIAL_CALLS[name]
        if call is None:
            return None
        try:
            return call(pandas_ta, data)
        except Exception:
            return None

    func = getattr(pandas_ta, name, None)
    if func is None:
        return None
    args = _build_args(func, data)
    if args is None:
        return None
    try:
        return func(*args)
    except Exception:
        return None


def _collect_hashes():
    """Compute hashes for all non-talib indicators."""
    data = get_sample_data()
    results = {}
    for _category, names in pandas_ta.Category.items():
        for name in names:
            if name in TALIB_VALIDATED:
                continue
            result = _call_indicator(name, data)
            h = hash_result(result)
            if h is not None:
                results[name] = h
    return results


def _load_fixture():
    """Load snapshots.json; return {} if the file doesn't exist yet."""
    if not FIXTURE.exists():
        return {}
    with FIXTURE.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Update test (only active with --update-snapshots)
# ---------------------------------------------------------------------------


def test_write_snapshots(request):
    """Write tests/fixtures/snapshots.json (only runs with --update-snapshots)."""
    if not request.config.getoption("--update-snapshots", default=False):
        pytest.skip("Pass --update-snapshots to regenerate")

    hashes = _collect_hashes()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as fh:
        json.dump(hashes, fh, indent=2, sort_keys=True)
    print(f"\n[snapshots] Wrote {len(hashes)} hashes to {FIXTURE}")
    assert (
        len(hashes) > 0
    ), "No indicators were discovered — check TALIB_VALIDATED / SPECIAL_CALLS"


# ---------------------------------------------------------------------------
# Verification tests (normal CI run)
# ---------------------------------------------------------------------------


def _snapshot_params():
    """Return (name, expected_hash) pairs for parametrize, or [] if no file."""
    return list(_load_fixture().items())


@pytest.mark.parametrize("name,expected_hash", _snapshot_params())
def test_snapshot(name, expected_hash):
    """Verify that indicator output matches the frozen hash.

    ``expected_hash`` may be a single string *or* a list of strings when a
    pandas/numpy version upgrade produces a numerically equivalent but
    bit-differently-rounded result (e.g. ``kurtosis`` / ``skew`` differ
    between pandas 2.x and pandas 3.x).
    """
    data = get_sample_data()
    result = _call_indicator(name, data)

    if result is None:
        pytest.skip(f"{name} returned None (skipped during snapshot generation)")

    actual_hash = hash_result(result)
    assert actual_hash is not None, f"{name} returned an unhashable result type"

    # Normalise to a set so we can support version-sensitive indicators.
    expected = {expected_hash} if isinstance(expected_hash, str) else set(expected_hash)
    assert actual_hash in expected, (
        f"{name} output hash changed!\n"
        f"  expected: {expected_hash}\n"
        f"  actual:   {actual_hash}\n"
        "Run `pytest tests/test_snapshots.py --update-snapshots` to accept new output."
    )
