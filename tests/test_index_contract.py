"""Every indicator returns a result aligned to the index it was handed.

The contract is one line: given a Series of ``n`` rows, an indicator returns
``n`` rows carrying the *same* index. Warmup bars are reported as leading
``NaN``; they are never dropped, and the index is never replaced by a fresh
``RangeIndex``.

Why it matters: callers align indicator output against the source frame.
``backtesting.py`` rejects an indicator whose length differs from the data
(`kernc/backtesting.py#752 <https://github.com/kernc/backtesting.py/issues/752>`_
reported exactly that for ``stoch``), and a plain ``df["K"] = ta.stoch(...)``
silently fills the missing bars with ``NaN`` at the *wrong* end of the frame
when the index was reset.

Three indicators used to break it:

``stoch``
    Sliced the warmup off %K with ``.loc[first_valid_index():]`` and never
    reindexed, returning ``n - (k - 1)`` rows.
``t3``
    ``_ema_chain`` fed each EMA stage the trimmed output of the previous one
    and returned the trimmed stages, so ``t3`` lost ``~length`` rows per stage.
``td_seq``
    Built its Series from a bare numpy array, dropping the DatetimeIndex for a
    ``RangeIndex`` of the same length.

The sweep below is generic on purpose: a new indicator is covered the moment it
is registered in ``ta.Category``, and any new offender fails here rather than in
a user's backtest.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

_N_ROWS = 250

# Series-valued parameters the harness knows how to synthesise by name. Kept to
# price-like names only: `fast`, `slow` and `signal` are window *lengths* on most
# indicators (macd, ppo, ao), so they are supplied per indicator below instead.
_SERIES_PARAMS = frozenset(
    {
        "open_",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "benchmark",
        "series_a",
        "series_b",
        "source",
    }
)

# Indicators whose output is deliberately not indexed like the input.
_EXEMPT = {
    # Volume Profile bins the data into `width` price buckets; one row per
    # bucket, not per bar.
    "vp",
}


@pytest.fixture(scope="module")
def frame() -> dict[str, pd.Series]:
    """A deterministic OHLCV frame long enough for every default window."""
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, _N_ROWS))
    index = pd.date_range("2020-01-01", periods=_N_ROWS, freq="D")
    close = pd.Series(base, index=index)
    return {
        "open_": close - 0.5,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": pd.Series(rng.integers(1_000, 5_000, _N_ROWS).astype(float), index=index),
        "benchmark": close * 0.99,
        "series_a": close,
        "series_b": close.shift(1).bfill(),
        "fast": close.rolling(4).mean(),
        "slow": close.rolling(8).mean(),
        "trend": (close > close.rolling(8).mean()).astype(int),
        "signal": close,
        "source": close,
    }


def _indicator_names() -> list[str]:
    return sorted(
        {name for names in ta.Category.values() for name in names if callable(getattr(ta, name, None)) and not inspect.isclass(getattr(ta, name))}
    )


def _extra_kwargs(name: str, frame: dict[str, pd.Series]) -> dict:
    """Arguments with no sensible default for a blind sweep."""
    if name == "ma":
        return {"name": "sma"}
    if name == "ichimoku":
        # Opt out of the deprecated tuple return so the sweep stays warning-free.
        return {"as_dataframe": True}
    if name in ("long_run", "short_run"):
        # These take two indicator outputs, not price series.
        return {"fast": frame["fast"], "slow": frame["slow"]}
    if name == "mavp":
        # `periods` is mavp's second input -- one window length per bar -- not a
        # tuning knob with a sensible default. Supply a constant schedule so the
        # sweep does not depend on mavp keeping an optional `periods`.
        return {"periods": pd.Series(10.0, index=frame["close"].index)}
    if name == "tsignals":
        return {"trend": frame["trend"]}
    if name == "xsignals":
        return {"signal": frame["close"], "xa": 105, "xb": 95}
    return {}


def _results(name: str, frame: dict[str, pd.Series]):
    """Call *name* with every series argument it accepts; return each result frame."""
    func = getattr(ta, name)
    kwargs = {param: frame[param] for param in inspect.signature(func).parameters if param in _SERIES_PARAMS}
    result = func(**kwargs, **_extra_kwargs(name, frame))
    parts = result if isinstance(result, tuple) else (result,)
    return [part for part in parts if part is not None]


@pytest.mark.parametrize("name", [n for n in _indicator_names() if n not in _EXEMPT])
def test_result_keeps_the_input_index(name: str, frame: dict[str, pd.Series]) -> None:
    """Output is as long as the input and carries the input's index."""
    expected = frame["close"].index
    results = _results(name, frame)
    assert results, f"{name} returned nothing on {_N_ROWS} rows"

    for part in results:
        assert isinstance(part, (pd.Series, pd.DataFrame)), f"{name} returned {type(part).__name__}"
        assert len(part) == _N_ROWS, f"{name} returned {len(part)} rows for {_N_ROWS} of input -- warmup bars must be NaN, not dropped"
        assert part.index.equals(expected), f"{name} replaced the input index (got {type(part.index).__name__})"


def test_exempt_indicators_still_exist() -> None:
    """An exemption for a renamed indicator would silently stop covering it."""
    missing = _EXEMPT - set(_indicator_names())
    assert not missing, f"exempt indicator(s) no longer registered: {sorted(missing)}"


def test_stoch_reports_warmup_as_nan() -> None:
    """Regression for kernc/backtesting.py#752 -- %K used to start 13 bars in."""
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-01", periods=_N_ROWS, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, _N_ROWS)), index=index)

    result = ta.stoch(close + 1.0, close - 1.0, close, k=14, d=3, smooth_k=3)

    assert result.index.equals(index)
    assert result.iloc[:13].isna().all().all()
    assert result.iloc[-1].notna().all()


def test_t3_reports_warmup_as_nan() -> None:
    """``_ema_chain`` trimmed each stage, so ``t3`` came back ~18 rows short."""
    rng = np.random.default_rng(2)
    index = pd.date_range("2020-01-01", periods=_N_ROWS, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, _N_ROWS)), index=index)

    result = ta.t3(close, length=10)

    assert result.index.equals(index)
    assert result.iloc[0] != result.iloc[0]  # NaN warmup
    assert result.iloc[-1] == result.iloc[-1]


def test_ema_chain_stages_share_the_close_index() -> None:
    """Each stage is fed a trimmed series but must be handed back full-length."""
    from pandas_ta_classic.overlap.ema import _ema_chain

    rng = np.random.default_rng(3)
    index = pd.date_range("2020-01-01", periods=_N_ROWS, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, _N_ROWS)), index=index)

    stages = _ema_chain(close, length=10, depth=6)

    assert stages is not None
    assert len(stages) == 6
    for depth, stage in enumerate(stages, start=1):
        assert stage.index.equals(index), f"EMA stage {depth} lost the close index"
