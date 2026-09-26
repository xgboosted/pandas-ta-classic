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


def _build_frame() -> dict[str, pd.Series]:
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


@pytest.fixture(scope="module")
def frame() -> dict[str, pd.Series]:
    """The shared frame for the read-only sweeps."""
    return _build_frame()


def _indicator_names() -> list[str]:
    return sorted(
        {name for names in ta.Category.values() for name in names if callable(getattr(ta, name, None)) and not inspect.isclass(getattr(ta, name))}
    )


def _extra_kwargs(name: str, frame: dict[str, pd.Series]) -> dict:
    """Arguments with no sensible default for a blind sweep."""
    if name == "ma":
        return {"name": "sma"}
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


def _input_state(series: pd.Series) -> dict:
    """Everything about a caller's Series an indicator must leave alone."""
    index = series.index
    return {
        "index_id": id(index),
        "freq": getattr(index, "freq", None),
        "index_values": index.to_numpy(copy=True),
        "index_name": index.name,
        "name": series.name,
        "dtype": series.dtype,
        "values": series.to_numpy(copy=True),
    }


@pytest.mark.parametrize("name", _indicator_names())
def test_call_leaves_its_inputs_untouched(name: str) -> None:
    """An indicator reads its inputs; it never writes to them.

    ``nvi``, ``pvi`` and ``vp`` used to select a subset with a boolean mask
    (``signed_volume[signed_volume < 0]``) and multiply it back against a
    full-length Series. Realigning the subset rebuilds the index, and pandas
    clears the frequency on the *shared* index object, so ``df.ta.nvi()``
    unset ``df.index.freq`` on the caller's own frame -- with ``append=False``,
    from a call that returns its result.

    ``index.equals()`` ignores ``freq``, so the output sweep above cannot see
    this. Each case gets a frame of its own: one offender must not decide
    whether the next indicator in the sweep looks clean.
    """
    frame = _build_frame()
    func = getattr(ta, name)
    passed = {param: frame[param] for param in inspect.signature(func).parameters if param in _SERIES_PARAMS}
    passed.update({key: value for key, value in _extra_kwargs(name, frame).items() if isinstance(value, pd.Series)})
    before = {param: _input_state(series) for param, series in passed.items()}

    _results(name, frame)

    for param, series in passed.items():
        after = _input_state(series)
        expected = before[param]
        assert after["freq"] == expected["freq"], f"{name}() cleared {param}.index.freq on the caller's index ({expected['freq']} -> {after['freq']})"
        assert after["index_id"] == expected["index_id"], f"{name}() replaced the index object of its {param} input"
        np.testing.assert_array_equal(after["index_values"], expected["index_values"], err_msg=f"{name}() rewrote {param}'s index labels")
        assert after["index_name"] == expected["index_name"], f"{name}() renamed {param}'s index"
        assert after["name"] == expected["name"], f"{name}() renamed its {param} input"
        assert after["dtype"] == expected["dtype"], f"{name}() changed the dtype of its {param} input"
        np.testing.assert_array_equal(after["values"], expected["values"], err_msg=f"{name}() rewrote the values of its {param} input")


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


# Columns the accessor reads off the frame itself; every other Series argument
# has to be handed in.
_ACCESSOR_COLUMNS = frozenset({"open_", "open", "high", "low", "close", "volume"})


def _build_df() -> pd.DataFrame:
    """The same data as `_build_frame`, shaped the way a user holds it."""
    frame = _build_frame()
    return pd.DataFrame({column: frame[column] for column in ("open", "high", "low", "close", "volume")})


def _accessor_kwargs(name: str, method, frame: dict[str, pd.Series]) -> dict:
    """Series arguments the accessor cannot find among the frame's columns.

    Without them the accessor fills the missing argument with ``None`` and hands
    the caller's own frame back (`beta`, `correl`, `tsignals`, ...), which would
    make this sweep pass without running the indicator at all.
    """
    kwargs = {param: frame[param] for param in inspect.signature(method).parameters if param in _SERIES_PARAMS and param not in _ACCESSOR_COLUMNS}
    kwargs.update(_extra_kwargs(name, frame))
    return kwargs


def _frame_state(df: pd.DataFrame) -> dict:
    """Everything about a caller's DataFrame that a read must leave alone."""
    return {
        "index_id": id(df.index),
        "freq": getattr(df.index, "freq", None),
        "index_values": df.index.to_numpy(copy=True),
        "index_name": df.index.name,
        "columns": df.columns.tolist(),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "values": df.to_numpy(copy=True),
        "attrs": dict(df.attrs),
    }


@pytest.mark.parametrize("name", _indicator_names())
def test_accessor_call_leaves_the_frame_untouched(name: str) -> None:
    """``df.ta.<name>()`` reads the frame; with ``append=False`` it writes nothing.

    The sweeps above hand Series to the function directly. Users reach the same
    code through the accessor, and that is where the damage shows: ``df.ta.nvi()``
    unset ``df.index.freq`` on the caller's own frame, from a call that returns
    its result and was explicitly told not to append.
    """
    df = _build_df()
    frame = {**_build_frame(), "close": df["close"]}
    method = getattr(df.ta, name)
    before = _frame_state(df)

    method(**_accessor_kwargs(name, method, frame))

    after = _frame_state(df)
    assert after["freq"] == before["freq"], f"df.ta.{name}() cleared df.index.freq ({before['freq']} -> {after['freq']})"
    assert after["index_id"] == before["index_id"], f"df.ta.{name}() replaced the caller's index object"
    np.testing.assert_array_equal(after["index_values"], before["index_values"], err_msg=f"df.ta.{name}() rewrote the frame's index labels")
    assert after["index_name"] == before["index_name"], f"df.ta.{name}() renamed the frame's index"
    assert after["columns"] == before["columns"], f"df.ta.{name}() changed the frame's columns with append=False"
    assert after["dtypes"] == before["dtypes"], f"df.ta.{name}() changed a column dtype"
    assert after["attrs"] == before["attrs"], f"df.ta.{name}() wrote to df.attrs"
    np.testing.assert_array_equal(after["values"], before["values"], err_msg=f"df.ta.{name}() rewrote the frame's data")


@pytest.mark.parametrize("name", _indicator_names())
def test_accessor_append_only_adds_columns(name: str) -> None:
    """``append=True`` may add columns -- it may not disturb the index or the data.

    This is the path where a cleared ``freq`` survives the call and reaches the
    user's next operation, so the index is checked here too, even though the
    frame is expected to grow.
    """
    df = _build_df()
    frame = {**_build_frame(), "close": df["close"]}
    method = getattr(df.ta, name)
    before = _frame_state(df)

    method(**_accessor_kwargs(name, method, frame), append=True)

    assert getattr(df.index, "freq", None) == before["freq"], f"df.ta.{name}(append=True) cleared df.index.freq"
    np.testing.assert_array_equal(df.index.to_numpy(), before["index_values"], err_msg=f"df.ta.{name}(append=True) rewrote the frame's index labels")
    assert df.columns.tolist()[: len(before["columns"])] == before["columns"], f"df.ta.{name}(append=True) reordered or dropped the input columns"
    np.testing.assert_array_equal(
        df[before["columns"]].to_numpy(), before["values"], err_msg=f"df.ta.{name}(append=True) rewrote the data of the input columns"
    )


def test_the_accessor_sweeps_actually_compute() -> None:
    """A guard for the guard: an accessor call that quietly does nothing proves nothing.

    `core.py` turns a non-Series/DataFrame result into ``return self._df``
    without raising, so a sweep that forgets a required argument still passes
    every assertion above while testing nothing. Pin the three names whose
    result is legitimately all-NaN on this data (``acos``/``asin`` take values
    outside [-1, 1]; ``vfi``'s warmup exceeds 250 rows) and require every other
    indicator to return real numbers.
    """
    expected_empty = {"acos", "asin", "vfi"}
    identity, empty = [], []

    for name in _indicator_names():
        df = _build_df()
        frame = {**_build_frame(), "close": df["close"]}
        method = getattr(df.ta, name)
        result = method(**_accessor_kwargs(name, method, frame))
        if result is df:
            identity.append(name)
        elif result is None or not np.asarray(result.notna()).any():
            empty.append(name)

    assert not identity, f"the accessor handed the input frame back for {identity} -- the sweeps above did not run them"
    assert set(empty) == expected_empty, f"indicators with no numeric output changed: {sorted(empty)} != {sorted(expected_empty)}"
