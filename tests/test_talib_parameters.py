"""talib=True must not silently ignore a parameter TA-Lib cannot express.

wma(asc=False, talib=True) used to return TA-Lib's ascending WMA; a sweep then
found 37 more parameters in 23 indicators (scalar, drift, mamode, ddof, uo's
weights) that changed the native result but were dropped on the TA-Lib path.
Those indicators now run natively whenever such a parameter is not at its
default, so talib=True and talib=False agree on it.
"""

import inspect
import warnings

import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from tests.config import get_sample_data

pytestmark = pytest.mark.skipif(not ta.Imports["talib"], reason="needs TA-Lib")

_SERIES = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
_SKIP = {"offset", "talib", "fillna", "fill_method"}
# Tried in order; the first value that changes the native result is used.
_CANDIDATES = [2, 3, 7, 21, 0.5, 1.5, 50.0, "ema", "wma", "sma", True, False]


def _talib_indicators():
    return sorted(n for v in ta.Category.values() for n in v if "talib" in inspect.signature(_find_indicator_func(n)).parameters)


def _same(a, b):
    return pd.DataFrame(a).equals(pd.DataFrame(b))


@pytest.mark.parametrize("name", _talib_indicators())
def test_talib_true_honours_every_parameter(name):
    frame = get_sample_data().iloc[:600]
    func = _find_indicator_func(name)
    params = inspect.signature(func).parameters
    series = {p: frame[c] for p, c in _SERIES.items() if p in params}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            native_default = func(**series, talib=False)
            talib_default = func(**series, talib=True)
        except Exception:  # noqa: BLE001 - indicators that need extra inputs are out of scope
            pytest.skip("needs inputs beyond price columns")
        ignored = []
        for param, spec in params.items():
            if param in _SKIP or param in series or spec.kind in (spec.VAR_KEYWORD, spec.VAR_POSITIONAL):
                continue
            for value in _CANDIDATES:
                try:
                    native = func(**series, talib=False, **{param: value})
                except (ValueError, TypeError):
                    continue
                if _same(native, native_default):
                    continue
                talib = func(**series, talib=True, **{param: value})
                if _same(talib, talib_default):
                    ignored.append(f"{param}={value!r}")
                break
    assert not ignored, f"{name}(talib=True) ignores {ignored}"


def test_non_default_parameter_runs_natively():
    close = get_sample_data().close.iloc[:300]
    assert ta.rsi(close, scalar=50, talib=True).equals(ta.rsi(close, scalar=50, talib=False))
    assert not ta.rsi(close, talib=True).equals(ta.rsi(close, talib=False))  # defaults still use TA-Lib
